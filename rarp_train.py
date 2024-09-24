
import torch
import sys
import os

#
# export PYTHONPATH=/home/ubuntu/Dropbox/Rezowan_codebase/dgx_code

from clean_code_dgx.models import asformer,asrf,my_asrf_asformer, mymodel
# import models.my_asrf_asformer
import torch.nn.functional as F


from datasets.rarp import create_dataframes, RARPDataset, collate_fn

# create_dataframes,RARPDataset,collate_fn
import random

from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from models import asrf
from utils.train_utils import train,train_ef,validate,evaluate,get_optimizer,get_class_weight,resume,save_checkpoint
from train.config import Config
from losses.focal_tmse import ActionSegmentationLoss,BoundaryRegressionLoss
config = Config()


sample_rate = 2
actions_dict={0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7}
num_classes = len(actions_dict)
model_dir = "/home/ubuntu/Dropbox/Rezowan_codebase/dgx_code/output/model_out/mymodel/myasformer"
results_dir = "/home/ubuntu/Dropbox/Rezowan_codebase/dgx_code/output/result/mymodel/myasformer"
root_folder= "/home/ubuntu/Dropbox/Rezowan_codebase/dgx_code/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
base_train_dir = '/home/ubuntu/Dropbox/Datasets/RARP_datasets/rarp_train_u'
video_filename ='video_left.avi'
feature_filename ='feat_2048.npy'
annot_filename ='action_continuous.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_test_dir= '/home/ubuntu/Dropbox/Datasets/RARP_datasets/rarp_test_u'
# cpu or cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cudnn.benchmark = True  # Optimizes performance for your GPU

# create dataframe
# =================================================================
train_dataframe = create_dataframes(base_train_dir,video_filename,feature_filename,annot_filename)
test_dataframe = create_dataframes(base_test_dir,video_filename,feature_filename,annot_filename)
train_dataframe.sort_values(by='frames', ascending=True)
test_dataframe.sort_values(by='frames', ascending=True)
# print('train df ',train_dataframe)
# print('test df ',test_dataframe)
batch_size=1

train_data = RARPDataset(
        train_dataframe,
        root_folder,
        num_classes,
        actions_dict,
        sample_rate=sample_rate,
    )

train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True if batch_size > 1 else False,
        collate_fn=collate_fn,
        pin_memory=True
    )

test_data = RARPDataset(
        test_dataframe[:2],
        root_folder,
        num_classes,
        actions_dict,
        sample_rate=sample_rate,
    )

val_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True if batch_size > 1 else False,
        collate_fn=collate_fn,
        pin_memory=True
    )
channel_masking_rate=0.3

# model = models.asrf.ActionSegmentRefinementFramework(
#     in_channel=2048,n_features=64,n_classes=num_classes,n_stages=4,n_stages_asb=4,n_stages_brb=4,n_layers=10,)
# model = asformer.MyTransformer(3,config.n_layers,2,2,config.n_features,config.in_channel,num_classes,0.3)

model = mymodel.MyAsformer(3,config.n_layers,config.n_features,config.in_channel,num_classes,channel_masking_rate,device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

print('Model Size: ', sum(p.numel() for p in model.parameters()))
# model= torch.nn.DataParallel(model)
optimizer = get_optimizer(config.optimizer,
        model,
        config.learning_rate,
        momentum=config.momentum,
        dampening=config.dampening,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    )

# resume if you want
columns = ["epoch", "lr", "train_loss"]

import pandas as pd
# if you do validation to determine hyperparams
if config.param_search:
    columns += ["val_loss", "cls_acc", "edit"]
    columns += [
        "segment f1s@{}".format(config.iou_thresholds[i])
        for i in range(len(config.iou_thresholds))
    ]
    columns += ["bound_acc", "precision", "recall", "bound_f1s"]

begin_epoch = 0
best_loss = float("inf")
log = pd.DataFrame(columns=columns)

result_path = results_dir
if config.resume:
    if os.path.exists(os.path.join(result_path, "checkpoint.pth")):
        checkpoint = resume(result_path, model, optimizer)
        begin_epoch, model, optimizer, best_loss = checkpoint
        log = pd.read_csv(os.path.join(result_path, "log.csv"))
        print("training will start from {} epoch".format(begin_epoch))
    else:
        print("there is no checkpoint at the result folder")


# criterion for loss
if config.class_weight:
    class_weight,boundary_weight = get_class_weight(
        num_classes,
        train_dataframe,
    )
    class_weight = class_weight.to(device)
else:
    class_weight = None

criterion_cls = ActionSegmentationLoss(
        ce=config.ce,
        focal=config.focal,
        tmse=config.tmse,
        gstmse=config.gstmse,
        weight=class_weight,
        ignore_index=-100,
        ce_weight=config.ce_weight,
        focal_weight=config.focal_weight,
        tmse_weight=config.tmse_weight,
        gstmse_weight=config.gstmse,

    )

criterion_bound = BoundaryRegressionLoss(pos_weight=boundary_weight)

# train and validate model
print("---------- Start training ----------")

for epoch in range(begin_epoch, config.max_epoch):
    # training
    train_loss = train_ef(
        train_loader,
        model,
        criterion_cls,
        criterion_bound,
        config.lambda_b,
        optimizer,
        epoch,
        device, mode="ms",test_loader=val_loader
    )
    # train_loss = train_ef(
    #     train_loader,
    #     model,
    #     criterion_cls,
    #     criterion_bound,
    #     config.lambda_b,
    #     optimizer,
    #     epoch,
    #     device, mode="ms",test_loader=val_loader
    # )
 # if you do validation to determine hyperparams
    if config.param_search:
        (
            val_loss,
            cls_acc,
            edit_score,
            segment_f1s,
            bound_acc,
            precision,
            recall,
            bound_f1s,
        ) = validate(
            val_loader,
            model,
            criterion_cls,
            criterion_bound,
            config.lambda_b,
            device,
            config.dataset,
            config.dataset_dir,
            config.iou_thresholds,
            config.boundary_th,
            config.tolerance,
        )

        # save a model if top1 acc is higher than ever
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(result_path, "best_loss_model.prm"),
            )

    # save checkpoint every epoch
    save_checkpoint(result_path, epoch, model, optimizer, best_loss)

    # write logs to dataframe and csv file
    tmp = [epoch, optimizer.param_groups[0]["lr"], train_loss]

    # if you do validation to determine hyperparams
    if config.param_search:
        tmp += [
            val_loss,
            cls_acc,
            edit_score,
        ]
        tmp += segment_f1s
        tmp += [
            bound_acc,
            precision,
            recall,
            bound_f1s,
        ]

    tmp_df = pd.Series(tmp, index=log.columns)

    # Use pd.concat instead of append
    log = pd.concat([log, tmp_df.to_frame().T], ignore_index=True)
    log.to_csv(os.path.join(result_path, "log.csv"), index=False)

    if config.param_search:
        # if you do validation to determine hyperparams
        print(
            "epoch: {}\tlr: {:.4f}\ttrain loss: {:.4f}\tval loss: {:.4f}\tval_acc: {:.4f}\tedit: {:.4f}".format(
                epoch,
                optimizer.param_groups[0]["lr"],
                train_loss,
                val_loss,
                cls_acc,
                edit_score,
            )
        )
    else:
        print(
            "epoch: {}\tlr: {:.4f}\ttrain loss: {:.4f}".format(
                epoch, optimizer.param_groups[0]["lr"], train_loss
            )
        )

# delete checkpoint
os.remove(os.path.join(result_path, "checkpoint.pth"))

# save models
torch.save(model.state_dict(), os.path.join(result_path, "final_model.prm"))

print("Done!")

