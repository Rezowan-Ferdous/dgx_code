import os
import torch
import sys

import sys
import os

# Print the Python path for debugging
print("Python path:", sys.path)

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Print the updated Python path
print("Updated Python path:", sys.path)

# Now import from the datasets package
from datasets.rarp import create_dataframes, RARPDataset, collate_fn

from models import asformer,asrf,my_asrf_asformer
# import models.my_asrf_asformer

from datasets.rarp import create_dataframes, RARPDataset, collate_fn

# create_dataframes,RARPDataset,collate_fn
import random

from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from models import asrf
from utils.train_utils import train,validate,evaluate,get_optimizer,get_class_weight,resume,save_checkpoint
from train.config import Config
from losses.focal_tmse import ActionSegmentationLoss,BoundaryRegressionLoss
config= Config()


sample_rate = 1
actions_dict={0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7}
num_classes = len(actions_dict)
model_dir = "/home/ubuntu/Dropbox/Rezowan_codebase/dgx_code/output/model_out/Trx"
results_dir = "/home/ubuntu/Dropbox/Rezowan_codebase/dgx_code/output/result/Trx"
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
    torch.backends.cudnn.benchmark = True

# create dataframe
# =================================================================
train_dataframe = create_dataframes(base_train_dir,video_filename,feature_filename,annot_filename)
test_dataframe = create_dataframes(base_test_dir,video_filename,feature_filename,annot_filename)
# print('train df ',train_dataframe)
# print('test df ',test_dataframe)
batch_size=1
num_workers=0
train_data = RARPDataset(
        train_dataframe[40:],
        root_folder,
        num_classes,
        actions_dict,
        sample_rate=1,
    )

train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True if batch_size > 1 else False,
        collate_fn=collate_fn,
    )

test_data = RARPDataset(
        test_dataframe[:3],
        root_folder,
        num_classes,
        actions_dict,
        sample_rate=1,
    )

val_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True if batch_size > 1 else False,
        collate_fn=collate_fn,
    )

# model = models.asrf.ActionSegmentRefinementFramework(
#     in_channel=2048,n_features=64,n_classes=num_classes,n_stages=4,n_stages_asb=4,n_stages_brb=4,n_layers=10,)
model = models.my_asrf_asformer.MyTransformer(3,10,2,2,64,2048,num_classes,0.3,device)

optimizer= get_optimizer(config.optimizer,
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

result_path = os.path.exists(results_dir)
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
        ignore_index=255,
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
    train_loss = train(
        train_loader,
        model,
        criterion_cls,
        criterion_bound,
        config.lambda_b,
        optimizer,
        epoch,
        device,
    )
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

    log = log.append(tmp_df, ignore_index=True)
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

