import os
import torch
import pandas as pd
import models.asrf
from datasets.rarp import create_dataframes,RARPDataset,collate_fn,RARPBatchGenerator
import random

from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from models import asrf
from utils.train_utils import train,validate,evaluate,get_optimizer,get_class_weight,resume,save_checkpoint
from train.config import Config
from losses.focal_tmse import ActionSegmentationLoss,BoundaryRegressionLoss
config= Config()


seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
num_stages = 4
num_layers = 10
num_f_maps = 64
features_dim = 2048
num_layers_PG = 11
num_layers_R = 10
num_R = 3
bz = 1
lr = 0.0005
num_epochs = 50
# use the full temporal resolution @ 15fpsf
sample_rate = 1
actions_dict={0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7}
num_classes = len(actions_dict)
model_dir = "/home/ubuntu/Dropbox/Rezowan_codebase/dgx_code/output/model_out/asrf_Trx"
results_dir = "/home/ubuntu/Dropbox/Rezowan_codebase/dgx_code/output/result/asrf_Trx"
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
all= [train_dataframe,test_dataframe]
dataframes= pd.concat(all)
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
from models.my_asrf_asformer import Trainer
num_epochs = 120

lr = 0.0005
num_layers = 11
num_f_maps = 64
features_dim = 2048
bz = 1

channel_mask_rate = 0.3
# if args.action == "train":
batch_gen = RARPBatchGenerator(dataframes,num_classes, actions_dict, sample_rate)
# batch_gen.read_data(vid_list_file)
# batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
# batch_gen_tst.read_data(vid_list_file_tst)
batch_gen_test= RARPBatchGenerator(test_dataframe,num_classes, actions_dict, sample_rate)


trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate)
trainer.train(model_dir, batch_gen, num_epochs, bz, lr,batch_gen_test)