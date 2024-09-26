
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
from utils.train_utils import train,train_ef,validate,evaluate,get_optimizer,get_class_weight,resume,save_checkpoint,get_class_weight_u,EarlyStopping
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
test_dataframe = create_dataframes(base_test_dir,video_filename,feature_filename,annot_filename)
test_dataframe.sort_values(by='frames', ascending=True)
# print('train df ',train_dataframe)
# print('test df ',test_dataframe)
batch_size=1
test_data = RARPDataset(
        test_dataframe[:4],
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
channel_masking_rate= 0.3
downsamp_rate = 2

model = mymodel.MyAsformer(3,config.n_layers,config.n_features,config.in_channel,num_classes,channel_masking_rate,device)

model_outpu_folder= '/home/ubuntu/Dropbox/Rezowan_codebase/dgx_code/output/model_out/TCN/epoch-100.model'
model_param= '/home/ubuntu/Dropbox/Rezowan_codebase/dgx_code/output/model_out/TCN/epoch-100.opt'
result_dir= '/home/ubuntu/Dropbox/Rezowan_codebase/dgx_code/output/result/mymodel/myasformer'