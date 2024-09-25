import os,torch
import pandas as pd
import numpy as np
from pathlib import Path
import torch.nn as nn
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def load_npy_features(file_path):
    return np.load(file_path)

def load_model(model,model_path,device):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # Set the model to inference mode
    return model

def load_comapare_frame_features(video_path, feature_path, annotations=None):
    # annotation_file = video_folder+'/action_continuous.txt'
    # annotations = load_rarp_annotations(annotation_file)
    # video_path = video_folder+'/video_left.avi'
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # feature_path = video_folder+'/frame_features.npy'
    video_features = load_npy_features(feature_path)

    print(f'total frames {total_frames} and video feature shape {video_features.shape}')

    return video_features

# from Datasets.rarp_dataset import create_dataframes
# from Preprocessing.extract_frame_features import load_comapare_frame_features,process_video_frames


# load pretrained models (img dino v2 )
def extract_features_dinov2(images_batch,device):

    inov2_vitl14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')

    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
    model = inov2_vitl14_reg
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    # images = [transform(Image.open(image_path).convert('RGB')) for image_path in image_paths]
    # images_batch = torch.stack(images).to(device)  # Create a batch and move to GPU
    with torch.no_grad():
        features_dict = model.module.forward_features(images_batch) #  model.forward_features(images_batch)
        features = features_dict['x_norm_patchtokens']

        # print(features.shape)
    return features



def get_video_prop(path):
    """Get properties of a video"""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return dict(fps=fps, num_frames=num_frames, height=height, width=width)



def process_video_frames(video_path,feature_extractor_func,feat_reducer,batch_size, device ,feature_filename,interval=1):
    video_dir = os.path.dirname(video_path)
    output_path= feature_filename #video_dir+feature_filename
    segmentation_folder = video_dir+'/segmentation/'
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_features = []
    frame_idx = 0
    last_mask = None  # To store the last valid mask

    if os.path.exists(segmentation_folder):

        if not os.path.isdir(segmentation_folder):
            raise ValueError(f"Segmentation folder {segmentation_folder} does not exist.")

    with tqdm(total=total_frames) as pbar:
        while frame_idx < total_frames:
        # while cap.isOpened():
            frames = []
            for _ in range(batch_size):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                # Load and apply the segmentation mask
                mask_filename = f"{frame_idx:08d}.png"
                mask_path = os.path.join(segmentation_folder, mask_filename)
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                elif last_mask is not None:
                    mask = last_mask  # Use the last valid mask
                else:
                    mask = np.ones(frame.shape[:2],
                                   dtype=np.uint8) * 255  # Default white mask if no mask file found and no previous mask

                # else:
                #     mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255  # White mask if no mask file found
                # Apply transformations to both image and mask
                transformed = transform(image=frame, mask=mask)
                img_t = transformed['image'].unsqueeze(0).to(device)
                frames.append(img_t)
                frame_idx += interval
                if frame_idx >= total_frames:
                    break


                # img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # img_t = image_transform(img).unsqueeze(0).to(device)
                # frames.append(img_t)

            if frames:
                images_batch = torch.cat(frames, dim=0)
                features = feature_extractor_func(images_batch, device)
                print('features shape', features.shape)
                reduced_feature= feat_reducer(features)
                print('reduced features shape', reduced_feature.shape)
                if isinstance(reduced_feature, torch.Tensor):
                    reduced_feature = reduced_feature.detach().cpu().numpy()
                frame_features.append(reduced_feature)
                # frame_features.append(features.cpu().numpy())
                print(f"Processed batch of {len(frames)} frames with feature shape: {features.shape}")
            pbar.update(1)
            if frame_idx >= total_frames:
                break

        cap.release()

    # Convert list of features to a single numpy array
    frame_features = np.vstack(frame_features)
    print(f"Total frame feature shape: {frame_features.shape}")

    # Save extracted features to a .npy file
    np.save(output_path, frame_features)
    # print(f"Features saved to {output_path}")
    return frame_features

class ReducedKernelConv(nn.Module):
    def __init__(self, device):
        super(ReducedKernelConv, self).__init__()
        self.device = device
        # Define a convolutional layer to reduce features
        self.conv_layer = nn.Conv2d(
            in_channels=256,   # Patch dimension becomes the input channels
            out_channels=512,  # Output channels, you can adjust this number
            kernel_size=(1, 1),  # 1x1 convolution to maintain spatial dimensions
            stride=1,          # No stride
            padding=0          # No padding
        ).to(self.device)

        # Define a linear layer for further dimensionality reduction
        self.fc_layer = nn.Linear(512 * 1024, 2048).to(self.device)  # Final feature dimension

    def forward(self, x):
        # Reshape to (batch_size, patch_dim, height=1, width=feature_dim)
        x = x.unsqueeze(2).to(self.device)  # Adds a dimension for height
        x = self.conv_layer(x)  # Apply 1x1 convolution
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layer(x)  # Apply linear layer for dimensionality reduction
        return x

def extract_features(df,pretrained_model,feature_reducer,batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_path= row['video_path']
        feat_path=row['feature_path']
        if os.path.exists(feat_path):
            print('feature path exist ')
            video_features = load_comapare_frame_features(video_path, feat_path)
        else:
            print(video_path)
            video_features = process_video_frames(video_path, pretrained_model, feature_reducer, batch_size, device,
                                                   feature_filename=feat_path,interval=1)


