import numpy as np
import pandas as pd

from clean_code_dgx.libs.utils.io import get_video_prop
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset,DataLoader
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import torch,os
import torch.nn.functional as F

cholec_root= "/home/ubuntu/Dropbox/Datasets/cholec80"

PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderRetraction",
    "CleaningCoagulation",
    "GallbladderPackaging"
]

action_dict= {"Preparation":0,
    "CalotTriangleDissection":1,
    "ClippingCutting":2,
    "GallbladderDissection":3,
    "GallbladderRetraction":4,
    "CleaningCoagulation":5,
    "GallbladderPackaging":6}

INSTRUMENTS = [
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag"
]

def get_base_name(file):
    return file.split('-timestamp')[0].split('.')[0]


cholec_root = "/home/local/data/rezowan/datasets/cholec/"

action_dict = {"Preparation": 0,
               "CalotTriangleDissection": 1,
               "ClippingCutting": 2,
               "GallbladderDissection": 3,
               "GallbladderRetraction": 4,
               "CleaningCoagulation": 5,
               "GallbladderPackaging": 6}


def calculate_segments(phase_file, action_dict, sample_rate=1):
    segments = []  # List to hold the segments
    start_frame = 0  # Starting frame for the first segment
    prev_phase = None  # Previous phase to track changes

    # if dataset=="cholec":
    with open(phase_file, "r") as f:
        gt = f.read().split("\n")[1:-1]
    # Total number of frames
    num_frames = len(gt)

    # Create effective frame indices based on the sample rate
    effective_indices = np.arange(0, num_frames, sample_rate)
    effective_length = len(effective_indices)

    # Create arrays to store ground truth phase information and boundaries
    gt_array = np.full(effective_length, -100)
    boundary_array = np.zeros(effective_length)

    for i in range(num_frames):
        frame_data = gt[i].split("\t")
        current_frame = int(frame_data[0])  # Frame number
        current_phase = frame_data[-1]  # Phase name
        if current_frame in effective_indices:
            effective_index = np.where(effective_indices == current_frame)[0][0]

            # Process only if the current phase is in the action dictionary
            if current_phase in action_dict:
                gt_array[effective_index] = action_dict[current_phase]

                # Track phase changes
                if prev_phase is None:
                    # Initialize the previous phase
                    prev_phase = current_phase
                    start_frame = effective_index  # Start segment at the first frame

                elif current_phase != prev_phase:
                    # When phase changes, record the previous phase segment
                    end_frame = effective_index - 1

                    # When phase changes, record the previous phase segment
                    segments.append({
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'phase': prev_phase,
                        'action': action_dict[prev_phase]
                    })

                    # Define boundary conditions: First 5 frames as start, last 5 frames as end
                    boundary_array[start_frame:start_frame + 5] = 1  # Start boundary
                    boundary_array[end_frame - 4:end_frame + 1] = 2  # End boundary

                    # All intermediate frames between start and end are refined (mark as 0)
                    boundary_array[start_frame + 5:end_frame - 4] = 0

                    start_frame = effective_index
                    prev_phase = current_phase

    # After the loop, handle the last active segment
    if prev_phase is not None:
        end_frame = effective_index

        segments.append({
            'start_frame': start_frame,
            'end_frame': end_frame,
            'phase': prev_phase,
            'label': action_dict[prev_phase]
        })

        # Define boundary for the last segment
        boundary_array[start_frame:start_frame + 5] = 1  # Start boundary
        boundary_array[end_frame - 4:end_frame + 1] = 2  # End boundary
        if end_frame - 4 >= start_frame + 5:
            boundary_array[start_frame + 5:end_frame - 4] = 0

    return segments, gt_array, boundary_array


def create_cholec_df(cholec_root, action_dict, sample_rate=1):
    phases_path = os.path.join(cholec_root, 'phase_annotations')
    tools_path = os.path.join(cholec_root, 'tool_annotations')
    features_path = os.path.join(cholec_root, 'features')
    videos_path = os.path.join(cholec_root, 'videos')
    # for folds in folders:
    phase_list = os.listdir(phases_path)
    tool_list = os.listdir(tools_path)
    feature_list = os.listdir(features_path)
    video_list = [f for f in os.listdir(videos_path) if f.endswith('.mp4')]
    phase_list.sort()
    tool_list.sort()
    feature_list.sort()
    video_list.sort()
    feat_files = []
    phase_files = []
    tool_files = []
    segments = []
    labels = []
    boundaries = []

    video_files = []
    for feat in zip(feature_list, tool_list, phase_list, video_list):
        base_name = feat[0].split('.')[0]
        if feat[1].startswith(base_name) and feat[2].startswith(base_name):
            phase_file = os.path.join(phases_path, feat[2])
            tool_file = os.path.join(phases_path, feat[1])
            feat_file = os.path.join(phases_path, feat[0])
            vid_file = os.path.join(phases_path, feat[3])
            segment, gt, bound = calculate_segments(phase_file, action_dict, sample_rate=1)
            segments.append(segment)
            labels.append(gt)
            boundaries.append(bound)
            feat_files.append(feat_file)
            phase_files.append(phase_file)
            tool_files.append(tool_file)
            video_files.append(vid_file)

    # Create a dataframe with the relevant paths
    df = pd.DataFrame({
        'video_file': video_files,
        'annotation_path': phase_files,
        'tool_annotation': tool_files,
        'feature_path': feat_files,
        'segments': segments,
        'labels': labels,
        'boundaries': boundaries,

    })
    return df


cholec_df = create_cholec_df(cholec_root, action_dict, sample_rate=1)
