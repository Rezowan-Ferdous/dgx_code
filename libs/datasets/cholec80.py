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

def make_cholecdf(cholec_root):
    video_files = []
    phase_annotations = []
    tool_annotations = []
    feature_files = []
    num_frames = []
    fps = []
    videos_path = os.path.join(cholec_root, 'videos')
    phases_path = os.path.join(cholec_root, 'phase_annotations')
    tools_path = os.path.join(cholec_root, 'tool_annotations')
    features_path = os.path.join(cholec_root, 'features')


    video_list = [f for f in os.listdir(videos_path) if f.endswith('.mp4')]
    phase_list= os.listdir(phases_path)
    tool_list= os.listdir(tools_path)
    feature_list= os.listdir(features_path)

    # Match video names with the corresponding annotations and features
    for video_file in video_list:
        base_name = get_base_name(video_file)

        # Find corresponding phase annotation, tool annotation, and feature files
        phase_file = next((f for f in phase_list if get_base_name(f) == base_name), None)

        # Initialize variables
        segments = []  # List to hold the segments
        start_frame = 0  # Starting frame for the first segment
        prev_phase = None  # Previous phase to track changes

        with open(phase_file, "r") as f:
            gt = f.read().split("\n")[1:-1]
        gt_array = np.full(len(gt), -100)
        for i in range(len(gt)):
            # gt_array[i] = action_dict[gt[i].split("\t")[-1]]
            # print(gt[i], action_dict[gt[i]], gt_array[i])
            # Get the frame and phase from the line
            frame_data = gt[i].split("\t")
            current_frame = int(frame_data[0])  # Frame number
            current_phase = frame_data[-1]  # Phase name

            # Map the phase to its corresponding action index
            if current_phase in action_dict:
                gt_array[i] = action_dict[current_phase]

                # Track phase changes
                if prev_phase is None:
                    # Initialize the previous phase
                    prev_phase = current_phase
                    start_frame = current_frame  # Start segment at the first frame
                elif current_phase != prev_phase:
                    # When phase changes, record the previous phase segment
                    segments.append({
                        'start_frame': start_frame,
                        'end_frame': current_frame - 1,
                        'phase': prev_phase,
                        'action': action_dict[prev_phase]
                    })
                    # Reset start frame and previous phase
                    start_frame = current_frame
                    prev_phase = current_phase

            # After loop, check if there's an active segment to close
        if prev_phase is not None:
            segments.append({
                'start_frame': start_frame,
                'end_frame': current_frame,  # Last frame processed
                'phase': prev_phase,
                'action': action_dict[prev_phase]
            })

        tool_file = next((f for f in tool_list if get_base_name(f) == base_name), None)
        feature_file = next((f for f in feature_list if get_base_name(f) == base_name), None)

        # video_prop= get_video_prop(video_file)
        # num_frame= video_prop['num_frames']
        # fp=video_prop['fps']
        # num_frames.append(num_frame)
        # fps.append(fp)
        # Append to the lists only if all corresponding files are found
        if phase_file and tool_file and feature_file:
            video_files.append(os.path.join('videos', video_file))
            phase_annotations.append(os.path.join('phase_annotations', phase_file))
            tool_annotations.append(os.path.join('tool_annotations', tool_file))
            feature_files.append(os.path.join('features', feature_file))

    # Create a dataframe with the relevant paths
    df = pd.DataFrame({
        'video_file': video_files,
        'phase_annotation': phase_annotations,
        'tool_annotation': tool_annotations,
        'feature_file': feature_files,
        # 'num_frames':num_frames,
        # 'fps':fps,

    })
    print(df.head())
    return df

make_cholecdf(cholec_root)
