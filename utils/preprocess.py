import os
import json
import pandas as pd
import cv2

@staticmethod
def _get_video_prop(path):
    # print(path)
    """Get properties of a video"""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return dict(fps=fps, num_frames=num_frames, height=height, width=width)
