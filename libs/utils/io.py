import cv2
import numpy as np


def get_video_prop(path):
    """Get properties of a video"""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return dict(fps=fps, num_frames=num_frames, height=height, width=width)



def load_numpy(fpath):
    return np.load(fpath)

def write_numpy(fpath, value):
    return np.save(fpath, value)

def load_pickle(fpath):
    return pickle.load(open(fpath, 'rb'))

def write_pickle(fpath, value):
    return pickle.dump(value, open(fpath, 'wb'))