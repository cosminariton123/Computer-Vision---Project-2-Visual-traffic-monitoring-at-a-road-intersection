import cv2
import numpy as np
from tqdm import tqdm

from consts import frames_key, rectangles_key

def process_one_video(video_path, background, query, visualise=False):
    
    in_video = cv2.VideoCapture(video_path)

    frames = query[frames_key]

    
