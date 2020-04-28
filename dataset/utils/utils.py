#  Code produced by David Chan @ UC Berkeley 
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

def downsample_video_fps(video_frames: np.ndarray, current_fps: int, target_fps: int) -> np.ndarray:
    if current_fps <= target_fps:
        raise AssertionError('The target FPS of a video should be less than the current FPS when downsampling')
    return video_frames[np.floor(np.arange(0, video_frames.shape[0], current_fps / target_fps)[:-1]).astype(np.int32)]

def resize_video(video_frames: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    assert len(video_frames.shape) == 4, 'Video should have shape [N_Frames x H x W x C]'
    
    output_array = np.zeros((video_frames.shape[0], target_size[0], target_size[1], video_frames.shape[3],))
    for i in range(video_frames.shape[0]):
        output_array[i] = cv2.resize(video_frames[i], target_size)
    return output_array