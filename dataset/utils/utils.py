#  Some code produced by David Chan @ UC Berkeley 
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import math

def download_video_openCV(video_path: str, downsample_fps: int = None) -> np.ndarray:
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        raise ValueError("{} is an invalid video".format(video_path))

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if downsample_fps is not None:
        fps_factor = int(np.ceil(fps / downsample_fps))
    
    frames = []
    curr_frame = -1

    ret, frame = cap.read()
    while ret:
        curr_frame += 1
        if downsample_fps is not None:
            if (curr_frame % fps_factor) != 0:
                ret, frame = cap.read()
                continue

        frames.append(frame) 

    cap.release()
    
    video_frames = np.stack(frames, axis=0)
    return video_frames

def downsample_video_fps(video_frames: np.ndarray, current_fps: int, target_fps: int) -> np.ndarray:
    if current_fps <= target_fps:
        raise AssertionError('The target FPS of a video should be less than the current FPS when downsampling')
    return video_frames[np.floor(np.arange(0, video_frames.shape[0], current_fps / target_fps)[:-1]).astype(np.int32)]

def resize_video(video_frames: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    assert len(video_frames.shape) == 4, 'Video should have shape [N_Frames x H x W x C]'
    
    output_array = np.zeros((video_frames.shape[0], target_size[0], target_size[1], video_frames.shape[3],))
    for i in range(video_frames.shape[0]):
        output_array[i] = cv2.resize(video_frames[i], (target_size[1], target_size[0]))
    return output_array

def random_clip(video_frames: np.ndarray, curr_fps: int, clip_length: int) -> np.ndarray:
    assert len(video_frames.shape) == 4, "video frames must be shape N_frame x H x W x C"

    video_length = video_frames.shape[0]

    
    try:
        start = np.random.choice(np.arange(0, video_length - clip_length, 1), 1)[0]
    except:
        print('ERROR')
        print(clip_length)
        print(video_length)

    return video_frames[start:start+clip_length]

def random_crop(video_frame: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    assert len(video_frame.shape) == 3, "video frame must be shape H x W x C"

    crop_height, crop_width = crop_size

    max_y = video_frame.shape[0] - crop_height
    max_x = video_frame.shape[1] - crop_width

    start_y = np.random.randint(0, max_y)
    start_x = np.random.randint(0, max_x)

    crop = video_frame[start_y : start_y + crop_height, start_x : start_x + crop_width, :]
    return crop

def process_video(video_frames: np.ndarray, curr_fps: int = 30, downsample_fps: int = None, resize_shape: Tuple[int, int] = None,
    clip_length_sec: int = None, num_clips: int = None, random_crop_size: Tuple[int, int] = None, num_random_crops: int = None) -> np.ndarray:

    #downsample
    if downsample_fps is not None:
        video_frames = downsample_video_fps(video_frames, curr_fps, downsample_fps)
        curr_fps = downsample_fps

    #resize
    if resize_shape is not None:
        video_frames = resize_video(video_frames, resize_shape)

    #random clips
    if clip_length_sec is not None and num_clips is not None:
        rand_clips = []
        clip_length = clip_length_sec * curr_fps
        if (clip_length < video_frames.shape[0]):
            for _ in range(num_clips):
                rand_clips.append( random_clip(video_frames, curr_fps, clip_length) ) #num_frame, H, W, C

            video_frames = np.concatenate(rand_clips, axis=0) #clip_len*num_clips, H, W, C

    #random crops
    if random_crop_size is not None and num_random_crops is not None:
        rand_crops = []

        for _ in range(num_random_crops):
            frame_num = np.random.randint(0, video_frames.shape[0] - 1)

            rand_crops.append( random_crop(video_frames[frame_num], random_crop_size) ) #H, W, C

        video_frames = np.stack(rand_crops, axis=0)

    return video_frames
    
        
        



    









