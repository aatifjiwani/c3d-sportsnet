import os
import json
from torch.utils.data import Dataset
import numpy as np
import skvideo.io
from pytube import YouTube
from dataset.utils.utils import *

youtube_link ='http://www.youtube.com/watch?v='

class Sports1mDataset(Dataset):
    def __init__(self, json_file, video_root, subsample=25, max_frames = 500):
        with open(json_file) as f:
            self.dataset = json.load(f)

        self.videoIDs = list(self.dataset.keys())
        self.video_root = video_root
        self.max_frames = max_frames
        self.subsample = subsample

    def __len__(self):
        return len(self.videoIDs)

    def __getitem__(self, idx):
        currIdx = idx
        video_path = None
        while video_path is None:
            ytID = self.videoIDs[currIdx]
            classes = [int(x) for x in self.dataset[ytID]]
            print("retrieving", ytID)

            #download raw video
            video_path, curr_fps = self.download_video(ytID)

            currIdx = np.random.choice(len(self), 1)[0]

        #process video
        curr_fps = curr_fps if curr_fps is not None else 30

        video_frames = skvideo.io.vread(video_path)
        video_frames = downsample_video_fps(video_frames, curr_fps, 5) # downsample video
        video_frames = resize_video(video_frames, (224, 224))[:self.max_frames] #resize video and cut to max_frames

        video_frames = video_frames.astype(np.float32) / 255.0
        if video_frames.shape[0] != self.max_frames:
            video_frames = np.pad(video_frames[:self.max_frames], ((0,self.max_frames - video_frames.shape[0]),(0,0),(0,0),(0,0)))
    
        #delete raw video
        os.remove(video_path)

        return {"video": video_frames, "class": np.random.choice(classes, 1)}

    
    def download_video(self, ytID):
        video_link = youtube_link + ytID
        try:
            yt = YouTube(video_link)
            stream = yt.streams.filter(only_video=True, resolution="240p", subtype='mp4').first()

            stream.download(filename=ytID, output_path=self.video_root)

            return os.path.join(self.video_root, "{}.mp4".format(ytID)), stream.fps
        except:
            return None, None
        
if __name__ == "__main__":
    # print(os.listdir())
    d = Sports1mDataset("sport1m_training_data.json", "training_videos")
    print(len(d))
    vid = d[6000]["video"]
    print(vid.shape)
