import os
import json
from torch.utils.data import Dataset
import numpy as np
import skvideo.io
from pytube import YouTube
import youtube_dl

try:
    from utils.utils import *
except ImportError:
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
        print("downloading")
        while video_path is None:
            ytID = self.videoIDs[currIdx]
            classes = [int(x) for x in self.dataset[ytID]]

            #download raw video
            video_path = self.download_video(ytID)

            currIdx = np.random.choice(len(self), 1)[0]

        # process video
        curr_fps = 30

        # video_frames = skvideo.io.vread(video_path) # DOWN-SAMPLE HERE WITH OPENCV
        print("loading")
        video_frames = download_video_openCV(video_path, downsample_fps=5)
        os.remove(video_path)
        print("processing")
        video_frames = process_video(video_frames=video_frames, curr_fps=curr_fps, downsample_fps=None, resize_shape=(128, 171),
            clip_length_sec=2, num_clips=5, random_crop_size=(117,117), num_random_crops=16)

        video_frames = video_frames.astype(np.float32) / 255.0
    
        #delete raw video

        return {"video": video_frames, "class": np.random.choice(classes, 1)}

    
    # OLD DOWNLOAD
    def download_video(self, ytID):
        ydl_opts = {
            'format':'133', ##mp4 240p
            'quiet':True,
            'verbose': False,
            'outtmpl': f'{self.video_root}/%(id)s.%(ext)s',
        }

        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info('http://www.youtube.com/watch?v={}'.format(ytID), download=True)
                filename = ydl.prepare_filename(info)

            if '.webm' in filename:
                filename = filename.replace('.webm', '.mkv')

            return filename
        except:
            return None


    # def download_video(self, ytID):
    #     video_link = youtube_link + ytID
    #     try:
    #         yt = YouTube(video_link)
    #         print("got the video")
    #         stream = yt.streams.filter(only_video=True, resolution="240p", subtype='mp4').first()
    #         print("got the stream")
    #         stream.download(filename=ytID, output_path=self.video_root)

    #         return os.path.join(self.video_root, "{}.mp4".format(ytID)), stream.fps
    #     except Exception as e:
    #         print(e)
    #         return None, None
        
if __name__ == "__main__":
    # print(os.listdir())
    d = Sports1mDataset("sport1m_training_data.json", "training_videos")
    p = d[4567]
        # meta = ydl.extract_info('https://www.youtube.com/watch?v={}'.format(d.videoIDs[200]), download=False) 
        # print(meta['formats'])
    print(p["video"].shape)
    # print(os.path.dirname("YaKeaTJe04s"))

