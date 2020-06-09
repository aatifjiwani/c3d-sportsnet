import os
import json
from torch.utils.data import Dataset
import numpy as np
import skvideo.io
from pytube import YouTube
import youtube_dl
import time

import multiprocessing
from joblib import Parallel, delayed

from tqdm import tqdm 
try:
    from utils.utils import *
except ImportError:
    from dataset.utils.utils import *

from typing import List, Dict, Optional

import argparse

# youtube_link ='http://www.youtube.com/watch?v='

class JSONSaver:
    def __init__(self, root_folder, file_name, max_entries = 100000):
        self.root = root_folder
        self.file_name = file_name
        self.curr_name = file_name

        self.max_entries = max_entries
        self.entries = {}

        self.hit_max = 0

    def merge_all_in_folder(self):
        allFiles = [os.path.join(self.root, x) for x in os.listdir(self.root) if '.json' in x]
        full_dict = {}

        for dataset in allFiles:
            with open(dataset) as json_file:
                data = json.load(json_file)
                full_dict.update(data)

        filename = os.path.join(self.root, self.file_name + ".json")
        with open(filename, "w+") as json_file:
            json.dump(full_dict, json_file, indent = 4, sort_keys = False)
    
    def save_entry(self, key, value):
        self.entries[key] = value
        self.save()

    def save(self):
        if len(self.entries) >= self.max_entries:
            self.save_file()
            self.reset()

    def save_file(self):
        if len(self.entries) > 0:
            with open(os.path.join(self.root, self.curr_name + ".json"), "w+") as f:
                json.dump(self.entries, f, indent = 4, sort_keys = False)

    def save_all(self, entries: List[Optional[Dict[str, Any]]]):
        """
        Bypasses max_entries

        Entries must be a list of dictionaries, possible some entries are None
        """
        entries = list( filter(lambda x: x is not None, entries) )
        while len(entries) != 1:
            entries[0].update(entries.pop())

        self.entries = entries[0]
        self.save_file()

    def reset(self):
        self.hit_max += 1
        self.entries = {}
        self.curr_name = "{}_{}".format(self.file_name, self.hit_max)
        
class Sports1mDataset(Dataset):
    def __init__(self, json_file, video_root, max_frames = 1000):
        with open(json_file) as f:
            self.dataset = json.load(f)

        self.videoIDs = list(self.dataset.keys())
        self.video_root = video_root
        self.max_frames = max_frames

        self.ydl_opts = {
            'format':'133', ##mp4 240p
            'quiet':True,
            'verbose': False,
            'outtmpl': f'{self.video_root}/%(id)s.%(ext)s',
        }

    def __len__(self):
        return len(self.videoIDs)

    def __getitem__(self, idx):
        currIdx = idx
        video_path = None
        #print("downloading")
        while video_path is None:
            ytID = self.videoIDs[currIdx]
            classes = [int(x) for x in self.dataset[ytID]]

            #download raw video
            video_path = self.download_video(ytID)

            currIdx = np.random.choice(len(self), 1)[0]

        # process video
        curr_fps = 30

        video_frames = download_video_openCV(video_path, downsample_fps=3)

        if video_frames.shape[0] > self.max_frames:
            sample_rate = video_frames.shape[0] // self.max_frames
            video_frames = video_frames[::sample_rate]

        os.remove(video_path)
        video_frames = process_video(video_frames=video_frames, curr_fps=curr_fps, downsample_fps=None, resize_shape=(128, 171),
            clip_length_sec=2, num_clips=5, random_crop_size=(117,117), num_random_crops=16)

        video_frames = video_frames.astype(np.float32) / 255.0
    

        return {"video": video_frames, "class": np.random.choice(classes, 1)}

    def get_video_metadata(self, ytID):
        try:
            with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info('http://www.youtube.com/watch?v={}'.format(ytID), download=False)

            return info
        except:
            return None
    
    # NEW DOWNLOAD
    def download_video(self, ytID):
        try:
            with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info('http://www.youtube.com/watch?v={}'.format(ytID), download=True)
                filename = ydl.prepare_filename(info)

            if '.webm' in filename:
                filename = filename.replace('.webm', '.mkv')

            return filename
        except:
            return None

    #OLD DOWNLOAD
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
        
def parallel_metadata(dataset, ytID):
    info = d.get_video_metadata(ytID) #d.videoIDs[2])
    time.sleep(1)
    if info is not None:
        format_133 = list( filter(lambda x: int(x["format_id"]) == 133, info["formats"]) )[0]
        if format_133['filesize'] is not None:
            format_133['filesize'] = format_133['filesize'] / 10**6

        return {ytID: {"duration": info["duration"], "filesize": format_133['filesize'],
            'fps':format_133['fps'], 'height': format_133['height'], "width": format_133['width'] }}
    else:
        return None

if __name__ == "__main__":
    # jsonSaver = JSONSaver("cleaned_dataset", "sports1m_training_cleaned")
    # jsonSaver.merge_all_in_folder()

    parser = argparse.ArgumentParser(description="C3D Sports1M METADATA extractor")

    parser.add_argument('--name', type=str, help='base file name', default='sports1m_training')
    parser.add_argument('--divisions', type=int, help="Number of divisions", default=12)
    parser.add_argument('--start', type=int, help='Starting division', default=0)

    args = parser.parse_args()

    d = Sports1mDataset("sport1m_training_data.json", "training_videos")
    jsonSaver = JSONSaver('cleaned_dataset', args.name)

    assert len(d.videoIDs) % args.divisions == 0, "num divisions must be evenly divisible by length"
    divisions = args.divisions
    samples_per_divisions = len(d.videoIDs) // args.divisions

    start = args.start
    for curr_div in range(start, divisions):
        print("STARTING EXTRACTION AT DIVISION {} / {} \n\n".format(curr_div, divisions - 1))

        start_ptr = curr_div * samples_per_divisions
        end_ptr = start_ptr + samples_per_divisions

        loader = tqdm(d.videoIDs[start_ptr:end_ptr])

        num_cores = multiprocessing.cpu_count()
        result = Parallel(n_jobs=num_cores, prefer="threads")(delayed(parallel_metadata)(d, ytID) for ytID in loader)

        # print("{} - {}".format(curr_div * samples_per_divisions, (curr_div + 1) * samples_per_divisions))

        print("SAVING EXTRACTION AT DIVISION {} / {} \n\n".format(curr_div, divisions - 1))
        jsonSaver.save_all(result)
        jsonSaver.reset()

    
    # loader = tqdm(d.videoIDs)

    # num_cores = multiprocessing.cpu_count()
    # result = Parallel(n_jobs=num_cores, prefer="threads")(delayed(parallel_metadata)(d, ytID) for ytID in loader)

    # jsonSaver.save_all(result)

    # ---- DATASET META EXTRACTION WITHOUT PARALLELIZATION ---- #

    # num_not_found = 0
    # loader.set_description("Videos not found: {}".format(num_not_found))

    # for ytID in loader:
    #     info = d.get_video_metadata(ytID) #d.videoIDs[2])

    #     if info is not None:
    #         format_133 = list( filter(lambda x: int(x["format_id"]) == 133, info["formats"]) )[0]
    #         if format_133['filesize'] is not None:
    #             format_133['filesize'] = format_133['filesize'] / 10**6

    #         jsonSaver.save_entry(ytID, {"duration": info["duration"], "filesize": format_133['filesize'],
    #             'fps':format_133['fps'], 'height': format_133['height'], "width": format_133['width'] })
    #     else:
    #         num_not_found += 1
            
    #     loader.set_description("Videos not found: {}".format(num_not_found))
    #     time.sleep(0.5) ##PREVENTS 429 ERROR. 
    
    # jsonSaver.save_file()

