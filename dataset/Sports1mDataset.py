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
import matplotlib.pyplot as plt

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
    def __init__(self, json_file, video_root, downsample_fps = 3, max_frames = 1000, timer=None):
        with open(json_file) as f:
            self.dataset = json.load(f)

        self.videoIDs = list(self.dataset.keys())
        self.video_root = video_root
        self.max_frames = max_frames
        self.downsample_fps = downsample_fps

        self.ydl_opts = {
            'format':'133', ##mp4 240p
            'quiet':True,
            'verbose': False,
            'no_warnings': True,
            'outtmpl': f'{self.video_root}/%(id)s.%(ext)s',
        }

        self.timer = timer

    def filter_videos(self, key, filter_func):
        nested_filter_func = lambda k: filter_func(self.dataset[k][key])
        self.videoIDs = list( filter ( nested_filter_func, self.dataset.keys() ))

    def __len__(self):
        return len(self.videoIDs)

    def __getitem__(self, idx):
        currIdx = idx
        video_path = None

        if self.timer is not None: self.timer.start()

        while video_path is None:
            ytID = self.videoIDs[currIdx]
            classes = [int(x) for x in self.dataset[ytID]["classes"]] #add classes for cleaned dataset

            #download raw video
            video_path = self.download_video(ytID)

            currIdx = np.random.choice(len(self), 1)[0]

        if self.timer is not None: self.timer.lap()

        # process video
        curr_fps = 30

        video_frames = download_video_openCV(video_path, downsample_fps= self.downsample_fps)

        if video_frames.shape[0] > self.max_frames:
            sample_rate = video_frames.shape[0] // self.max_frames
            video_frames = video_frames[::sample_rate]

        if self.timer is not None: self.timer.lap()

        os.remove(video_path)
        video_frames = process_video(video_frames=video_frames, curr_fps=curr_fps, downsample_fps=None, resize_shape=(128, 171),
            clip_length_sec=2, num_clips=5, random_crop_size=(117,117), num_random_crops=16)

        video_frames = video_frames.astype(np.float32) / 255.0
    
        if self.timer is not None: self.timer.lap()

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
        
def test_time(maximum_entries = 1000):
    timer = Timer()
    dataset = Sports1mDataset("cleaned_dataset/sports1m_training_cleaned.json", "training_videos", timer=timer)

    times_set = []
    durations = []

    holdout_times_set = []
    holdout_durations = []

    for x in tqdm(range(len(dataset))):
        curr_id = dataset.videoIDs[x]

        dataset[x]
        

        if dataset.dataset[curr_id]["filesize"] is None:
            holdout_times_set.append(timer.laps)
            holdout_durations.append(dataset.dataset[curr_id]["duration"])
        else:
            times_set.append(timer.laps)
            durations.append(dataset.dataset[curr_id]["duration"])

        timer.reset()

        if (len(times_set) + len(holdout_times_set)) >= maximum_entries:
            break

    times_set = np.array(times_set)
    download, load, process = times_set[:, 0], times_set[:, 1], times_set[:, 2]

    durations = np.array(durations)

    plt.scatter(durations, download, color="red")
    plt.scatter(durations, load, color="blue")
    plt.scatter(durations, process, color="green")

    holdout_times_set = np.array(holdout_times_set)
    download, load, process = holdout_times_set[:, 0], holdout_times_set[:, 1], holdout_times_set[:, 2]

    holdout_durations = np.array(holdout_durations)

    plt.scatter(holdout_durations, download, color="red", marker='x')
    plt.scatter(holdout_durations, load, color="blue", marker='x')
    plt.scatter(holdout_durations, process, color="green", marker='x')

    plt.legend(["Download", "Load", "Process",
                "HO Download", "HO Load", "HO Process"], loc="upper left")

    plt.savefig("plots/timing_0.png")
    plt.show()
    plt.clf()

    with open("plots/count_0.txt", "w+") as f:
        f.write("Number of samples WITH    filesize: {}\n".format(len(durations)))
        f.write("Number of samples WITHOUT filesize: {}\n".format(len(holdout_durations)))

def parallel_metadata(dataset, ytID):
    info = dataset.get_video_metadata(ytID) #d.videoIDs[2])
    time.sleep(4)
    if info is not None:
        format_133 = list( filter(lambda x: int(x["format_id"]) == 133, info["formats"]) )[0]
        if format_133['filesize'] is not None:
            format_133['filesize'] = format_133['filesize'] / 10**6

        return {ytID: {"duration": info["duration"], "filesize": format_133['filesize'],
            'fps':format_133['fps'], 'height': format_133['height'], "width": format_133['width'] }}
    else:
        return None

if __name__ == "__main__":

    # ---- DATASET FILTERING TEST ---- #

    dataset = Sports1mDataset("cleaned_dataset/sports1m_training_cleaned.json", "training_videos")
    dataset.filter_videos('duration', lambda x: x <= 120)

    print(len(dataset.videoIDs))

    # ---- DATASET QUERY TIMING EXPERIMENT ---- #

    #test_time()

    #WpOLaYzj3gM
    #WEsgBn8UoeY

    # ---- DATASET META EXTRACTION WITH PARALLELIZATION ---- #

    # jsonSaver = JSONSaver("cleaned_dataset", "sports1m_training_cleaned")
    # jsonSaver.merge_all_in_folder()

    # parser = argparse.ArgumentParser(description="C3D Sports1M METADATA extractor")

    # parser.add_argument('--name', type=str, help='base file name', default='sports1m_training')
    # parser.add_argument('--divisions', type=int, help="Number of divisions", default=12)
    # parser.add_argument('--start', type=int, help='Starting division', default=0)

    # args = parser.parse_args()

    # d = Sports1mDataset("sport1m_training_data.json", "training_videos")
    # jsonSaver = JSONSaver('cleaned_dataset', args.name)

    # assert len(d.videoIDs) % args.divisions == 0, "num divisions must be evenly divisible by length"
    # divisions = args.divisions
    # samples_per_divisions = len(d.videoIDs) // args.divisions

    # start = args.start
    # for curr_div in range(start, divisions):
    #     print("STARTING EXTRACTION AT DIVISION {} / {} \n\n".format(curr_div, divisions - 1))

    #     start_ptr = curr_div * samples_per_divisions
    #     end_ptr = start_ptr + samples_per_divisions

    #     loader = tqdm(d.videoIDs[start_ptr:end_ptr])

    #     num_cores = multiprocessing.cpu_count()
    #     result = Parallel(n_jobs=num_cores, prefer="threads")(delayed(parallel_metadata)(d, ytID) for ytID in loader)

    #     # print("{} - {}".format(curr_div * samples_per_divisions, (curr_div + 1) * samples_per_divisions))

    #     print("SAVING EXTRACTION AT DIVISION {} / {} \n\n".format(curr_div, divisions - 1))
    #     jsonSaver.save_all(result)
    #     jsonSaver.reset()

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

