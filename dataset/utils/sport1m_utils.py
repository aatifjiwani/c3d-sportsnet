import json
import os
from pytube import YouTube

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

def convert_txt_json(srcFile, trainFile, valFile, trainPartition):
    with open(srcFile) as src:
        with open(trainFile, "w+") as trainF:
            with open(valFile, "w+") as valF:
                train_dict = {}
                val_dict = {}
                i = 0
                for line in src:
                    youtubeURL, classes = line.strip().split(None, 1) 
                    uniqueID = youtubeURL[youtubeURL.index("=")+1:]
                    
                    if ("," in classes):
                        classes = classes.split(",")
                    else:
                        classes = [classes]

                    if i < trainPartition:
                        train_dict[uniqueID] = classes
                    else:
                        val_dict[uniqueID] = classes

                    i+=1
                
                json.dump(train_dict, trainF, indent = 4, sort_keys = False) 
                json.dump(val_dict, valF, indent = 4, sort_keys = False)
                


def download_videos(json_file, save_dir, max_len=100000, min_len=-1, video_format='mp4', video_resolution='360p'):
    data_json = None
    with open(json_file) as f:
        data_json = json.load(f)
    
    print("finished reading data...")

    for i, youtubeID in enumerate(data_json):
        classes = data_json[youtubeID]
        
        video_link = youtube_link + youtubeID
        try:
            print("downloading {}".format(video_link))
            yt = YouTube(video_link)
            stream = yt.streams.filter(only_video=True, resolution="360p", subtype='mp4').first()
            stream.download(filename=youtubeID, output_path="training_videos")
        except Exception as e:
            print(e)
            print ('Cannot download YouTube video {}'.format(video_link))
            break
        
        break

class Sport1mMetaExtractor:
    def __init__(self, dataset):
        with open(dataset) as json_file:
            self.dataset = json.load(json_file)

    def getDurations(self):
        durations = [self.dataset[k]["duration"] for k in self.dataset.keys()]
        return durations
    
    def plotDurationsHistogram(self, interval:int = 60, maximum_secs: int = 1860):
        durations = np.array(self.getDurations())
        bins = np.arange(0, maximum_secs + interval, interval)

        fig, ax = plt.subplots(figsize=(20, 5))
        n, bins, patches = plt.hist(np.clip(durations, bins[0], bins[-1]), density=False, bins=bins)

        plt.xlim([0, maximum_secs])
        plt.xticks(bins)
        ax.set_xticklabels([b // 60 for b in bins])

        print("BIN COUNTS: \n", list(zip(n, bins)))
        plt.show()

    def getFPS(self):
        fpss = [self.dataset[k]["fps"] for k in self.dataset.keys()]
        return fpss
    
    def plotFPSHistogram(self, interval:int = 1, maximum_fps: int = 32):
        fpss = np.array(self.getFPS())
        print(max(fpss))
        bins = np.arange(0, maximum_fps + interval, interval)
    
        fig, ax = plt.subplots(figsize=(20, 5))
        n, bins, patches = plt.hist(fpss, density=False, bins=bins)

        plt.xlim([0, maximum_fps])
        plt.xticks(bins)
        ax.set_xticklabels([b for b in bins])

        print("BIN COUNTS: \n", list(zip(n, bins)))
        plt.show()

    def getFilesize(self):
        sizes = [x for x in [self.dataset[k]["filesize"] for k in self.dataset.keys()] if x is not None]
        return sizes

    def plotFilesizeHistogram(self, interval:int = 5, maximum_size: int = 201):
        sizes = np.array(self.getFilesize())
        bins = np.arange(0, maximum_size + interval, interval)
    
        fig, ax = plt.subplots(figsize=(20, 5))
        n, bins, patches = plt.hist(np.clip(sizes, bins[0], bins[-1]), density=False, bins=bins)

        plt.xlim([0, maximum_size])
        plt.xticks(bins)
        ax.set_xticklabels(bins)

        print("BIN COUNTS: \n", list(zip(n, bins)))
        plt.show()

    def getSpatialDimensions(self):
        h = [self.dataset[k]["height"] for k in self.dataset.keys()]
        w = [self.dataset[k]["width"] for k in self.dataset.keys()]

        return h,w

    def describeSpatialDimensions(self,):
        h, w = self.getSpatialDimensions()
        print("HEIGHT SUMMARY:", stats.describe(h))
        print("WIDTH SUMMARY:", stats.describe(w))


if __name__ == "__main__":
    # run as python utils/sport1m_utils.py 
    mx = Sport1mMetaExtractor("cleaned_dataset/sports1m_training_cleaned.json")

    # mx.plotDurationsHistogram()
    # mx.plotFPSHistogram()
    # mx.plotFilesizeHistogram()
    # mx.plotFilesizeHistogram()
    mx.describeSpatialDimensions()


    # convert_txt_json("train_partition.txt", "sport1m_training_data.json", "sport1m_validation_data.json", 600000)
    # download_videos("sport1m_training_data.json", "training_videos")