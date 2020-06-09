import json
import os
from pytube import YouTube

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

if __name__ == "__main__":
    convert_txt_json("train_partition.txt", "sport1m_training_data.json", "sport1m_validation_data.json", 600000)
    # download_videos("sport1m_training_data.json", "training_videos")