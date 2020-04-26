import json
import os
from pytube import YouTube

def convert_txt_json(srcFile, destFile):
    with open(srcFile) as src:
        with open(destFile, "w+") as dest:
            save_dict = {}
            for line in src:
                youtubeURL, classes = line.strip().split(None, 1) 
                uniqueID = youtubeURL[youtubeURL.index("=")+1:]
                
                if ("," in classes):
                    classes = classes.split(",")
                else:
                    classes = [classes]

                save_dict[uniqueID] = classes
            
            json.dump(save_dict, dest, indent = 4, sort_keys = False) 
                


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
    # convert_txt_json("train_partition.txt", "sport1m_training_data.json")
    download_videos("sport1m_training_data.json", "training_videos")