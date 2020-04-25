import json
import os

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
                

youtube_link ='https://www.youtube.com/watch?v='
def download_videos(json_file, save_dir, max_len=100000, min_len=-1, video_format='mp4', video_resolution='360p', dstfile=None):
    

if __name__ == "__main__":
    convert_txt_json("train_partition.txt", "sport1m-training_data.json")