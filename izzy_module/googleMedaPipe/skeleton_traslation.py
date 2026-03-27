import re
import cv2
import os
import shutil
from database import DataBase
from SkeletonExtractor import SkeletonExtractor
import projectPoints as projectPoints


dbPath = os.path.join("../dataSet/wlasl-complete")

base_dir = "./savedVideoPoints"
# Remove everything inside base_dir
if os.path.exists(base_dir):
    print(f"Clearing {base_dir} ...")
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)   # Deletes folder and all contents
        else:
            os.remove(item_path)       # Deletes file



database_video_path = os.path.join(dbPath, "videos")
video_index_path = os.path.join(dbPath, "wlasl_class_list.txt")

# Initialize your database
db = DataBase(database_path=dbPath, video_folder=database_video_path)


# Build missing words and word-to-video dictionary
db.build_dictionary(word_to_video_path=video_index_path, video_folder_path=database_video_path)

# Input sentence
input_sentence = input("Enter a sentence: ")

# Split sentence into words
split_sentence = re.split(r'[;,\s]+', input_sentence.strip())

# Process each word
for word in split_sentence:
    video_path = db.get_video_path(word)

    if word in db.missing_words or video_path is None:
        print(f"Warning: The word '{word}' is marked as missing or has no video.")
        continue
    else:
        
        
        os.makedirs(os.path.join(base_dir, word), exist_ok=True)
        # Recreate subfolders
        os.makedirs(os.path.join(base_dir, word+"/pose"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, word+"/hands"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, word+"/combined"), exist_ok=True)
        
    

    print(f"Playing video for word: '{word}' -> {video_path}")
    
    skeleton_extractor = SkeletonExtractor(
        video_source=video_path,
        run_face=False,
        run_hands=True,
        run_pose=True,
        save_hand_coords=True,
        save_pose_coords=True,
        draw_skeleton=True,
        show_video=True,
        no_play=False,
        skeleton_only=True,       # if you want black background with skeleton
        base_dir=("./savedVideoPoints/" + word)
        )
    skeleton_extractor.run()
    
for word in split_sentence:
    projectPoints.run("./savedVideoPoints/"+word)
    
