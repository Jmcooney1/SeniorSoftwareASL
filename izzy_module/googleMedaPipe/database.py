import os
import re


class DataBase:

    def __init__(self, database_path: str, video_folder: str):

        if not os.path.exists(database_path):
            raise ValueError("Base path does not exist.")

        if not os.path.isdir(database_path):
            raise ValueError("Base path must be a directory.")
        
        if not os.path.exists(video_folder):
            raise ValueError("Video folder does not exist.")
        
        if not os.path.isdir(video_folder):
            raise ValueError("Video folder must be a directory.")
        
        

        self.database_path = database_path
        self.video_folder = video_folder
        self.word_to_path = {}
        self.missing_words = set()

    def build_missing_set(self, missing_list_path: str):
        try:
            with open(missing_list_path, 'r', encoding='utf8') as file:
                for line in file:
                    self.missing_words.add(line.strip())

        except FileNotFoundError:
            print(f"Error: The file '{missing_list_path}' was not found.")
        except Exception as e:
            print(f"An error occurred while building missing set: {e}")

    def build_dictionary(self, word_to_video_path: str, video_folder_path: str):

        try:
            with open(word_to_video_path, 'r', encoding='utf8') as file:
                for line in file:
                    print(line)
                    result = re.split(r'[;,\s]+', line.strip())
                    video_index = result[0]
                    
                    video_word = result[1]
                    
                    
                    while len(video_index) < 5:
                        video_index = '0' + video_index
                    print(video_index)
                    path_to_video = os.path.join(video_folder_path, f"{video_index}.mp4")

                    if os.path.exists(path_to_video):
                        self.word_to_path[video_word] = video_index
                    else:
                        self.missing_words.add(video_word)

            print(
                f"Dictionary built successfully. "
                f"{len(self.word_to_path)} entries added. "
                f"{len(self.missing_words)} missing words."
            )

        except FileNotFoundError:
            print(f"Error: The file '{word_to_video_path}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_video_path(self, word: str) -> str:
        
        if word in self.missing_words:
            print(f"Warning: The word '{word}' is marked as missing.")
            return None

        video_index = self.word_to_path.get(word)

        if video_index is None:
            print(f"Warning: The word '{word}' is not found in the dictionary.")
            return None

        return os.path.join(self.video_folder, f"{video_index}.mp4")
        