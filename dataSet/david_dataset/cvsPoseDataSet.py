import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
landmarkPath = os.path.join(APP_DIR, "landmark")
class CVSPoseDataSet:

    def __init__(self, dataset_path: str | None = None):
        if dataset_path is None:
            dataset_path = landmarkPath

        if not os.path.exists(dataset_path):
            raise ValueError("Dataset path does not exist.")
        
        if not os.path.isdir(dataset_path):
            raise ValueError("Dataset path must be a directory.")
        
        self.landmarkPath = dataset_path
        self.name_to_csv = {}

    def build_dictionary(self):
        for file in os.listdir(self.landmarkPath):
            if not file.endswith(".csv"):
                continue

            full_path = os.path.join(self.landmarkPath , file)

            if not os.path.isfile(full_path):
                continue

            name = file
            name = (
                file
                .replace("SignSchool ", "")
                .replace(" [1920x1080]", "")
                .strip()
                .replace(" ", "_")
            )
            self.name_to_csv[name] = full_path
            
    def get_pose_csv(self, name: str) -> str:
        if name not in self.name_to_csv:
            raise ValueError(f"Name '{name}' not found in dataset.")
        return self.name_to_csv[name]