import os
import sys

def download_dataset(dataset_name="piotrstefaskiue/poland-vehicle-license-plate-dataset"):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    kaggle_dir = os.path.join(project_root, '.kaggle')

    if not os.path.exists(kaggle_dir):
        print(f"Folder {kaggle_dir} does not exist! Put kaggle.json there.")
        sys.exit(1)

    os.environ['KAGGLE_CONFIG_DIR'] = kaggle_dir

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print("[INFO] Downloading dataset...")
    api.dataset_download_files(dataset_name, path=os.path.join(project_root, 'data', 'raw'), unzip=True)
    print("[INFO] Dataset downloaded and extracted.")

if __name__ == "__main__":
    download_dataset()
