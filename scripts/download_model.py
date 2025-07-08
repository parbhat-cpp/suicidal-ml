import os
import sys
import kaggle

def download_model_from_kaggle():
    username = os.getenv('KAGGLE_USERNAME')
    
    os.makedirs('./models', exist_ok=True)
    
    try:
        kaggle.api.dataset_download_files(
            f"{username}/suicide-detection-model",
            path="./models",
            unzip=True,
        )
        
        for root, dirs, files in os.walk('./models'):
            for file in files:
                print(f"Downloaded file: {os.path.join(root, file)}")
    except Exception as e:
        raise Exception(e, sys)
