'''This program will create and train a model for the Titanic kaggle competition'''
# configure API Credentials path befor kaggle import
import os
os.environ['KAGGLE_CONFIG_DIR'] = r"C:\Users\maxdu\.kaggle\kaggle_private"

import kaggle

# authenticate api
kaggle.api.authenticate()

def download_data():
    # download seperately, as opposed to competition_download_files
    # because that will create a zip folder, and there is no unzip option
    kaggle.api.competition_download_file(competition="titanic", file_name="train.csv", path='./data')
    kaggle.api.competition_download_file(competition="titanic", file_name="test.csv", path="./data")

def main():
    pass

if __name__ == "__main__":
    main()