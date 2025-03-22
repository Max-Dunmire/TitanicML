'''This program will create and train a model for the Titanic kaggle competition'''
# configure API Credentials path befor kaggle import
import os
os.environ['KAGGLE_CONFIG_DIR'] = r"C:\Users\maxdu\.kaggle\kaggle_private"

import kaggle
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import model
import data_processing

# authenticate api
kaggle.api.authenticate()

def download_data(competition, file_name, path) -> None:
    '''Will download given file from corresponding competition'''
    # download seperately, as opposed to competition_download_files
    # because that will create a zip folder, and there is no unzip option
    kaggle.api.competition_download_file(competition=competition, file_name=file_name, path=path)

def read_data(path : str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def main():
    pass

if __name__ == "__main__":
    main()