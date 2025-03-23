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

def apply_encoding(df : pd.DataFrame):
    df['Sex'] = df['Sex'].apply(lambda sex: {'male':0, 'female':1}[sex])
    df['Embarked'] = df['Embarked'].apply(lambda port: {'Q':0, 'S':1, 'C':2}[port])
    df['Name'] = data_processing.encode_names(df['Name'])
    df['Ticket'] = data_processing.encode_tickets(df['Ticket'])

def main():
    download_data('titanic', 'train.csv', "./data")
    download_data('titanic', 'test.csv', "./data")

    train_data = read_data("./data/train.csv")
    test_data = read_data("./data/test.csv")

    apply_encoding(train_data)
    print(train_data.head(10))

if __name__ == "__main__":
    main()