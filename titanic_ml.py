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

def test_train_split(data : pd.DataFrame, seed : int) -> tuple[pd.DataFrame, pd.DataFrame]:
    generator = torch.Generator().manual_seed(seed)
    train_size = int(0.8*len(data))
    val_size = len(data) - train_size
    train, validation = torch.utils.data.random_split(data, [train_size, val_size], generator=generator)
    return (train, validation)

def main():
    download_data('titanic', 'train.csv', "./data")
    download_data('titanic', 'test.csv', "./data")

    train_data = read_data("./data/train.csv")
    test_data = read_data("./data/test.csv")

    data_processing.preprocess(train_data)
    data_processing.preprocess(test_data)

    apply_encoding(train_data)
    apply_encoding(test_data)

    # to sample data transformations
    # train_data.to_csv("sample1.csv")
    # test_data.to_csv("sample2.csv")

    train, val = test_train_split(train_data, 42)

if __name__ == "__main__":
    main()