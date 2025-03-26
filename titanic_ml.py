'''This program will create and train a model for the Titanic kaggle competition'''
# configure API Credentials path befor kaggle import
# import os
# os.environ['KAGGLE_CONFIG_DIR'] = r"C:\Users\maxdu\.kaggle\kaggle_private"

# import kaggle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import data_tools
from model import TitanicModel

# authenticate api
# kaggle.api.authenticate()

# def download_data(competition, file_name, path) -> None:
#     '''Will download given file from corresponding competition'''
#     # download seperately, as opposed to competition_download_files
#     # because that will create a zip folder, and there is no unzip option
#     kaggle.api.competition_download_file(competition=competition, file_name=file_name, path=path)

# def read_data(path : str) -> pd.DataFrame:
#     '''Will read csv of downloaded data and return a dataframe'''
#     df = pd.read_csv(path)
#     return df

# def apply_encoding(df : pd.DataFrame, longest_name : int, longest_ticket : int) -> None:
#     '''Applies the encoding to the dataframe needed to remove strings'''
#     df['Sex'] = df['Sex'].apply(lambda sex: {'male':0, 'female':1}[sex])
#     df['Embarked'] = df['Embarked'].apply(lambda port: {'Q':0, 'S':1, 'C':2}[port])
#     df['Name'] = data_tools.encode_names(df['Name'], longest_name)
#     df['Ticket'] = data_tools.encode_tickets(df['Ticket'], longest_ticket)

# def test_train_split(data : torch.utils.data.Dataset, seed : int) -> tuple[pd.DataFrame, pd.DataFrame]:
#     '''Split the training data into train and validation'''
#     generator = torch.Generator().manual_seed(seed)
#     train_size = int(0.8*len(data))
#     val_size = len(data) - train_size
#     train, validation = torch.utils.data.random_split(data, [train_size, val_size], generator=generator)
#     return (train, validation)

# def longest_element(series1 : pd.Series, series2 : pd.Series) -> int:
#     '''Finds the longest element in a column with string data across both data frames'''
#     train_max = data_tools.grab_max_len(series1)
#     test_max = data_tools.grab_max_len(series2)
#     return max(train_max, test_max)

# def main():
    # download_data('titanic', 'train.csv', "./data")
    # download_data('titanic', 'test.csv', "./data")

    # train_data = read_data("./data/train.csv")
    # test_data = read_data("./data/test.csv")

    # data_tools.preprocess(train_data)
    # data_tools.preprocess(test_data)

    # longest_name = longest_element(train_data['Name'], test_data['Name'])
    # longest_ticket = longest_element(train_data['Ticket'], test_data['Ticket'])

    # apply_encoding(train_data, longest_name, longest_ticket)
    # apply_encoding(test_data, longest_name, longest_ticket)

    # to sample data transformations
    # train_data.to_csv("sample3.csv", index=False)

    # print(len(train_data.columns) + len(train_data['Ticket'].iat[0]) + len(train_data['Name'].iat[0]) - 2)

    # train_data = data_tools.unpack_lists(train_data, 'Name')
    # train_data = data_tools.unpack_lists(train_data, 'Ticket')

    # train_vals = train_data.drop('Survived', axis=1)
    # train_labels = train_data['Survived']

    # train_vals = torch.FloatTensor(train_vals.values)
    # train_labels = torch.LongTensor(train_labels.values)

    # titanic_dataset = TensorDataset(train_vals, train_labels)

    # train, validation = test_train_split(titanic_dataset, 42)

    # train_loader = DataLoader(train, batch_size=32)

    # model = TitanicModel()

    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # epochs = 10
    
    # for epoch in range(epochs):

    #     running_loss = 0.0

    #     for index, data in enumerate(train_loader):
    #         data, labels = data

    #         optimizer.zero_grad()

    #         predictions = model(data)

    #         loss : torch.Tensor = criterion(predictions, labels)
    #         loss.backward()

    #         optimizer.step()

    #         running_loss += loss.item()

    #     print(f'Loss: {running_loss / len(train_loader):.4f}')

    # torch.save(model.state_dict(), "titanic_model.pth")

# if __name__ == "__main__":
#     main()