import pandas as pd
import data_tools

def read_data(path : str) -> pd.DataFrame:
    '''Will read csv of downloaded data and return a dataframe'''
    df = pd.read_csv(path)
    return df

def apply_encoding(df : pd.DataFrame, longest_name : int, longest_ticket : int) -> None:
    '''Applies the encoding to the dataframe needed to remove strings'''
    df['Sex'] = df['Sex'].apply(lambda sex: {'male':0, 'female':1}[sex])
    df['Embarked'] = df['Embarked'].apply(lambda port: {'Q':0, 'S':1, 'C':2}[port])
    df['Name'] = data_tools.encode_names(df['Name'], longest_name)
    df['Ticket'] = data_tools.encode_tickets(df['Ticket'], longest_ticket)

def longest_element(series1 : pd.Series, series2 : pd.Series) -> int:
    '''Finds the longest element in a column with string data across both data frames'''
    train_max = data_tools.grab_max_len(series1)
    test_max = data_tools.grab_max_len(series2)
    return max(train_max, test_max)

def main():
    train_data = read_data("data/train.csv")
    test_data = read_data("data/test.csv")

    data_tools.preprocess(train_data, train=True)
    data_tools.preprocess(test_data, train=False)

    longest_name = longest_element(train_data['Name'], test_data['Name'])
    longest_ticket = longest_element(train_data['Ticket'], test_data['Ticket'])

    apply_encoding(train_data, longest_name, longest_ticket)
    apply_encoding(test_data, longest_name, longest_ticket)

    train_data = data_tools.unpack_lists(train_data, 'Name')
    train_data = data_tools.unpack_lists(train_data, 'Ticket')
    test_data = data_tools.unpack_lists(test_data, 'Name')
    test_data = data_tools.unpack_lists(test_data, 'Ticket')

    train_vals = train_data.drop('Survived', axis=1)
    train_labels = train_data['Survived']

    train_vals.to_csv("data/training_data.csv", index=False)
    train_labels.to_csv("data/training_labels.csv", index=False)
    test_data.to_csv("data/test_model.csv", index=False)

if __name__ == "__main__":
    main()