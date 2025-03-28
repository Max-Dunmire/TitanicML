'''Contains important data processing methods for making data neural network friendly'''
import pandas as pd

def _create_vocabulary(names : pd.Series, longest : int) -> dict:
    '''Will create a vocabulary of all characters seen in names list with integer code'''

    vocabulary = {'max' : longest} # max will keep track of longest name
    index = 1 # we will reserve 0 for padding

    for i in range(len(names)):
        name = names.iat[i] # grab each name
        for letter in name:
            # if the letter has not yet been seen then add it
            if not vocabulary.get(letter):
                vocabulary[letter] = index
                index += 1

    return vocabulary

def _assign_values(names : pd.Series, vocabulary : dict[str, int]) -> pd.Series:
    '''Will take list of names and vocabulary and translate the name to integer list'''

    encoded_names = []

    for i in range(len(names)):

        name = names.iat[i] # grab name

        # find value for each letter in name
        number_encoded = [vocabulary[letter] for letter in name]

        # use max value to determine number of 0s needed for padding
        zeros = vocabulary['max'] - len(number_encoded)
        zeros = [0]*zeros # create array of zeros

        number_encoded = number_encoded + zeros # combine lists
        encoded_names.append(number_encoded)

    return encoded_names

def encode_names(names : pd.Series, longest_value : int) -> pd.Series:
    '''puts create vocab and assign values functions together to encode names'''

    vocabulary = _create_vocabulary(names, longest_value)
    encoded_names = _assign_values(names, vocabulary)
    return encoded_names

def encode_tickets(tickets : pd.Series, longest_ticket : int) -> pd.Series:
    ''''Uses same name encoding functions to encode ticket numbers'''

    vocabulary = _create_vocabulary(tickets, longest_ticket)
    encoded_tickets = _assign_values(tickets, vocabulary)
    return encoded_tickets

def preprocess(df : pd.DataFrame, train: bool) -> None:
    '''Will be remove NaN values from the data frame'''
    # drops all rows containing a NaN value
    # may be changed in the future to replace with median value
    df.drop('Cabin', axis=1, inplace=True) # Cabin row is mostly null, don't want to drop 77% of rows
    if train:
        df.dropna(axis=0, inplace=True)

def grab_max_len(series : pd.Series) -> int:
    '''Will grab the longest item length and return it.'''
    return int(series.apply(len).max())

def unpack_lists(df : pd.DataFrame, column : str) -> pd.DataFrame:
    '''Takes a column containing list data and turns 
       it into regular indices of the data frame'''
    # grab the index of the inputted column
    columns = list(df.columns)
    col_index = columns.index(column)

    # divide the dataframe into the parts before and after the column
    first_half = df.iloc[:,0:col_index]
    second_half = df.iloc[:, col_index+1:]

    # resets indexes to counting order after dropping NaN rows
    # avoids issues with combining data frames with conflicting index numbers
    first_half.reset_index(drop=True, inplace=True)
    second_half.reset_index(drop=True, inplace=True)

    # extract the column with lists
    list_column = df[column]

    # create a list of lists
    lists = []
    for i in range(len(list_column)):
        lists.append(list_column.iat[i])

    # convert list of lists into dataframe
    middle = pd.DataFrame(lists)

    # put all parts into a list
    all_parts = [first_half, middle, second_half]
    # concatenate all the data frames together
    total_df = pd.concat(all_parts, axis=1)
    total_df.reindex
    return total_df