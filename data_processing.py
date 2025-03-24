'''Contains important data processing methods for making data neural network friendly'''
import pandas as pd

def _create_vocabulary(names : pd.Series) -> dict:
    '''Will create a vocabulary of all characters seen in names list with integer code'''

    vocabulary = {'max' : 0} # max will keep track of longest name
    index = 1 # we will reserve 0 for padding

    for i in range(len(names)):
        name = names.iat[i] # grab each name
        # set the value for 'max' key to be longest seen name
        vocabulary['max'] = max(vocabulary['max'], len(name))
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

    return pd.Series(encoded_names)

def encode_names(names : pd.Series) -> pd.Series:
    '''puts create vocab and assign values functions together to encode names'''

    vocabulary = _create_vocabulary(names)
    encoded_names = _assign_values(names, vocabulary)
    return encoded_names

def encode_tickets(tickets : pd.Series) -> pd.Series:
    ''''Uses same name encoding functions to encode ticket numbers'''

    vocabulary = _create_vocabulary(tickets)
    encoded_tickets = _assign_values(tickets, vocabulary)
    return encoded_tickets

def preprocess(df : pd.DataFrame) -> None:
    '''Will be remove NaN values from the data frame'''
    # drops all rows containing a NaN value
    # may be changed in the future to replace with median value
    df.drop('Cabin', axis=1, inplace=True) # Cabin row is mostly null, don't want to drop 77% of rows
    df.dropna(axis=0, inplace=True)