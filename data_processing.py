'''Contains important data processing methods for making data neural network friendly'''
import pandas as pd

def create_vocabulary(names : pd.Series) -> None:
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

def assign_values(names : pd.Series, vocabulary : dict[str, int]) -> pd.Series:
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