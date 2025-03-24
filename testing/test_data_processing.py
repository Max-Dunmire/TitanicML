'''Will test the data_processing module for functionality.'''
import unittest
import sys
import pandas as pd

# set the path to search for the external module
sys.path.insert(0, r"C:\Users\maxdu\Github\Kaggle\TitanicML")

import data_processing # module to be tested

class TestCreateVocabulary(unittest.TestCase):
    '''Will test the encode_names funtion.'''

    def test_dictionary_creation(self):
        '''Will test that the vocabulary dictionary is created correctly for a sample list of names'''

        # names and correct vocabulary for associated names
        names = pd.Series(['Alice', 'Amy', 'Bertram', 'Zokamyous', 'Yytrrium'])
        correct_vocabulary = {'A':1, 'l':2, 'i':3, 'c':4, 'e':5, 'm':6,
                              'y':7, 'B':8, 'r':9, 't':10, 'a':11, 'Z':12,
                              'o':13, 'k':14, 'u':15, 's':16, 'Y':17, "max":9}
        
        vocabulary = data_processing._create_vocabulary(names, 9)
        # print(vocabulary) # uncomment to see the dictionary (debugging)
        self.assertTrue(correct_vocabulary == vocabulary, "Dictionaries did not match")

class TestAssignValues(unittest.TestCase):
    '''Will test the assign_values() function to switch'''

    def test_name_values(self):
        '''This will test that the proper number lists are created for given names'''

        # test names
        names = pd.Series(['Alice', 'Amy', 'Bertram'])
        # correct number lists and vocabulary
        correct_lists = pd.Series([[1, 2, 3, 4, 5, 0, 0], [1, 6, 7, 0, 0, 0, 0], [8, 5, 9, 10, 9, 11, 6]])
        correct_vocabulary = {'A':1, 'l':2, 'i':3, 'c':4, 'e':5, 'm':6,
                              'y':7, 'B':8, 'r':9, 't':10, 'a':11, 'max':7}

        name_lists = data_processing._assign_values(names, correct_vocabulary)
        self.assertTrue(correct_lists.equals(name_lists))

class TestEncodeNames(unittest.TestCase):
    '''Will test the encode names function as a whole.'''

    def test_encoding(self):
        names = pd.Series(['Alice', 'Amy', 'Bertram'])
        correct_lists = pd.Series([[1, 2, 3, 4, 5, 0, 0], [1, 6, 7, 0, 0, 0, 0], [8, 5, 9, 10, 9, 11, 6]])

        encoded_names = data_processing.encode_names(names, 7)

        self.assertTrue(correct_lists.equals(encoded_names))

class TestPreprocess(unittest.TestCase):
    '''Will test for removal of NaN values of the data frame.'''

    def test_remove_NaN(self):
        '''Tests that rows with NaN values are dropped'''
        df = pd.DataFrame({'a':[1,2,None], 'b':[None, 3,4], 'c':[5,6,7], 'Cabin':[None,3,4]})
        data_processing.preprocess(df)

        self.assertTrue(len(df) == 1)
        self.assertTrue(not bool(df.isna().any().any()))

if __name__ == "__main__":
    unittest.main()