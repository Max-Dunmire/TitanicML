'''Will test the data_processing module for functionality.'''
import unittest
import sys
import pandas as pd

# set the path to search for the external module
sys.path.insert(0, r"C:\Users\maxdu\Github\TitanicML")

import data_processing # module to be tested

class TestCreateVocabulary(unittest.TestCase):
    '''Will test the encode_names funtion.'''

    def test_dictionary_creation(self):
        '''Will test that the vocabulary dictionary is created correctly for a sample list of names'''
        names = pd.Series(['Alice', 'Amy', 'Bertram', 'Zokamyous', 'Yytrrium'])
        correct_vocabulary = {'A':1, 'l':2, 'i':3, 'c':4, 'e':5, 'm':6,
                              'y':7, 'B':8, 'r':9, 't':10, 'a':11, 'Z':12,
                              'o':13, 'k':14, 'u':15, 's':16, 'Y':17}
        vocabulary = data_processing.create_vocabulary(names)
        # print(vocabulary) # uncomment to see the dictionary (debugging)
        self.assertTrue(correct_vocabulary == vocabulary, "Dictionaries did not match")

class TestAssignValues(unittest.TestCase):
    '''Will test the assign_values() function to switch'''

if __name__ == "__main__":
    unittest.main()