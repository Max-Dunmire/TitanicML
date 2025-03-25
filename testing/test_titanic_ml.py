'''Tests each component of the titanic model program'''
import unittest
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# set the path to find the external module
sys.path.insert(0, r"C:\Users\maxdu\Github\Kaggle\TitanicML")

import titanic_ml

class TestAPIConfig(unittest.TestCase):
    '''Testing class for the api finding the kaggle folder containing api key'''

    def test_api_setup(self):
        '''Will test that the authentication of the kaggle api using alternate folder is functional.'''

        # import os for kaggle.json folder configuration
        import os
        os.environ['KAGGLE_CONFIG_DIR'] = r"C:\Users\maxdu\.kaggle\kaggle_private"

        passed = True
        error = None

        try:
            # import kaggle api
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
        except OSError as e:
            error = e
            passed = False

        self.assertTrue(passed, error)

class TestDownloadDatasets(unittest.TestCase):
    '''Testing class for the download_dat() function'''

    def setUp(self):
        '''Authenticate the api before downloading'''
        titanic_ml.kaggle.api.authenticate()
        titanic_ml.download_data("titanic", "train.csv", "./data")
        titanic_ml.download_data("titanic", "test.csv", "./data")

    def test_file_creation(self):
        '''will test that the proper file names are created in the right folder'''

        train_path = Path("./data/train.csv")
        test_path = Path("./data/test.csv")

        self.assertTrue(train_path.exists())
        self.assertTrue(test_path.exists())

    def test_file_contents(self):
        '''Will test that the right columns are present in the dataframe'''
        # will read from cwd, should be updated in the future to handle different directories
        train_df = pd.read_csv("./data/train.csv")
        columns = {"PassengerId", "Survived", "Pclass", "Name", 
                   "Sex", "Age", "SibSp", "Parch", "Ticket",
                   "Fare", "Cabin", "Embarked"}
        self.assertTrue(set(train_df.columns) == columns)

class TestApplyEncoding(unittest.TestCase):
    '''Will test the apply encoding function for replaced values, tests will not
       function if the preprocess function is not correct.'''

    def setUp(self):
        '''Makes sure the dataset has been downloaded for testing'''
        titanic_ml.download_data('titanic', 'train.csv', './data')
        
    def test_names(self):
        '''Will test that the names column of the dataset has been set to the right data-type'''
        df = pd.read_csv('./data/train.csv')
        titanic_ml.data_processing.preprocess(df)
        len_name = titanic_ml.data_processing.grab_max_len(df['Name'])
        len_ticket = titanic_ml.data_processing.grab_max_len(df['Ticket'])
        titanic_ml.apply_encoding(df, len_name, len_ticket)
        encoded_names = df['Name']

        self.assertIs(type(encoded_names.iat[0]), list, "Names have not been converted to lists")
        self.assertIs(type(encoded_names.iat[0][0]), int, "Name lists are not of integers")
        self.assertIs(type(df['Sex'].iat[0]), np.int64, "Sexs have not been converted to int")
        self.assertIs(type(df['Embarked'].iat[0]), np.int64, "Ports have not been converted to int")
        self.assertIs(type(df['Ticket'].iat[0]), list, "Tickets have not been converted to lists")
        self.assertIs(type(df['Ticket'].iat[0][0]), int, "Name lists are not of integers")


if __name__ == "__main__":
    unittest.main()