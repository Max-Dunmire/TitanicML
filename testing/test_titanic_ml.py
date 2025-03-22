'''Tests each component of the titanic model program'''
import unittest
import titanic_ml
import pandas as pd
from pathlib import Path

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
            import kaggle
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

    def test_file_creation(self):
        '''will test that the proper file names are created in the right folder'''
        titanic_ml.download_data()

        train_path = Path("./data/train.csv")
        test_path = Path("./data/test.csv")

        self.assertTrue(train_path.exists())
        self.assertTrue(test_path.exists())

    def test_file_contents(self):
        '''Will test that the right columns are present in the dataframe'''
        train_df = pd.read_csv("./data/train.csv")
        columns = {"PassengerId", "Survived", "Pclass", "Name", 
                   "Sex", "Age", "SibSp", "Parch", "Ticket",
                   "Fare", "Cabin", "Embarked"}
        self.assertTrue(set(train_df.columns) == columns)

if __name__ == "__main__":
    unittest.main()