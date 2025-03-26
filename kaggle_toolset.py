import os
os.environ['KAGGLE_CONFIG_DIR'] = r"C:\Users\maxdu\.kaggle\kaggle_private"

import kaggle

def authenticate():
    kaggle.api.authenticate()

def download_data(competition, file_name, path) -> None:
    '''Will download given file from corresponding competition'''
    # download seperately, as opposed to competition_download_files
    # because that will create a zip folder, and there is no unzip option
    kaggle.api.competition_download_file(competition=competition, file_name=file_name, path=path)