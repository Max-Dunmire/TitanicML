import os
os.environ['KAGGLE_CONFIG_DIR'] = r"C:\Users\maxdu\.kaggle\kaggle_private"

import kaggle
import time

def authenticate():
    kaggle.api.authenticate()

def download_data(competition, file_name, path) -> None:
    '''Will download given file from corresponding competition'''
    # download seperately, as opposed to competition_download_files
    # because that will create a zip folder, and there is no unzip option
    kaggle.api.competition_download_file(competition=competition, file_name=file_name, path=path)

def submit(competition_name, file_path):
    res = kaggle.api.competition_submit(file_name=file_path, message="Final Predictions", competition=competition_name)
    if res == "Could not submit to competition":
        print("Submission Failed")

    time.sleep(3)

    submissions = kaggle.api.competition_submissions(competition_name)

    for index, submission in enumerate(submissions):

        if index < 10:
            id = submission.ref
            score = submission.public_score

            print(f"Id: {id}\tScore: {score or 'Error'}")
        else:
            break

if __name__ == "__main__":
    submit("titanic", "data/predictions.csv")
