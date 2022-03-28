import os
import json
import kaggle

# define secret path
secret_path = "/home/dave117/MLOps/projects/Kaggle_Competitions/Protect-Great-Barrier-Reef/.secret/api.json"

def get_keys(path):
    with open(path) as f:
        return json.load(f)

# import authentication informations (tokens)
auth_keys = get_keys(secret_path)
KAGGLE_USERNAME = auth_keys['KAGGLE_USERNAME']
KAGGLE_KEY = auth_keys["KAGGLE_KEY"]

# import authentication informations (tokens)
os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME # username from the json file
os.environ['KAGGLE_KEY'] = KAGGLE_KEY # key from the json file (generate new key everytime from account setting)

# download data
os.system("kaggle competitions download -c tensorflow-great-barrier-reef")

# extract the data
os.system("unzip tensorflow-great-barrier-reef")
