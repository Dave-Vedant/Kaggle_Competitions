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
os.system("mkdir Papa")
os.system("mkdir Papa/external")
os.system("kaggle competitions download -c tensorflow-great-barrier-reef -p Papa/external")

print('data(*.zip) downloaded successfully, move to unzip stage')

# extract the datasys
os.system("mkdir Papa")
os.system ("mkdir Papa/raw")
os.system("mkdir Papa/interim")
os.system("mkdir Papa/processed")


os.system("unzip tensorflow-great-barrier-reef -d Papa/raw")
print('unzip completed')
