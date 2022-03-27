import wandb 
import json
secret_path = '/home/dave117/MLOps/projects/Kaggle_Competitions/Protect-Great-Barrier-Reef/.secret/api.json'

def get_keys(path):
    with open(path) as f:
        return json.load(f)

try:
    auth_keys = get_keys(secret_path)
    api_key = auth_keys['WANDB_API_Key']
    wandb.login(key = api_key)
    print('wandb login succeed')
except:
    wandb.login(anonymous='must')
    print('To use your W&B account,\n \
        Go to Add-ons ==>  Secrets and provide your W&B access token. \n \
            Use the Label name as WANDB_API_Key. ==> Get your W&B access token from here: https://wandb.ai/authorize')

print('::::: wandb.py end ::::::')

