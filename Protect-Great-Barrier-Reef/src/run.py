import os

command_sequences = ("python3 wandb_login.py",
                    "python3 data/data_downlaod.py",
                    "python3 helper.py"
                    "python3 data/data_processing.py",
                    "python3 define_directory.py",
                    "python3 visualization/data_visualization.py"
                    "python3 models/model_dev.py",
                    "python3 train.py"
)

for i, x in enumerate(command_sequences):
  print("{}. RUNNING: [{}]".format(i, x))
  os.system(x)