import os
import yolo5



module_path = os.path.abspath(os.path.join('..'))

# run the following code one timeonly
'''
if module_path not in sys.path:
    sys.path.append(module_path + "Data/processed/yolov5")
    print(sys.path)
    '''

from yolov5 import utils
display = utils.notebook_init()