import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def data_extraction(path_file: str):
    with open(path_file) as f:
        data_file = f.read()
    data_file = json.loads(data_file)
    return data_file
    


if __name__ == '__main__':

    # taking path of the current folder
    PWD =  os.path.dirname(os.path.realpath(__file__))
    # path of the project folder
    PROJECT_FOLDER = control = os.path.split(PWD)[0]
    # path of the data folder
    DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data_files')
    print(PWD)
    print(PROJECT_FOLDER)
    print(DATA_FOLDER)
    data_file = os.path.join(DATA_FOLDER, 'objs_per_night.json')

