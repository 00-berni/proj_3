import os
import json
import numpy as np
import matplotlib.pyplot as plt


def data_extraction(path_file: str) -> dict:
    """Extracting data from a .json file

    :param path_file: path of the data file
    :type path_file: str
    
    :return: data organized into nights of aquisition and objects
    :rtype: dict
    """
    # opening the file
    with open(path_file) as f:
        data_file = f.read()
    # converting in a python dictonary
    data_file = json.loads(data_file)
    return data_file

# names for the selection of the night observation
NIGHTS = [f'0{i}_night' for i in range(1,6)]

# taking path of the current folder
PWD = os.path.dirname(os.path.realpath(__file__))
# path of the project folder
PROJECT_FOLDER = os.path.split(PWD)[0]
# path of the data folder
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data_files')
# path of calibration folder
CAL_FOLDER = os.path.join(DATA_FOLDER, 'calibration')
# path of results folder
RESULT_FOLDER = os.path.join(PROJECT_FOLDER, 'results')

# file with objects collection
OBJ_FILE = os.path.join(DATA_FOLDER, 'objs_per_night.json')

# extracting data
DATA_ALL = data_extraction(OBJ_FILE)

def collect_fits(night: int, obj: str) -> tuple:
    """Collecting data fits for a chosen night observation
    and object.

    :param night: index of chosen night
    :type night: int
    :param obj: name of the object
    :type obj: str
    
    :return: the list with data fit of that object in that night and section limits for the images
    :rtype: list
    """
    cut = np.loadtxt(os.path.join(DATA_FOLDER, NIGHTS[night], obj, 'cut_indicies.txt'), dtype=int, unpack=False)
    return DATA_ALL[NIGHTS[night]][obj], cut

def data_file_path(night: int, obj: str, data_file: str) -> str:
    return os.path.join(DATA_FOLDER, NIGHTS[night], obj , data_file + '.fit')



##*
def fastplot(x: np.ndarray, y: np.ndarray, numfig: int = None, fmt: str = '-', title: str = '', labels: list[str] = ['',''], dim: list[int] = [10,7]) -> None:
    """Function to display a plot quickly.
    You can choose to make a simple plot or adding some stuff.

    (I wrote it only because of my laziness in writing code).

    :param x: Data on x axis
    :type x: np.ndarray
    :param y: Data on y axis
    :type y: np.ndarray
    :param numfig: figure number, defaults to None
    :type numfig: int, optional
    :param title: title of the figure. default to ' '
    :type title: str, optional
    :param labels: axes label [x,y] format, defaults to ['','']
    :type labels: list[str], optional
    :param dim: figure size, defaults to [10,7] 
    :type dim: list[int], optional
    """
    xl,yl = labels
    plt.figure(numfig,figsize=dim)
    plt.title(title)
    plt.plot(x,y,fmt)
    plt.xlabel(xl)
    plt.ylabel(yl)



##*
def showfits(data: np.ndarray, v: int = -1, title: str = '', n: int = None, dim: list[int] = [10,7]) -> None:
    """Function to display the fits image.
    You can display simply the image or set a figure number and a title.

    :param data: image matrix of fits file
    :type data: np.ndarray
    :param v: cmap parameter: 1 for false colors, 0 for grayscale, -1 for reversed grayscale; defaults to -1
    :type v: int, optional
    :param title: title of the image, defaults to ''
    :type title: str, optional
    :param n: figure number, defaults to None
    :type n: int, optional
    :param dim: figure size, defaults to [10,7]
    :type dim: list[int], optional
    """
    plt.figure(n,figsize=dim)
    plt.title(title)
    if v == 1 : color = 'viridis'
    elif v == 0 : color = 'gray'
    else : color = 'gray_r'
    plt.imshow(data, cmap=color, norm='log')
    plt.colorbar()
##*
