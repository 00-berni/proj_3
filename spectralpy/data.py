"""
    # Data module
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from .stuff import angle_correction, hotpx_remove
from .display import showfits


def data_extraction(path_file: str) -> dict:
    """Extracting data from a .json file

    :param path_file: path of the data file
    :type path_file: str
    
    :return: data organized into nights of aquisition and objects
    :rtype: dict
    """
    import json 
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
def get_data_fit(path: str, lims: list = [0,-1,0,-1], hotpx: bool = True, v: int = -1, title: str = '', n: int = None, dim: list[int] = [10,7], display_plots: bool = True) -> tuple:
    """Function to open fits file and extract data.
    It brings the path and extracts the data, giving a row image.
    You can set a portion of image and also the correction for hotpx.

    It calls the functions `hotpx_remove` and `showfits`.

    :param path: path of the fits file
    :type path: str
    :param lims: edges of the fits, defaults to [0,-1,0,-1]
    :type lims: list, optional
    :param hotpx: parameter to remove or not the hot pixels, defaults to True
    :type hotpx: bool, optional
    :param v: cmap parameter: 1 for false colors, 0 for grayscale, -1 for reversed grayscale; defaults to -1
    :type v: int, optional
    :param title: title of the image, defaults to ''
    :type title: str, optional
    :param n: figure number, defaults to None
    :type n: int, optional
    :param dim: figure size, defaults to [10,7]
    :type dim: list[int], optional

    :return: `hdul` list of the chosen fits file and `data` of the spectrum
    :rtype: tuple

    .. note:: `lims` parameter controls the x and y extremes in such the form [lower y, higher y, lower x, higher x]
    """
    # open the file
    hdul = fits.open(path)
    # print fits info
    hdul.info()
    # print header
    hdr = hdul[0].header
    print(' - HEADER -')
    for parameter in hdr:
        hdr_info = f'{parameter} =\t{hdr[parameter]}' 
        comm = hdr.comments[parameter]
        if comm != '': hdr_info = hdr_info + ' \ ' + comm 
        print(hdr_info)
    print()

    # data extraction
    # format -> data[Y,X]
    data = hdul[0].data
    ly,ry,lx,rx = lims
    data = data[ly:ry,lx:rx]
    # hot px correction
    if hotpx == True:
        data = hotpx_remove(data)
    # Spectrum image
    if display_plots == True: 
        from .display import showfits
        showfits(data, v=v,title=title,n=n,dim=dim) 
    return hdul,data
##*


def extract_data(ch_obs: int, ch_obj: str) -> list[str]:
    """Collecting data from data files.
    
    Given a selected observation night and object, the function 
    returns both the paths of the target and the calibration lamp
    (also the flat if there is) and the corrisponding values for 
    the edges of the images. 

    :param ch_obs: chosen obeservation night
    :type ch_obs: int
    :param ch_obj: chosen object name
    :type ch_obj: str
    
    :return: the list with paths and extrema
    :rtype: list
    """
    # only for the first two observation nights Alpy was used
    if ch_obs < 2:
        # extracting informations
        obj, lims = collect_fits(ch_obs, ch_obj)
        # collecting in different variables
        obj_fit, obj_lamp = obj[:2] 
        lims_fit, lims_lamp = lims[:2]

        # appending the path
        obj_fit = data_file_path(ch_obs, ch_obj, obj_fit)
        obj_lamp = data_file_path(ch_obs, ch_obj, obj_lamp)
        
        # storing in `data` list
        #   format is [obj, its extrema, ...]
        data = [obj_fit, lims_fit, obj_lamp, lims_lamp]   
        # condition for the presence of flat
        if ch_obj == 'giove' or ch_obj == 'arturo':
            obj_flat = obj[-1]
            lims_flat = lims[-1]
            obj_flat = data_file_path(ch_obs, ch_obj, obj_flat)
            data += [obj_flat, lims_flat]             
    # for echelle data extraction
    else:
        # extracting informations
        obj, _ = collect_fits(ch_obs, ch_obj, cutignore=True)
        # collecting in different variables
        obj_fit, obj_lamp = obj[:2]
        thor, tung = obj_lamp

        # appending the path
        obj_fit = data_file_path(ch_obs, ch_obj, obj_fit)
        thor = data_file_path(ch_obs, ch_obj, thor)
        tung = data_file_path(ch_obs, ch_obj, tung)
        
        # storing in `data` list
        #   format is [obj, its extrema, ...]
        data = [obj_fit, thor, tung]   
    return data



def get_data(ch_obj: str, obj_fit: str, lims_fit: list[int | None] = [None,None,None,None] , angle: float | None = None, display_plots: bool = False) -> tuple:
    """Extracting the fits data
    The function gets the data of the spectrum from the fits file of a selected target and corrects for the
    inclination, returning the fits information (`hdul`), the spectrum data (`sp_data`) and the angle of
    inclination (`angle`).

    :param ch_obj: chosen object name
    :type ch_obj: str
    :param obj_fit: path of the target
    :type obj_fit: str
    :param lims_fit: extrema of the image
    :type lims_fit: list[int]
    :param angle: the inclination angle to rotate the image. If it is None it will be estimated, defaults to None
    :type angle: float | None, optional
    :param display_plots: if it is True images/plots are displayed, defaults to False
    :type display_plots: bool, optional
    
    :return: fits information, spectrum data, inclination angle 
    :rtype: tuple
    """
    # collecting fits informations and spectrum data
    hdul, sp_data = get_data_fit(obj_fit, lims=lims_fit, title='Row spectrum of '+ ch_obj, n=1, display_plots=display_plots)
    # correcting for inclination angle
    angle, sp_data = angle_correction(sp_data, angle=angle, display_plots=display_plots)
    # condition to display the images/plots
    if display_plots == True:
        showfits(sp_data, title='Rotated image')
        plt.show()
    return hdul, sp_data, angle


