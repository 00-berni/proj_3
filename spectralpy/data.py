"""
DATA PACKAGE
============

***

::METHODS::
-----------

***

!TO DO!
-------
    - [] **Update the file `.json` and methods: `collects_fits`, `get_data_fit`, `extract_data`, `get_data`**


***
    
?WHAT ASK TO STEVE?
-------------------
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.io.fits import HDUList
from .stuff import angle_correction, hotpx_remove
from .display import showfits
from numpy.typing import NDArray


def data_extraction(path_file: str) -> dict:
    """Extracting data from a .json file

    Parameters
    ----------
    path_file : str
        path of the data file

    Returns
    -------
    data_file : dict
        data organized into nights of aquisition and objects
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

def collect_fits(night: int, obj: str) -> tuple[NDArray, NDArray]:
    """Collecting data fits for a chosen night observation and object.

    Parameters
    ----------
    night : int
        index of chosen night
    obj : str
        name of the object

    Returns
    -------
    extracted : NDArray
        object in that night
    cut : NDArray
        section limits for the images
    """
    cut = np.loadtxt(os.path.join(DATA_FOLDER, NIGHTS[night], obj, 'cut_indicies.txt'), dtype=int, unpack=False)
    cut = np.where(cut == -1, None, cut)
    extracted = DATA_ALL[NIGHTS[night]][obj]
    return extracted, cut

def data_file_path(night: int, obj: str, data_file: str) -> str:
    """

    Parameters
    ----------
    night : int
        _description_
    obj : str
        _description_
    data_file : str
        _description_

    Returns
    -------
    str
        _description_
    """
    return os.path.join(DATA_FOLDER, NIGHTS[night], obj , data_file + '.fit')

##* 
def get_data_fit(path: str, lims: list = [0,None,0,None], hotpx: bool = True, v: int = -1, title: str = '', n: int = None, dim: list[int] = [10,7], display_plots: bool = True) -> tuple[HDUList, NDArray]:
    """Function to open fits file and extract data.
    It brings the path and extracts the data, giving a row image.
    You can set a portion of image and also the correction for hotpx.

    Parameters
    ----------
    path : str
        path of the fits file
    lims : list, optional
        edges of the fits, by default `[0,None,0,None]`
        `lims` parameter controls the x and y extremes 
        in such the form `[lower y, higher y, lower x, higher x]`
    hotpx : bool, optional
        parameter to remove or not the hot pixels, by default `True`
        It calls the functions `hotpx_remove`
    v : int, optional
        cmap parameter, by default `-1` 
            -  `1` for false colors
            -  `0` for grayscale
            - `-1` for reversed grayscale
    title : str, optional
        title of the image, by default `''`
    n : int, optional
        figure number, by default `None`
    dim : list[int], optional
        figure size, by default `[10,7]`
    display_plots : bool, optional
        _description_, by default `True`

    Returns
    -------
    hdul : HDUList
        information about the chosen fits file
    data : NDArray
        data of the spectrum
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
    return hdul, data
##*


def extract_data(ch_obs: int, ch_obj: str, sel: list[str] | str = 'all') -> list[str]:
    """Collecting data from data files.
    
    Parameters
    ----------
    ch_obs : int
        chosen obeservation night
    ch_obj : str
        chosen object name
    sel : list[str] | str, optional
        _description_, by default 'all'

    Returns
    -------
    data : list[str]
        the list with paths and extrema
    
    Notes
    -----
    Given a selected observation night and object, the function 
    returns both the paths of the target and the calibration lamp
    (also the flat if any) and the corrisponding values for the 
    edges of the images. 

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
        data = []
        if (sel == 'all') or ('obj' in sel) or ('obj_fit' in sel):
            data += [obj_fit]   
        if (sel == 'all') or ('obj' in sel) or ('lims_fit' in sel):
            data += [lims_fit]
        if (sel == 'all') or ('lamp' in sel) or ('obj_lamp' in sel):
            data += [obj_lamp]
        if (sel == 'all') or ('lamp' in sel) or ('lims_lamp' in sel):
            data += [lims_lamp]
        # condition for the presence of flat
        if ch_obj == 'giove' or ch_obj == 'arturo':
            obj_flat = obj[-1]
            lims_flat = lims[-1]
            obj_flat = data_file_path(ch_obs, ch_obj, obj_flat)
            if (sel == 'all') or ('flat' in sel) or ('obj_flat' in sel):
                data += [obj_flat]             
            if (sel == 'all') or ('flat' in sel) or ('lims_flat' in sel):
                data += [lims_flat]             
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
        data = []
        if (sel == 'all') or ('obj' in sel) or ('obj_fit' in sel):
            data += [obj_fit]
        if (sel == 'all') or ('lamp' in sel) or ('thor' in sel):
            data += [thor]  
        if (sel == 'all') or ('lamp' in sel) or ('tung' in sel):  
            data += [tung]  
    return data



def get_data(ch_obj: str, obj_fit: str, lims_fit: list[int | None] = [None,None,None,None] , angle: float | None = None, display_plots: bool = False) -> tuple[HDUList, NDArray, float]:
    """Extracting the fits data

    Parameters
    ----------
    ch_obj : str
        chosen object name
    obj_fit : str
        path of the target
    lims_fit : list[int  |  None], optional
        extrema of the image, by default `[None,None,None,None]`
    angle : float | None, optional
        the inclination angle to rotate the image, by default `None`
        If it is `None` it will be estimated
    display_plots : bool, optional
        if it is `True` images/plots are displayed, by default `False`

    Returns
    -------
    hdul : HDList
        fits information
    sp_data : NDArray
        spectrum data
    angle : float
        inclination angle
    
    Notes
    -----
    The function gets the data of the spectrum from the fits file of a selected target and corrects for the
    inclination, returning the fits information (`hdul`), the spectrum data (`sp_data`) and the angle of
    inclination (`angle`).
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



