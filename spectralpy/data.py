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
from .stuff import angle_correction, make_cut_indicies, Spectrum
from .display import showfits
from numpy.typing import NDArray, ArrayLike
from typing import Sequence, Literal


def data_extraction(path_file: str) -> dict:
    """To extract data from a .json file

    Parameters
    ----------
    path_file : str
        path of the data file

    Returns
    -------
    data_file : dict
        data organized into nights of aquisition and targets names
    """
    import json 
    # open the file
    with open(path_file) as f:
        data_file = f.read()
    # convert in a python dictonary
    data_file = json.loads(data_file)
    return data_file


NIGHTS = ('16-08-26','17-03-27','18-04-22','18-11-27',
          '22-07-26_ohp','22-07-27_ohp','23-03-28')         #: nights of observation
PWD = os.path.dirname(os.path.realpath(__file__))           #: path of the current dir
PROJECT_DIR = os.path.split(PWD)[0]                         #: path of the project dir
DATA_DIR = os.path.join(PROJECT_DIR, 'data_files')          #: path of the data dir
CAL_DIR = os.path.join(DATA_DIR, 'calibration')             #: path of calibration dir
RESULT_DIR = os.path.join(PROJECT_DIR, 'results')           #: path of results dir
OBJ_FILE = os.path.join(DATA_DIR, 'objs_per_night.json')    #: path of file with objects collection
DATA_ALL = data_extraction(OBJ_FILE)                        #: dictionary with the targets per night

def collect_fits(night: str, obj: str) -> tuple[NDArray, ArrayLike]:
    """To collect data fits for a chosen night observation and object.

    Parameters
    ----------
    night : str
        selected night
    obj : str
        name of the target

    Returns
    -------
    extracted : NDArray
        object in that night
    cut : ArrayLike
        section limits for the images
    """
    # take target name
    extracted = DATA_ALL[night][obj]
    # load the edges of the cut
    #.. if the file `cut_indicies.txt` is absent
    #.. then one is generated
    lims_path = os.path.join(DATA_DIR, night, obj, 'cut_indicies.txt')
    try:
        cut = np.loadtxt(lims_path, dtype=int, unpack=False)
        cut = np.where(cut == -1, None, cut)
    except:
        numlines = len(extracted[0])+1 if isinstance(extracted[0],list) else 2
        cut = make_cut_indicies(lims_path,numlines)
    return extracted, cut

def data_file_path(night: str, obj: str, data_file: str) -> str:
    """To compute the exact path of a fits file

    Parameters
    ----------
    night : int
        selected observation night
    obj : str
        name of the target
    data_file : str
        selected fits file fot that target

    Returns
    -------
    str
        the path of the file
    """
    return os.path.join(DATA_DIR, night, obj, data_file + '.fit')

##* 
def get_data_fit(path: str, lims: Sequence[int | None] = [0,None,0,None], hotpx: bool = True, display_plots: bool = True, **kwargs) -> Spectrum:
    """To open fits file and extract data.

    Parameters
    ----------
    path : str
        path of the fits file
    lims : list, optional
        edges of the cut, by default `[0,None,0,None]`
        `lims` parameter controls the x and y extremes 
        in such the form `[lower y, higher y, lower x, higher x]`
    hotpx : bool, optional
        parameter to check and remove hot pixels, by default `True`
    display_plots : bool, optional
        parameter to plot data, by default `True`
    **kwargs:
        Parameters for the plot, see `display.showfits()`

    Returns
    -------
    target : Spectrum
        information about the selected spectrum, see `stuff.Spectrum`
        
    Notes
    -----
    Function brings the path and extracts the data, giving a row image.
    You can set a portion of image and also the correction for hotpx.
    
    """
    # open the file
    hdul = fits.open(path)
    # print fits info
    hdul.info()
    # data extraction
    #.. format -> data[Y,X]
    data = hdul[0].data
    if len(lims) == 1: lims = lims[0]
    ly,ry,lx,rx = lims
    # cut the image
    data = data[ly:ry,lx:rx]
    # store in `Spectrum` class
    target = Spectrum(hdul, data, lims=lims, hotpx=hotpx)
    # print the header
    target.print_header()
    # display target image
    if display_plots == True: 
        from .display import showfits
        kwargs['subtitle'] = kwargs['subtitle'] + f', exp. time: {target.get_exposure()} s' 
        showfits(data, **kwargs) 
    return target
##*

def get_data(ch_obj: str, obj_fit: str, lims_fit: list[int | None] = [None,None,None,None] , angle: float | None = None, display_plots: bool = False) -> tuple[Spectrum, float]:
    """To extract the fits data
  
    Parameters
    ----------
    ch_obj : str
        chosen object name
    obj_fit : str
        path of the target
    lims_fit : list[int | None], optional
        extrema of the image, by default `[None,None,None,None]`
    angle : float | None, optional
        the inclination angle to rotate the image, by default `None`
        If `angle is None` then the value is estimated by a 
        fitting rountine, see `stuff.angle_correction()`
    display_plots : bool, optional
        parameter to plot data, by default `True`
   
    Returns
    -------
    target : Spectrum
        information about the selected spectrum, see `stuff.Spectrum`
        If `angle != 0` then the inclination of data is corrected by 
        rotating rigidly
    
    Notes
    -----
    The function gets the data of the spectrum from the fits file of a selected target and corrects for the
    inclination.
    """
    # collect fits informations and spectrum data
    target = get_data_fit(obj_fit, lims=lims_fit, display_plots=display_plots, title='Row spectrum', subtitle=ch_obj, n=1)
    # correct for inclination angle
    angle, target.data = angle_correction(target.data, angle=angle, display_plots=display_plots)
    # display the target image
    if display_plots == True:
        showfits(target.data, title='Rotated image', subtitle=ch_obj+f', exp. time: {target.get_exposure()} s', n=4)
        plt.show()
    return target, angle

def extract_cal_data(ch_obs: Literal['17-03-27','18-11-27','22-07-26_ohp','22-07-27_ohp','23-03-28'], sel_cal: Literal['dark','flat','bias','all'] = 'all', angle: float | None = 0) -> list[Spectrum]:
    """To get data of dark, flat or bias, if any

    Parameters
    ----------
    ch_obs : str
        selected night
    sel_cal : Literal['dark','flat','bias','all'], optional
        selected kind of calibration file, by default 'all'
        If `sel_cal == 'all'` function returns all possible files
    angle : float | None, optional
        inclination angle of the slit, by default `0`
        If `angle is None` then the value is estimated by a 
        fitting rountine, see `stuff.angle_correction()`

    Returns
    -------
    results : list[Spectrum]
        function returns different results depending on the 
        parameter `sel_cal` and the selected night of 
        observation. If there are different acquisitions of 
        dark, flat or bias, the mean is computed

    Raises
    ------
    Exception
        _description_
    Exception
        _description_
    """
    # initialize some quantities
    results = []
    dark = None
    flat = None
    bias = None
    # only certain nights have information about calibration
    if ch_obs == '17-03-27':
        # collect information
        calibration, lims = collect_fits(ch_obs,'calibrazione')
        if sel_cal in ['dark', 'all']:
            # average on different acquisitions
            mean_dark = Spectrum.empty()
            dark = calibration['dark']
            for d in dark:
                d = data_file_path(ch_obs,'calibrazione',d)
                tmp, angle = get_data('dark', d, angle=angle) 
                mean_dark.hdul += [tmp.hdul]
                mean_dark.data += [tmp.data]
            mean_dark.data = np.mean(mean_dark.data,axis=0)
            # store the result
            results += [mean_dark]
        if sel_cal in ['flat', 'all']:
            # average on different acquisitions
            mean_flat = Spectrum.empty()
            flat = calibration['flat'] 
            for (f, lim) in zip(flat,lims):
                f = data_file_path(ch_obs,'calibrazione',f)
                tmp, angle = get_data('flat', f, lims_fit=lim, angle=angle) 
                mean_flat.hdul += [tmp.hdul]
                mean_flat.data += [tmp.data]
                mean_flat.lims = lim
            mean_flat.data = np.mean(mean_flat.data,axis=0)
            # store the result
            results += [mean_flat]
        if sel_cal in ['bias', 'all']:
            # average on different acquisitions
            master_bias = Spectrum.empty() 
            bias = calibration['bias'] 
            for b in bias:
                b = data_file_path(ch_obs,'calibrazione',b)
                tmp, angle = get_data('bias', b, angle=angle) 
                master_bias.hdul += [tmp.hdul]
                master_bias.data += [tmp.data]
            master_bias.data = np.mean(master_bias.data,axis=0)
            # store the result
            results += [master_bias]
    elif ch_obs in ['18-11-27','22-07-26_ohp','22-07-27_ohp']:
        calibration, lims = collect_fits(ch_obs,'Calibration')
        if sel_cal in ['flat', 'all']:
            flat = data_file_path(ch_obs,'Calibration',calibration)
            flat, angle = get_data('flat', flat, lims, angle=angle) 
            results += [flat]
        else: raise Exception(f'No {sel_cal} for this observation')
    elif ch_obs == '23-03-28':
        calibration, lims = collect_fits(ch_obs,'Calibration')
        if sel_cal == 'bias': raise Exception(f'No {sel_cal} for this observation')
        if sel_cal in ['dark', 'all']:
            # average on different acquisitions
            mean_dark = Spectrum.empty()
            dark = calibration['dark'] 
            for d in dark:
                d = data_file_path(ch_obs,'Calibration',d)
                tmp, angle = get_data('dark', d, angle=angle) 
                mean_dark.hdul += [tmp.hdul]
                mean_dark.data += [tmp.data]
            mean_dark.data = np.mean(mean_dark.data,axis=0)
            # store the result
            results += [mean_dark]
        if sel_cal in ['flat', 'all']:
            # average on different acquisitions
            mean_flat = Spectrum.empty()
            flat = calibration['flat'] 
            for (f, lim) in zip(flat,lims):
                f = data_file_path(ch_obs,'Calibration',f)
                tmp, angle = get_data('flat', f, lims_fit=lim, angle=angle) 
                mean_flat.hdul += [tmp.hdul]
                mean_flat.data += [tmp.data]
                mean_flat.lims = lim
            mean_flat.data = np.mean(mean_flat.data,axis=0)
            # store the result
            results += [mean_flat]
    return results



def extract_data(ch_obs: str, ch_obj: str, selection: int | Literal['mean'], angle: float | None = 0, display_plots: bool = True, **kwargs) -> tuple[Spectrum, Spectrum]:
    """To get data of target and calibration lamp spectrum.
    
    Parameters
    ----------
    ch_obs : str
        chosen obeservation night
    ch_obj : str
        chosen target name
    selection : int | Literal['mean']
        if there are more than one acquisition it is possible to select 
        one of them (`selection` is the number of the chosen acquisition)
        or to average on them (`selection == 'mean'`) 
    angle : float | None, optional
        inclination angle of the slit, by default `0`
        If `angle is None` then the value is estimated by a 
        fitting rountine, see `stuff.angle_correction()`
    display_plots : bool, optional
        parameter to plot data, by default `True`

    Returns
    -------
    target : Spectrum
        information of the target spectrum
    lamp : Spectrum
        information of the calibration lamp spectrum
    
    """
    ## Paths
    # get the information about the paths
    obj, lims = collect_fits(ch_obs, ch_obj)
    # each observation night has different ways of acquisition
    if ch_obs in NIGHTS[:4]:
        # collect info of the target
        obj_fit, lims_fit = obj[0], lims[:-1]
        # Aldebaran has no lamp data
        if ch_obj != 'Aldebaran':
            # collect info of the lamp
            obj_lamp, lims_lamp = obj[1], lims[-1]
        else:
            obj_lamp, lims_lamp = None, None
    elif ch_obs in NIGHTS[4:]:
        # collect info of the target and the lamp
        obj_fit, lims_fit = obj[0], lims[:-1]
        obj_lamp, lims_lamp = obj[1], lims[-1]
        print(lims_fit,lims_lamp)
    ## Data
    # collect the target data
    if isinstance(obj_fit, list):
        # for more acquisitions it is possible to select one or average on all
        if isinstance(selection, int):
            obj_fit, lims_fit = obj_fit[selection], lims_fit[selection]
            obj_fit = data_file_path(ch_obs, ch_obj, obj_fit)
            target, angle = get_data(ch_obj,obj_fit,lims_fit,angle=angle,display_plots=display_plots,**kwargs)
        elif selection == 'mean':
            target = Spectrum.empty()
            for (fit, lim) in zip(obj_fit, lims_fit):
                fit = data_file_path(ch_obs, ch_obj, fit)
                tmp, angle = get_data(ch_obj,fit,lim,angle=angle,display_plots=display_plots,**kwargs)
                target.hdul += [tmp.hdul]
                target.data += [tmp.data]
                target.lims = lim
            target.data = np.mean(target.data, axis=0)
    else:
        obj_fit = data_file_path(ch_obs, ch_obj, obj_fit)
        target, angle = get_data(ch_obj,obj_fit,lims_fit,angle=angle,display_plots=display_plots,**kwargs)
    # collect the lamp data, if any
    if obj_lamp is not None:
        obj_lamp = data_file_path(ch_obs, ch_obj, obj_lamp)
        lamp, angle = get_data(ch_obj,obj_lamp,lims_lamp,angle=angle,display_plots=display_plots,**kwargs)
    else: 
        lamp = Spectrum.empty()
    return target, lamp       

def quick_view(ch_obs: str, ch_obj: str, selection: int | Literal['mean'],**kwargs) -> None:
    if ch_obj in ['dark','flat','bias','all']:
        results = extract_cal_data(ch_obs,sel_cal=ch_obj)
        for obj in results:
            obj = obj.data
            plt.figure(1)
            showfits(obj,n=1,**kwargs)
            plt.show()
    else:
        results = extract_data(ch_obs,ch_obj,selection,display_plots=False)
        for obj in results:
            obj = obj.hdul[0].data
            plt.figure(1)
            showfits(obj,n=1,**kwargs)
            plt.show()