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
from .stuff import make_cut_indicies, Spectrum, mean_n_std
from .display import *
from numpy.typing import ArrayLike
from typing import Sequence, Literal
from numpy import ndarray


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

BALMER = [6562.79, 4861.350, 4340.472, 4101.734, 3970.075, 3889.064, 3835.397, 3797.909, 3770.633, 3750.151]
BALERR = [0.03,0.05]+[0.006]*7


def get_cal_lines(ch_obs: str, ch_obj: str) -> ndarray:
    """To load the edges of the image cut

    Parameters
    ----------
    ch_obs : str
        chosen obeservation night
    ch_obj : str
        chosen target name

    Returns
    -------
    data : ndarray
        information about the cut
    """
    file_path = os.path.join(DATA_DIR,ch_obs,ch_obj,'calibration_lines.txt')
    data = np.loadtxt(file_path, unpack=True)
    return data

def get_balm_lines(ch_obs: str, ch_obj: str) -> ndarray:
    """To get data for the calibration of balmer serie

    Parameters
    ----------
    ch_obs : str
        chosen obeservation night
    ch_obj : str
        chosen target name

    Returns
    -------
    data : ndarray
        balmer lines to calibrate
    """
    file_path = os.path.join(DATA_DIR,ch_obs,ch_obj,'H_calibration.txt')
    data = np.loadtxt(file_path, unpack=True)
    return data

def get_standard(name: str = 'Vega', sel: int = 0, diagn_plots: bool = False,**pltargs) -> tuple[ndarray, ndarray]:
    """To get data of the standard for the absolute calibration

    Parameters
    ----------
    name : str, optional
        name of chosen standard, by default `'Vega'`
    sel : int, optional
        to select the wanted data, by default `0`
    diagn_plots : bool, optional
        if it is `True` images/plots are displayed, by default `False`

    Returns
    -------
    wlen : ndarray
        wavelengths in A
    data : ndarray
        corresponding absolute surface flux values in erg/s/cm^2/A
    """
    dir_path = os.path.join(CAL_DIR,'standards',name)
    if 'fontsize' not in pltargs.keys():
        pltargs['fontsize'] = 18
    fontsize = pltargs['fontsize']
    pltargs.pop('fontsize')
    if 'figsize' not in pltargs.keys():
        pltargs['figsize'] = (13,10)
    figsize = pltargs['figsize']
    pltargs.pop('figsize')
    if name == 'Vega':
        file_name = 'vega_std.txt'

        wlen = np.empty(0)
        data = np.empty(0)

        data_file = os.path.join(dir_path,file_name)
        l, spec,_ = np.loadtxt(data_file, unpack=True)
        wlen = np.append(wlen,l)
        data = np.append(data,spec)*1e16 # erg/cm²/s/A 

        pos = np.argsort(wlen)
        wlen = wlen[pos]
        data = data[pos]
        name = 'alf Lyr'
        
    if diagn_plots:
        plt.figure(figsize=figsize)
        plt.title('Standard Spectrum of ' + name,fontsize=fontsize+2)
        plt.plot(wlen,data,'.-')
        plt.grid()
        plt.ylabel('$I_{std}$ [erg/(s cm$^2$ $\\AA$)]',fontsize=fontsize)
        plt.xlabel('$\\lambda$ [$\\AA$]',fontsize=fontsize)
        plt.show()
    return wlen, data

def store_results(file_name: str, data: ArrayLike, ch_obs: str, ch_obj: str, **txtkw) -> None:
    """To store results in a `.txt` file

    Parameters
    ----------
    file_name : str
        name of the file
    data : ArrayLike
        data to store
    ch_obs : str
        chosen observation
    ch_obj : str
        chosen target name
    **txtkw
        parameters of `numpy.savetxt()`
        the parameter `'delimiter'` is set to `'\t'` by default
    """
    # check delimiter
    if 'delimiter' not in txtkw.keys():
        txtkw['delimiter'] = '\t'
    # build the path
    p_dir = RESULT_DIR
    for directory in [ch_obs,ch_obj]:
        n_dir = os.path.join(p_dir,directory)
        if not os.path.isdir(n_dir): os.mkdir(n_dir)
        p_dir = n_dir
    file_path = os.path.join(n_dir, file_name + '.txt')
    # save data
    np.savetxt(file_path, np.column_stack(data), **txtkw)


def open_targets_list(filename: str = 'targets.csv', delimiter: str = ',') -> ndarray:
    """To collect chosen targets

    Parameters
    ----------
    filename : str, optional
        file name with the list of targets, by default `'targets.csv'`
    delimiter : str, optional
        delimiter of columns, by default `','`

    Returns
    -------
    data : ndarray
        list of chosen targets
    """
    from pandas import read_csv
    TARGET_FILE = os.path.join(DATA_DIR,filename)
    data = read_csv(TARGET_FILE, delimiter=delimiter).to_numpy()
    return data
    

def collect_fits(night: str, obj: str) -> tuple[ndarray, ArrayLike]:
    """To collect data fits for a chosen night observation and object.

    Parameters
    ----------
    night : str
        selected night
    obj : str
        name of the target

    Returns
    -------
    extracted : ndarray
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
    night : str
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
def get_data_fit(path: str, lims: Sequence[int | None | list[int]] = [0,None,0,None], hotpx: bool = True, obj_name: str = '', check_edges: bool = True, diagn_plots: bool = True, **figargs) -> Spectrum:
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
    obj_name: str, optional
        name of the target, by default `''`
    check_edges: bool, optional
        parameter to check edges values, by default `True`
    diagn_plots : bool, optional
        parameter to plot data, by default `True`
    **figargs
        parameters for the images, see `display.showfits()`

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
    # print fits file info
    hdul.info()
    # data extraction
    #.. format -> data[Y,X]
    data = hdul[0].data
    if len(lims) == 1: lims = lims[0]
    cut = [0, None, 0, None]    #: edges fot the cut
    # edges before and after the inclination correction 
    if obj_name not in ['flat','dark','bias']:
        cut = lims[:4]      #: edges before inclination correction
        lims = lims[4:]     #: edges after inclination correction
        print('NEWS',cut, lims)
    # store in `Spectrum` class
    target = Spectrum(hdul, data, lims=lims, cut=cut, hotpx=hotpx, name=obj_name, check_edges=check_edges)
    # print the header
    target.print_header()
    # display target image
    if diagn_plots: 
        _ = show_fits(target, **figargs) 
        plt.axhline(500,0,1)
    return target
##*


def extract_cal_data(ch_obs: Literal['17-03-27','18-11-27','22-07-26_ohp','22-07-27_ohp','23-03-28'], sel_cal: Literal['dark','flat','bias','all'] = 'all', display_plot: bool = False, **figargs) -> list[Spectrum]:
    """To get data of dark, flat or bias, if any

    Parameters
    ----------
    ch_obs : str
        selected night
    sel_cal : Literal['dark','flat','bias','all'], optional
        selected kind of calibration file, by default 'all'
        If `sel_cal == 'all'` function returns all possible files
    diagn_plots : bool, optional
        parameter to plot data, by default `True`
    **figargs
        parameters for the images, see `display.showfits()`

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
    # only certain nights have information about calibration
    if ch_obs == '17-03-27':
        # collect information
        calibration, lims = collect_fits(ch_obs,'calibrazione')
        if sel_cal in ['flat', 'all']:
            # average on different acquisitions
            mean_flat = Spectrum.empty()
            flat = calibration['flat'] 
            for (f, lim) in zip(flat,lims):
                f = data_file_path(ch_obs,'calibrazione',f)
                tmp = get_data_fit(f, lims=lim, obj_name='flat',diagn_plots=False,**figargs) 
                mean_flat.hdul += [tmp.hdul]
                mean_flat.data += [tmp.data]
                mean_flat.lims = lim
            mean_flat.update_header()
            mean_flat.data, mean_flat.sigma = mean_n_std(mean_flat.data,axis=0)
            mean_flat.name = 'Mean Flat'
            if display_plot: _ = show_fits(mean_flat)
            # store the result
            results += [mean_flat]
        if sel_cal in ['dark', 'all']:
            # average on different acquisitions
            mean_dark = Spectrum.empty()
            dark = calibration['dark']
            for d in dark:
                d = data_file_path(ch_obs,'calibrazione',d)
                tmp = get_data_fit(d, obj_name='dark',check_edges=False,diagn_plots=False, **figargs) 
                mean_dark.hdul += [tmp.hdul]
                mean_dark.data += [tmp.data]
            mean_dark.update_header()
            mean_dark.data, mean_dark.sigma = mean_n_std(mean_dark.data,axis=0)
            mean_dark.name = 'Mean Dark'
            if display_plot: _ = show_fits(mean_dark, **figargs)
            # store the result
            results += [mean_dark]
        if sel_cal in ['bias', 'all']:
            # average on different acquisitions
            master_bias = Spectrum.empty() 
            bias = calibration['bias'] 
            for b in bias:
                b = data_file_path(ch_obs,'calibrazione',b)
                tmp = get_data_fit(b, obj_name='bias',check_edges=False,diagn_plots=False, **figargs) 
                master_bias.hdul += [tmp.hdul]
                master_bias.data += [tmp.data]
            master_bias.update_header()
            master_bias.data, master_bias.sigma = mean_n_std(master_bias.data,axis=0)
            master_bias.name = 'Master Bias'
            if display_plot: _ = show_fits(master_bias, **figargs)
            # store the result
            results += [master_bias]
    elif ch_obs in ['18-11-27','22-07-26_ohp','22-07-27_ohp']:
        calibration, lims = collect_fits(ch_obs,'Calibration')
        if sel_cal in ['flat', 'all']:
            flat = data_file_path(ch_obs,'Calibration',calibration)
            flat = get_data_fit(flat, lims, obj_name='flat',diagn_plots=False, **figargs) 
            flat.name = 'flat'
            if display_plot: _ = show_fits(flat, **figargs)
            results += [flat]
        else: raise Exception(f'No {sel_cal} for this observation')
    elif ch_obs == '23-03-28':
        calibration, lims = collect_fits(ch_obs,'Calibration')
        if sel_cal == 'bias': raise Exception(f'No {sel_cal} for this observation')
        if sel_cal in ['flat', 'all']:
            # average on different acquisitions
            mean_flat = Spectrum.empty()
            flat = calibration['flat'] 
            for (f, lim) in zip(flat,lims):
                f = data_file_path(ch_obs,'Calibration',f)
                tmp = get_data_fit(f, lims=lim, obj_name='flat',diagn_plots=False, **figargs) 
                mean_flat.hdul += [tmp.hdul]
                mean_flat.data += [tmp.data]
                mean_flat.lims = lim
            mean_flat.data, mean_flat.sigma = mean_n_std(mean_flat.data,axis=0)
            mean_flat.name = 'Mean Flat'
            if display_plot: _ = show_fits(mean_flat,title='Mean Flat', **figargs)
            # store the result
            results += [mean_flat]
        if sel_cal in ['dark', 'all']:
            # average on different acquisitions
            mean_dark = Spectrum.empty()
            dark = calibration['dark'] 
            for d in dark:
                d = data_file_path(ch_obs,'Calibration',d)
                tmp = get_data_fit(d, obj_name='dark',check_edges=False,diagn_plots=False, **figargs) 
                mean_dark.hdul += [tmp.hdul]
                mean_dark.data += [tmp.data]
            mean_dark.data, mean_dark.sigma = mean_n_std(mean_dark.data,axis=0)
            mean_dark.name = 'Mean Dark'
            if display_plot: _ = show_fits(mean_dark,title='Mean Dark', **figargs)
            # store the result
            results += [mean_dark]
    # check the chosen night
    else: raise Exception(f'{ch_obs} has no calibration file')
    plt.show()
    return results


def extract_data(ch_obs: str, ch_obj: str, selection: str | Literal['mean'], obj_name: str = '', diagn_plots: bool = True, **figargs) -> tuple[Spectrum, Spectrum]:
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
    diagn_plots : bool, optional
        parameter to plot data, by default `True`
    **figargs
        parameters for the images, see `display.showfits()`

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
    if obj_name == '': obj_name = ch_obj
    # collect the target data
    if isinstance(obj_fit, list):
        # for more acquisitions it is possible to select one or average on all
        if selection == 'mean':
            exp_times = []
            target = Spectrum.empty()
            for (fits, lim) in zip(obj_fit, lims_fit):
                fits = data_file_path(ch_obs, ch_obj, fits)
                tmp = get_data_fit(fits,lim,obj_name=obj_name,diagn_plots=diagn_plots,**figargs)
                exptime = tmp.hdul[0].header['EXPTIME']
                if (len(exp_times) != 0) and (exptime not in exp_times):
                    raise ValueError(f'Cannot average, different exposure times -> {fits}')
                exp_times += [exptime]
                target.hdul += [tmp.hdul]
                target.data += [tmp.data]
                # check edges
                target.lims = lim[4:]
                target.cut  = lim[:4]
            target.update_header()
            target.data, target.sigma = mean_n_std(target.data,axis=0)
            target.format_ends()
            print('LIMS',target.cut,target.lims)
            target.name = tmp.name
        elif int(selection) in range(len(obj_fit)):
            selection = int(selection)
            obj_fit, lims_fit = obj_fit[selection], lims_fit[selection]
            obj_fit = data_file_path(ch_obs, ch_obj, obj_fit)
            target = get_data_fit(obj_fit,lims_fit,obj_name=obj_name,diagn_plots=diagn_plots,**figargs)
        else: raise Exception('Invalid value for `selection`: '+ selection +' is not allowed')
    else:
        obj_fit = data_file_path(ch_obs, ch_obj, obj_fit)
        target = get_data_fit(obj_fit,lims_fit,obj_name=obj_name,diagn_plots=diagn_plots,**figargs)
    if diagn_plots: _ = show_fits(target, title='Target spectrum extracted', show=True,**figargs)
    # collect the lamp data, if any
    if obj_lamp is not None:
        obj_lamp = data_file_path(ch_obs, ch_obj, obj_lamp)
        lamp = get_data_fit(obj_lamp,lims_lamp,obj_name='Lamp of ' + obj_name, diagn_plots=diagn_plots,**figargs)
        if diagn_plots:  _ = show_fits(lamp, show='True',**figargs)
    else: 
        lamp = Spectrum.empty()
    return target, lamp       


def open_results(file_name: str | Sequence[str], ch_obs: str, ch_obj: str) -> ndarray | list[ndarray]:
    """To get data of results

    Parameters
    ----------
    file_name : str | Sequence[str]
        name(s) of the file(s) to open 
    ch_obs : str
        chosen observation night
    ch_obj : str
        chosen target name

    Returns
    -------
    results : ndarray | list[ndarray]
        selected data or in case of multiple files list of data
    """
    results = []    #: list to store load data
    # if only one file has to be opened
    if isinstance(file_name, str): file_name = [file_name]
    # load and collect data
    for name in file_name:
        file_path = os.path.join(RESULT_DIR, ch_obs, ch_obj, name + '.txt')
        data = np.loadtxt(file_path,unpack=True)
        results += [data]
    if len(results) == 1: results = results[0]
    return results