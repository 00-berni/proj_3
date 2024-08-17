"""
CALCORR PACKAGE
===============

***

::METHODS::
-----------

***

!TO DO!
-------
    - [] **Update `compute_flat` and `calibration`**


***
    
?WHAT ASK TO STEVE?
-------------------
"""
import os
import numpy as np
from numpy import ndarray
from typing import Callable, Literal
from scipy import odr
from .display import *
from .data import get_data_fit, extract_data, extract_cal_data, get_cal_lines
from .stuff import FuncFit

def compute_master_dark(mean_dark: Spectrum | None, master_bias: Spectrum | None = None, display_plots: bool = False) -> Spectrum:
    """To compute master dark

    Parameters
    ----------
    mean_dark : Spectrum | None
        averaged dark data
    master_bias : Spectrum | None, optional
        averaged bias data, by default `None`
    display_plots : bool, optional
        to plot figures, by default `False`

    Returns
    -------
    master_dark : Spectrum
        dark corrected
    
    Notes
    -----
    According to [] (see README), master dark is defined as
    ```
        mean_dark - master_bias
    ```
    """
    master_dark = mean_dark.copy()
    if master_bias is not None: 
        master_dark.data = mean_dark - master_bias
    # condition to display the images/plots
    if display_plots == True:
        show_fits(master_dark,show=True)
    return master_dark

def compute_master_flat(flat: Spectrum, master_dark: Spectrum | None = None, master_bias: Spectrum | None = None, display_plots: bool = False) -> Spectrum:
    """To estimate the flat gain

    Parameters
    ----------
    flat : Spectrum
        flat data
    master_dark : Spectrum | None, optional
        master dark if any, by default `None`
    master_bias : Spectrum | None, optional
        master bias if any, by default `None`
    display_plots : bool, optional
        to plot figures, by default `False`

    Returns
    -------
    master_flat : Spectrum
        estimated master flat
    
    Notes
    -----
    According to [] (see README), master flat is defined as
    ```
        flat = flat - mean_dark - master_bias
        master_flat = flat / mean(flat)
    ```
    If bias and/or dark data are not given the first step is skipped
    """
    # correct by bias, if any
    if master_bias is not None: flat.data = flat - master_bias
    # correct by dark, if any
    if master_dark is not None: 
        # check the exposure times
        fl_exp = flat.get_exposure()
        dk_exp = master_dark.get_exposure()
        if fl_exp != dk_exp:
            master_dark.data = master_dark.data / dk_exp * fl_exp
        flat.data = flat - master_dark 
    
    master_flat = flat.copy()
    master_flat.data = flat.data / np.mean(flat.data)
    master_flat.name = 'Master Flat'
    # condition to display the images/plots
    if display_plots == True:
        show_fits(master_flat,show=True)
    return master_flat

def get_target_data(ch_obs: str, ch_obj: str, selection: int | Literal['mean'], cut: bool = True, angle: float | None = 0, display_plots: bool = False,**kwargs) -> tuple[Spectrum, Spectrum]:
    """To get the science frames of target and its calibration lamp

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
    cut : bool, optional
        to cut the image, by default `True`
    angle : float | None, optional
        inclination angle of the slit, by default `0`
        If `angle is None` then the value is estimated by a 
        fitting rountine
    display_plots : bool, optional
        to plot figures, by default `False`

    Returns
    -------
    target : Spectrum 
        science frame of the target
    lamp : Spectrum
        science frame of its calibration lamp
    """
    ## Data extraction
    # get the light frames
    target, lamp = extract_data(ch_obs,ch_obj,selection,display_plots=display_plots,**kwargs)
    if not display_plots: 
        show_fits(target, title='Light Frame',**kwargs)
        if lamp.name != 'empty':
            show_fits(lamp, title='Light Frame',**kwargs)
        plt.show()
    
    ## Calibration correction
    # if there is calibration information
    # try:
    calibration = extract_cal_data(ch_obs)
    if len(calibration) > 1:
        if len(calibration) == 3:   #: in this case `calibration = [flat, dark, bias]` 
            # compute master dark
            calibration[1] = compute_master_dark(*calibration[1:],display_plots=display_plots,**kwargs)
            # bias correction
            target.data = target - calibration[2]
            if lamp.name != 'empty':
                lamp.data = lamp - calibration[2]
        # check the exposure times
        tg_exp = target.get_exposure()
        dk_exp = calibration[1].get_exposure()
        if tg_exp != dk_exp:
            calibration[1].data = calibration[1].data / dk_exp * tg_exp
        # dark correction
        target.data = target - calibration[1]
        if lamp.name != 'empty':
            lamp.data = lamp - calibration[1]
    # compute master flat
    master_flat = compute_master_flat(*calibration,display_plots=display_plots,**kwargs)
    print('MIN',master_flat.data.min())
    # flat correction
    target.data = target.data / master_flat.data
    if lamp.name != 'empty':
        lamp.data = lamp.data / master_flat.data  
    # except Exception:
    #     print('!EXception')
    #     pass
    if cut:    
        target.cut_image()
        target, angle = target.angle_correction(angle=angle)      
        if lamp.name != 'empty':
            lamp.cut_image()
            lamp, _ = lamp.angle_correction(angle=angle)      
    show_fits(target, title='Science Frame',**kwargs)
    if lamp.name != 'empty':
        show_fits(lamp, title='Science Frame',**kwargs)
    plt.show()
    return target, lamp

def lines_calibration(ch_obs: str, ch_obj: str, trsl: int, ord: int = 2, initial_values: Sequence[float] | None = None) -> tuple[Callable[[ndarray], ndarray], Callable[[ndarray],ndarray]]:
    """To compute the calibration function

    Parameters
    ----------
    ch_obs : str
        chosen obeservation night
    ch_obj : str
        chosen target name
    trsl : int
        values of the pixels have to be traslated by
        the value of pixel at which the image is cut
    ord : int, optional
        order of the polynomial function used for the
        fit, by default `2`
    initial_values : Sequence[float] | None, optional
        inital values for the fit, by default `None`

    Returns
    -------
    px_to_arm : Callable[[ndarray], ndarray] 
        function to convert pixel values in armstrong
    err_func : Callable[[ndarray],ndarray]
        uncertainties related to the estimated function values
    """
    ## Data
    # extract the data for the fit
    lines, pxs, errs = get_cal_lines(ch_obs,ch_obj)
    # traslate the pixels
    pxs += trsl
    Dlines = np.full(lines.shape,3.63)      #: uncertainties of the lines in armstrong
    if initial_values is None:
       initial_values = [0] + [1]*(ord-1) + [np.mean(pxs)]
    
    ## Fit
    fit = FuncFit(xdata=pxs,ydata=lines,xerr=errs,yerr=Dlines)
    fit.pol_fit(ord=ord,initial_values=initial_values)
    pop, _ = fit.results()
    cov = fit.res['cov']
    
    ## Functions 
    px_to_arm = lambda x : fit.res['func'](x, *pop)     #: function to pass from px to A
    # compute the function to evaluate the uncertainty associated with `px_to_arm`
    def err_func(x):
        err = [ x**(2*ord-(i+j)) * cov[i,j] for i in range(ord+1) for j in range(ord+1)]
        return np.sum(err, axis=0)
    
    ## Plot
    plt.figure()
    pp = np.linspace(pxs.min(),pxs.max(),200)
    plt.errorbar(pxs,lines,Dlines,errs,'.',color='orange')
    plt.plot(pp,px_to_arm(pp))
    plt.figure()
    plt.errorbar(pxs,lines-fit.res['func'](pxs,*pop),Dlines,errs,'.',color='orange')
    plt.axhline(0,0,1)
    plt.show()
    return px_to_arm, err_func

def lamp_correlation(lamp1: Spectrum, lamp2: Spectrum) -> float:
    """To compute the lag to pass from a lamp to another

    Parameters
    ----------
    lamp1 : Spectrum
        first lamp
    lamp2 : Spectrum
        second lamp

    Returns
    -------
    shift : float
        lag between the two lamps
        If it is negative, `lamp1` has to be shifted
        If it is positive, vice versa
    """
    ## Data elaboration 
    # prevent errors
    lamp1 = lamp1.copy()
    lamp2 = lamp2.copy()
    # data must have same length
    edge1 = lamp1.lims[2:]
    edge2 = lamp2.lims[2:]
    edge = edge1 - edge2
    edge1 = [-edge[0] if edge[0] < 0 else 0, edge[1] if edge[1] < 0 else None]
    edge2 = [edge[0] if edge[0] > 0 else 0, -edge[1] if edge[1] > 0 else None]
    lamp1 = lamp1.spec[slice(*edge1)].copy()
    lamp2 = lamp2.spec[slice(*edge2)].copy()

    ## Correlation procedure
    # normalize the data
    lamp1 = lamp1.copy()/lamp1.max()
    lamp2 = lamp2.copy()/lamp2.max()
    # compute cross-correlation
    from scipy.signal import correlate
    corr = correlate(lamp1,lamp2)
    print('MAX POS: ',np.argmax(corr)/len(corr)*100, len(corr) // np.argmax(corr))
    quickplot(corr,title=f'Cross corr -> {np.argmax(corr)/len(corr)*100} : {len(corr) // np.argmax(corr)}')
    plt.show()
    # compute the lag
    shift = np.argmax(corr) - (len(corr)/2)  
    return shift

def calibration(ch_obs: str, ch_obj: str, selection: int | Literal['mean'], angle: float | None = None, height: int | None = None, other_lamp: Spectrum | None = None, ord: int = 2, initial_values: Sequence[float] | None = None, display_plots: bool = False, **kwargs) -> tuple[Spectrum, Spectrum]:
    """To open and calibrate data

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
        inclination angle of the slit, by default `None`
        If `angle is None` then the value is estimated by a 
        fitting rountine
    height : int | None, optional
        height at which the lamp spectrum is taken, by default `None`
        If `height is None`, the mean height is taken 
    other_lamp : Spectrum | None, optional
        parameter for the computing of the calibration function, by default `None`
          * If `other_lamp is None`, the calibration function is obtained by a fit,
          using the lines in `data_files/calibration/SpecrteArNeLISA.pdf` file
          * Otherwise instead of sampling the lamp, the cross-correlation with a 
          sampled previous one is computed and the data is shifted by the estimated
          lag
    ord : int, optional
        order of the polynomial function used for the
        fit, by default `2`
    initial_values : Sequence[float] | None, optional
        inital values for the fit, by default `None`
    display_plots : bool, optional
        if it is `True` images/plots are displayed, by default `False`

    Returns
    -------
    target : Spectrum 
        calibrated science frame of the target
    lamp : Spectrum
        calibrated science frame of its lamp
    """
    ## Data
    # extract data
    target, lamp = get_target_data(ch_obs, ch_obj, selection, angle=angle, display_plots=display_plots,**kwargs)
    # average along the y axis
    target.spec = np.mean(target.data, axis=0)
    # compute the height for the lamp
    if height is None: height = int(len(lamp.data)/2)
    # take lamp spectrum at `height` 
    lamp.spec = lamp.data[height]
    if display_plots:
        quickplot(target.spec,title='Uncalibrated spectrum of '+target.name,labels=('x [a.u.]','I [a.u.]'),numfig=1)
        quickplot(lamp.spec,title='Uncalibrated spectrum of its lamp',labels=('x [a.u.]','I [a.u.]'),numfig=2)
        plt.show()
    
    ## Calibration
    if other_lamp is None:      
        # compute the calibration function via fit
        cal_func, err_func = lines_calibration(ch_obs, ch_obj, trsl=lamp.lims[2], ord=ord)
        # store results
        lamp.func = [cal_func, err_func]
        target.func = [*lamp.func]
        # compute the values in armstrong
        lamp.compute_lines()
        target.compute_lines()
        # plot
        quickplot((lamp.lines,lamp.spec),labels=('$\\lambda$ [$\\AA$]','I [a.u.]'),numfig=3,**kwargs)
        plt.errorbar(lamp.lines, lamp.spec,xerr=lamp.errs,fmt='.')
    else:
        # compute the lag between the two lamps
        shift = lamp_correlation(lamp, other_lamp)
        # store the functions
        lamp.func = [*other_lamp.func]
        target.func = [*other_lamp.func]
        # compute the values in armstrong
        lamp.compute_lines(shift=shift)
        target.compute_lines(shift=shift)
        # plot
        quickplot((lamp.lines,lamp.spec),labels=('$\\lambda$ [$\\AA$]','I [a.u.]'),numfig=3,**kwargs)
        plt.errorbar(lamp.lines, lamp.spec,xerr=lamp.errs,fmt='.')
    quickplot((target.lines,target.spec),labels=('$\\lambda$ [$\\AA$]','I [a.u.]'),numfig=4,**kwargs)
    plt.show()        
    return target, lamp


def calibrated_spectrum(ch_obs: int, ch_obj: str, flat: None | ndarray = None, cal_func: Callable[[ndarray],ndarray] | None = None, err_func: Callable[[ndarray,ndarray,bool],ndarray] | None = None, display_plots: bool = False, initial_values: list[float] | tuple[float] = [3600, 2.6,0.], ret_values: str = 'few') -> list[ndarray] | list[ndarray | dict]:
    """Getting the spectrum of a selceted target for a chosen observation night

    Parameters
    ----------
    ch_obs : int
        chosen observation night
    ch_obj : str
        chosen obj
    flat : None | ndarray, optional
        if the flat gain is not evaluated yet, the flat target name is passed, by default `None`
    cal_func : Callable[[ndarray],ndarray] | None, optional
        if it is None, the calibration function will be computed, by default `None`
    err_func : Callable[[ndarray,ndarray,bool],ndarray] | None, optional
        _description_, by default `None`
    display_plots : bool, optional
        if it is True images/plots are displayed, by default `False`
    initial_values : list[float] | tuple[float], optional
        _description_, by default `[3600, 2.6,0.]`
    ret_values : str, optional
        _description_, by default `'few'`

    Returns
    -------
    spectrum : ndarray
        cumulative spectrum 
    lengths : ndarray
        corrisponding wavelenghts
    data : dict[str,ndarray], optional
        information about the image
            * `'hdul'` : fits information
            * `'sp_data'` : spectrum image data
        It is returned only if `ret_values == 'data' or 'all'` 
    cal_data : dict[str, float | ndarray | Callable], optional
        information about calibration
            * `'angle'` : correction angle value
            * `'flat_value'` : estimate flat value
            * `'func'` : calibration function
            * `'err'` : function to compute the uncertainties associated with `'func'`
        It is returned only if `ret_values == 'calibration' or 'all'` 

    Notes
    -----
    The function extracts fits data for a target and evaluates inclination correction, flat gain and calibration function to return the
    calibrated spectrum. If the flat gain or the calibration function are already computed then one can pass them to the function, 
    avoiding an other estimation.
        
    It calls the functions:
      - `extract_data()`
      - `get_data()`
      - `showfits()`
      - `quickplot()`
      - `compute_flat()`
      - `calibration()`

    """
    # collecting data
    obj_fit, lims_fit, obj_lamp, lims_lamp = extract_data(ch_obs, ch_obj,sel=['obj','lamp'])
    # extracting fits data and correcting for inclination
    hdul, sp_data, angle = get_data(ch_obj,obj_fit,lims_fit, display_plots=display_plots)
    if display_plots == False: 
        showfits(sp_data, title='Spectrum of '+ ch_obj)
        plt.show() 

    # computing the cumulative spectrum over columns
    spectrum = sp_data.sum(axis=0)
    if display_plots == True:
        quickplot(np.arange(len(spectrum)), spectrum, title='Cumulative spectrum', labels=['x','counts'],grid=True)
        plt.show()

    ret_cond = (flat is None) and (cal_func is None) and (err_func is None)

    # estimating the flat gain
    flat_value = compute_flat(ch_obs, display_plots=display_plots) if flat is None else flat
    # correcting for the flat gain
    spectrum = spectrum / flat_value[:len(spectrum)]
    if display_plots == True:
        quickplot(np.arange(len(spectrum)), spectrum, title='Cumulative spectrum corrected by flat', labels=['x','counts'],grid=True)    
        plt.show()
        
    # condition to compute the calibration function
    if cal_func is None:
        # getting the path of the calibration file
        cal_file = os.path.join(os.path.split(obj_fit)[0],'calibration_lines.txt')
        # estimating the calibration function
        cal_func, err_func = calibration(cal_file=cal_file, obj_lamp=obj_lamp, lims_lamp=lims_lamp, angle=angle,initial_values=initial_values,display_plots=display_plots)
    # getting the corrisponding wavelengths
    lengths = cal_func(np.arange(len(spectrum)))
    # displaying the calibrated spectrum
    quickplot(lengths, spectrum,title='Corrected and calibrated spectrum of ' + ch_obj,labels=['$\lambda$ [$\AA$]','counts'],dim=[17,9],grid=True)
    plt.show()

    if (ret_values == 'all') or (ret_values == 'few') or (ret_values == 'data') or (ret_values == 'calibration'):
        results = [spectrum, lengths]
    
    if (ret_values == 'all') or (ret_values == 'data'):
        data = { 'hdul' : hdul,
                 'sp_data' : sp_data }
        results += [data]
    if (ret_values == 'all') or (ret_values == 'calibration'):
        if ret_cond:
            cal_data = { 'angle' : angle,
                         'flat'  : flat_value,
                         'func'  : cal_func,
                         'err'   : err_func }
        else:
            cal_data = angle
        results += [cal_data]
    
    return results


def mean_line(peaks: ndarray, spectrum: ndarray, dist: int = 3, height: int | float = 800) -> tuple[ndarray,ndarray]:
    """    

    Parameters
    ----------
    peaks : ndarray
        peaks
    spectrum : ndarray
        spectrum
    dist : int, optional
        maximum distance between peaks, by default `3`
    height : int | float, optional
        the maximum value to start interpolation, by default `800`

    Returns
    -------
    peaks : ndarray
        average peaks
    spectrum[peaks] : ndarray
        corresponding values of the spectrum
    """
    # making a copy of data
    peaks = np.copy(peaks)
    spectrum = np.copy(spectrum)

    # method to compute the difference of two different spectrum values
    spectral_distance = lambda pos : abs(spectrum[pos[0]].astype(float)-spectrum[pos[1]].astype(float))

    # height estimation routine
    # computing the differences in height around peaks
    height_diff = np.array([ min(spectral_distance((pk,pk+1)), spectral_distance((pk,pk-1)))  for pk in peaks])
    # selecting only differences less than `height` value
    ind_diff = np.where(height_diff <= height)[0]
    if len(ind_diff) != 0:
        # taking the maximum
        height = max(height_diff[ind_diff])

    ## ?
    print(height_diff)
    print(f'height -> {height}')
    ## ?

    # computing the distance between peaks (along x axis)
    pksdiff = np.diff(peaks)
    # class to interpolate
    from scipy.interpolate import CubicSpline
    # condition for 4 points
    if dist == 3:
        # storing the indecies of these peaks
        pos = np.where(pksdiff == dist)[0]
        if len(pos) != 0:
            #: variable to store delectable indecies
            delpos = []
            for i in range(len(pos)):
                x = pos[i]
                # selected peaks
                pk0, pk3 = peaks[x], peaks[x+1]
                # corresponding spectrum values
                sp0, sp3 = spectrum[pk0], spectrum[pk3]
                # points between 2 peaks
                pk1, pk2 = pk0+1, pk0+2
                # corresponding spectrum values
                sp1, sp2 = spectrum[pk1], spectrum[pk2]
                # condition on height
                if max(sp0-sp1,sp3-sp2) <= height:
                    # interpolation routine
                    # defining data
                    xdata = np.array([pk0-1,pk0,pk3,pk3+1])
                    ydata = spectrum[xdata]
                    # computing the interpolation
                    int_line = CubicSpline(xdata,ydata)
                    # changing the peaks position
                    peaks[x] = pk1 if sp0 >= sp3 else pk2
                    # computing the corresponging interpolated value
                    spectrum[peaks[x]] = int_line(peaks[x])
                else:
                    # storing the index
                    delpos += [i]
            # removing unused indecies 
            pos = np.delete(pos,delpos)
            # reducing the number of peaKs
            peaks = np.delete(peaks,pos+1)
            # freeing memory
            del pk0,pk1,pk2,pk3,pos, delpos
        # updating distance to 2
        dist -= 1
    # storing the indecies of these peaks
    pos = np.where(pksdiff == dist)[0]
    if len(pos) != 0:
        #: variable to store delectable indecies
        delpos = []
        for i in range(len(pos)):
            x = pos[i]
            # selected peaks
            pk0, pk2 = peaks[x], peaks[x+1]
            # corresponding spectrum values
            sp0, sp2 = spectrum[pk0], spectrum[pk2]
            # point between 2 peaks
            pk1 = pk0+1
            # corresponding spectrum values
            sp1 = spectrum[pk1]
            # condition on height
            if max(sp0-sp1,sp2-sp1) <= height:
                # interpolation routine
                xdata = np.array([pk0-1,pk0,pk2,pk2+1])
                ydata = spectrum[xdata]
                int_line = CubicSpline(xdata,ydata)
                peaks[x] = pk1
                spectrum[peaks[x]] = int_line(peaks[x])
            else:
                # storing the index
                delpos += [i]
        # removing unused indecies 
        pos = np.delete(pos,delpos)
        # reducing the number of peaKs
        peaks = np.delete(peaks,pos+1)

    return peaks, spectrum[peaks]




def lamp_corr(nights: tuple[int,int] | list[int] | int, objs_name: tuple[str,str] | list[str], angles: tuple[float | None, float | None] | list[float | None, float | None] | float | None = None, height: int | float = 2000, display_plots: bool = False) -> float:
    """    

    Parameters
    ----------
    nights : tuple[int,int] | list[int] | int
        selected observation nights (e.g. `(0,1)` or `0`)
    objs_name : tuple[str,str] | list[str]
        names of the selected targets (e.g. `('betaLyr','vega')`)
    angles : tuple[float  |  None, float  |  None] | list[float  |  None, float  |  None] | float | None, optional
        the corresponding angles for image correction (if any), by default `None`
    height : int | float, optional
        the smallest height of peaks, by default `2000`
    display_plots : bool, optional
        if `True` functional plots are displayed, by default `False`

    Returns
    -------
    maxlag : float
        maximum lag
    """
    # collecting data
    if type(nights) == int:
        obs1 = nights
        obs2 = nights
    else:
        obs1, obs2 = nights
 
    obj1, obj2 = objs_name
    
    if isinstance(angles, (tuple,list)):
        angle1, angle2 = angles
    else:
        angle1 = angles 
        angle2 = angles 
    
    ## ?
    plots = False
    ## ?

    obj1, cut1, lamp1, lims1 = extract_data(obs1,obj1,sel=['obj','lamp'])
    obj2, cut2, lamp2, lims2 = extract_data(obs2,obj2,sel=['obj','lamp'])

    _, _, angle1 = get_data('',obj1,cut1,display_plots=plots)
    _, _, angle2 = get_data('',obj2,cut2,display_plots=plots)

    _, lamp1 = get_data_fit(lamp1,lims1,title='lamp1',display_plots=plots)
    _, lamp2 = get_data_fit(lamp2,lims2,title='lamp2',display_plots=plots)

    _, lamp1 = angle_correction(lamp1,angle=angle1,display_plots=plots)
    _, lamp2 = angle_correction(lamp2,angle=angle2,display_plots=plots)

    plt.show()

    # selecting the spectrum row
    sel_height1 = int(np.mean(np.argmax(lamp1,axis=0)))
    sel_height2 = int(np.mean(np.argmax(lamp2,axis=0)))

    ## ?
    print(f'Sel heightt 1: {sel_height1}')
    print(f'Sel heightt 2: {sel_height2}')
    ## ?

    # extracting spectra
    lamp1 = lamp1[sel_height1]
    lamp2 = lamp2[sel_height2]

    # storing maximum values
    maxlamp1 = lamp1.max()
    maxlamp2 = lamp2.max()

    # computing this ratio
    fact = min(maxlamp1,maxlamp2)/max(maxlamp1,maxlamp2)

    # condition to module the minimum height for peaks
    if maxlamp1 > maxlamp2:
        height1 = height
        height2 = height*fact
    elif maxlamp1 < maxlamp2:
        height1 = height*fact
        height2 = height
    else:
        height1, height2 = height, height

    ## ?
    print(height1,height2)
    ## ?

    # importing the function to detect peaks
    from scipy.signal import find_peaks
    # finding the positions of the peaks in lamp spectra
    pkslamp1, _ = find_peaks(lamp1,height=height1)
    pkslamp2, _ = find_peaks(lamp2,height=height2)
    # checking the type
    pkslamp1 = pkslamp1.astype(int)
    pkslamp2 = pkslamp2.astype(int)
    # approximating the trend
    mpks1, mline1 = mean_line(pkslamp1,lamp1)
    mpks2, mline2 = mean_line(pkslamp2,lamp2)
    
    ## ?
    print('0: ',len(mpks1),len(mpks2))
    cnt = 0
    ## ?

    # computing the difference of array lenghts    
    dim_diff = len(mpks1)-len(mpks2)
    # correction routine
    while(dim_diff != 0):
        
        ## ?
        cnt += 1
        ## ?

        # condition to increase the boundary limit to one or to the other
        if dim_diff > 0: height1 += 100
        else: height2 += 100

        # same procedure
        pkslamp1, _ = find_peaks(lamp1,height=height1)
        pkslamp2, _ = find_peaks(lamp2,height=height2)

        pkslamp1 = pkslamp1.astype(int)
        pkslamp2 = pkslamp2.astype(int)

        mpks1, mline1 = mean_line(pkslamp1,lamp1)
        mpks2, mline2 = mean_line(pkslamp2,lamp2)
        
        ## ?
        print(f'{cnt}: ',len(mpks1),len(mpks2))
        ## ?
        
        # computing the difference again
        dim_diff = len(mpks1)-len(mpks2)

    # plots
    plt.figure()
    plt.subplot(2,1,1)
    plt.title('Spectrum of the lamps')
    plt.plot(lamp1,'b',label='lamp1')
    plt.plot(pkslamp1,lamp1[pkslamp1],'.r',label='detected peaks')
    plt.plot(mpks1,mline1,'xg',label='after interpolation')
    plt.axhline(height1,xmin=0,xmax=1,linestyle='-.',color='black',alpha=0.5)
    plt.ylabel('$I_1$ [a.u.]')
    plt.legend()
    # plt.xticks(np.arange(0,len(lamp1),len(lamp1)//4),[])
    plt.subplot(2,1,2)
    plt.plot(lamp2,'c',label='lamp2')
    plt.plot(pkslamp2,lamp2[pkslamp2],'.r',label='detected peaks')
    plt.plot(mpks2,mline2,'xg',label='after interpolation')
    plt.axhline(height2,xmin=0,xmax=1,linestyle='-.',color='black',alpha=0.5)
    plt.ylabel('$I_2$ [a.u.]')
    plt.legend()
    plt.xlabel('x [a.u.]')
    plt.show()

    ## ?
    print(len(pkslamp1),len(pkslamp2))
    print(f'Max diff: {abs(mpks1-mpks2).max()}')
    ## ?

    # computing the correlation and the autocorrelation for peaks positions
    corr = np.correlate(mpks1.astype(float),mpks2.astype(float),'full')
    autocorr = np.correlate(mpks2.astype(float),mpks2.astype(float),'full')
    # computing the max lag
    maxlag = np.abs(corr/max(corr)-autocorr/max(autocorr)).max()
    # printing the max lag
    print(f'max(|auto_corr - corr|) = {maxlag}')
    # condition for additional plots
    if display_plots:
        plt.figure('Correlation of Lamps',figsize=[10,7])
        plt.suptitle('Correlation between the peaks position of two lamps:\nmaxlag $\equiv \max{ | \\bar{C}(lamp_1,lamp_2) - \\bar{C}(lamp_2,lamp_2) | } =$' + f'{maxlag}')

        plt.subplot(2,2,1)
        #
        plt.title('Spectrum of the lamps')
        plt.plot(lamp1,'b',label='lamp1')
        plt.plot(pkslamp1,lamp1[pkslamp1],'.r',label='detected peaks')
        plt.plot(mpks1,mline1,'xg',label='after interpolation')
        plt.ylabel('$I_1$ [a.u.]')
        plt.legend()
        plt.xticks(np.arange(0,len(lamp1),len(lamp1)//4),[])
        #
        plt.subplot(2,2,3)
        plt.plot(lamp2,'c',label='lamp2')
        plt.plot(pkslamp2,lamp2[pkslamp2],'.r',label='detected peaks')
        plt.plot(mpks2,mline2,'xg',label='after interpolation')
        plt.ylabel('$I_2$ [a.u.]')
        plt.legend()
        plt.xlabel('x [a.u.]')


        plt.subplot(2,2,2)
        plt.title('Correlation and autocorrelation')
        plt.plot(corr/max(corr),'b',label='normalized correlation')
        plt.ylabel('$\\bar{C}(lamp_1,lamp_2)$')
        plt.grid(axis='x',linestyle='dashed',alpha=0.3)
        plt.legend()
        # plt.xticks(np.arange(0,len(corr),len(corr)//6),[])
        #
        plt.subplot(2,2,4)
        plt.plot(autocorr/max(autocorr),'c',label='normalized autocorrelation')
        plt.ylabel('$\\bar{C}(lamp_2,lamp_2)$')
        plt.grid(axis='x',linestyle='dashed',alpha=0.3)
        plt.legend()
        plt.xlabel('idx')
        plt.show()

    return maxlag

    




