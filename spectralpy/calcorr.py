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
from numpy.typing import NDArray
from typing import Callable, Literal
from scipy import odr
from .display import *
from .data import get_data_fit, extract_data, extract_cal_data

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


def calibration(ch_obs: str, ch_obj: str, selection: int | Literal['mean'], angle: float | None = None, height: int | None = None, initial_values: list[float] = [3600, 2.6, 0.], display_plots: bool = False, **kwargs) -> tuple[Callable[[NDArray],NDArray], Callable[[NDArray],NDArray]]:
    """Evaluating the calibration function to pass from a.u. to Armstrong

    Parameters
    ----------
    cal_file : str
        file with calibration lines sampled from the `SpecrteArNeLISA.pdf` file
    obj_lamp : str
        path for the lamp file
    lims_lamp : list
        extrema of the lamp image
    angle : float
        inclination angle
    initial_values : list[float], optional
        initial values for the fit, by default `[3600, 2.6, 0.]`
    display_plots : bool, optional
        if it is `True` images/plots are displayed, by default `False`

    Returns
    -------
    cal_func : Callable[[NDArray],NDArray]
        the calibration function
    err_func : Callable[[NDArray],NDArray]
        _description_
    
    Notes
    -----
    From spectrum data of the selected calibration lamp and corrisponding detected lines 
    (obtained from `SpecrteArNeLISA.pdf` file) the function to map positions along the x 
    axis in wavelengths is computed. To estimate the parameters a fit for a linear 
    function is implemented

    """
    target, lamp = get_target_data(ch_obs, ch_obj, selection, angle=angle, display_plots=display_plots)
    target.spec = np.mean(target.data, axis=0)
    if height is None: height = int(len(lamp.data)/2) 
    lamp.spec = lamp.data[height]
    if display_plots:
        quickplot(target.spec,title='Uncalibrated spectrum of '+target.name,labels=('x [a.u.]','y [a.u.]'),numfig=1)
        quickplot(lamp.spec,title='Uncalibrated spectrum of its lamp',labels=('x [a.u.]','y [a.u.]'),numfig=2)
        plt.show()

    # # fit method
    # # defining the linear function for the fit
    # def fit_func(param,x):
    #     p0,p1,p2 = param
    #     return p0 + p1*x + p2*x**2
    # # extracting the data for the calibration from `cal_file`
    # lines, x, Dx = np.loadtxt(cal_file, unpack=True)
    # Dy = np.full(lines.shape,3.63)

    # pop, perr, pcov = fit_routine(x,lines,initial_values=initial_values,fit_func=fit_func,xerr=Dx,yerr=Dy,display_res=display_plots,return_res=['pcov'])
    # _,p1,p2 = pop
    # Dp0,Dp1,Dp2 = perr

    # # defining the calibration function
    # cal_func = lambda x : fit_func(pop,x)
    
    # def err_func(x: NDArray, dx: NDArray, all_res: bool = False):
    #     dfdx = p1 + p2*2*x
    #     err = (dfdx*dx)**2
    #     if all_res:
    #         dfdp1 = x
    #         dfdp2 = x**2
    #         err += Dp0**2 + (dfdp1*Dp1)**2 + (dfdp2*Dp2)**2 + 2*( dfdp1*pcov[0,1] + dfdp2*pcov[0,2] + dfdp1*dfdp2*pcov[1,2]) 
    #     return np.sqrt(err)
        
    # # condition to display the images/plots
    # if display_plots == True:
    #     sigma = np.sqrt(Dy**2 + (p1*Dx + p2*x*Dx*2)**2)
    #     fig = plt.figure('Calibration',figsize=[8,7])
    #     fig.suptitle('Fit for lamp calibration')
    #     ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
    #     # plt.title('Calibration fit')
    #     ax1.errorbar(x,lines,xerr=Dx,yerr=Dy,fmt='.',color='green',label='data')
    #     ax1.plot(x,cal_func(x),color='orange',label='Best-fit')
    #     ax1.set_ylabel('$\lambda$ [$\AA$]')
    #     ax1.legend(numpoints=1)

    #     # plt.figure(figsize=[10,7])
    #     # plt.title('Residuals')
    #     ax2.axhline(0,xmin=0,xmax=1,linestyle='-.',color='black',alpha=0.5)
    #     ax2.errorbar(x,lines-cal_func(x),xerr=Dx,yerr=sigma,fmt='v',linestyle=':',color='green')
    #     ax2.set_xlabel('x [px]')
    #     ax2.set_ylabel('Residuals [$\AA$]')

    #     plt.show()

    # return cal_func, err_func




def calibrated_spectrum(ch_obs: int, ch_obj: str, flat: None | NDArray = None, cal_func: Callable[[NDArray],NDArray] | None = None, err_func: Callable[[NDArray,NDArray,bool],NDArray] | None = None, display_plots: bool = False, initial_values: list[float] | tuple[float] = [3600, 2.6,0.], ret_values: str = 'few') -> list[NDArray] | list[NDArray | dict]:
    """Getting the spectrum of a selceted target for a chosen observation night

    Parameters
    ----------
    ch_obs : int
        chosen observation night
    ch_obj : str
        chosen obj
    flat : None | NDArray, optional
        if the flat gain is not evaluated yet, the flat target name is passed, by default `None`
    cal_func : Callable[[NDArray],NDArray] | None, optional
        if it is None, the calibration function will be computed, by default `None`
    err_func : Callable[[NDArray,NDArray,bool],NDArray] | None, optional
        _description_, by default `None`
    display_plots : bool, optional
        if it is True images/plots are displayed, by default `False`
    initial_values : list[float] | tuple[float], optional
        _description_, by default `[3600, 2.6,0.]`
    ret_values : str, optional
        _description_, by default `'few'`

    Returns
    -------
    spectrum : NDArray
        cumulative spectrum 
    lengths : NDArray
        corrisponding wavelenghts
    data : dict[str,NDArray], optional
        information about the image
            * `'hdul'` : fits information
            * `'sp_data'` : spectrum image data
        It is returned only if `ret_values == 'data' or 'all'` 
    cal_data : dict[str, float | NDArray | Callable], optional
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


def mean_line(peaks: NDArray, spectrum: NDArray, dist: int = 3, height: int | float = 800) -> tuple[NDArray,NDArray]:
    """    

    Parameters
    ----------
    peaks : NDArray
        peaks
    spectrum : NDArray
        spectrum
    dist : int, optional
        maximum distance between peaks, by default `3`
    height : int | float, optional
        the maximum value to start interpolation, by default `800`

    Returns
    -------
    peaks : NDArray
        average peaks
    spectrum[peaks] : NDArray
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

    




