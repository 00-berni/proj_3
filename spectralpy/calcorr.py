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
from .stuff import FuncFit, compute_err, mean_n_std

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
        master_dark.data  = mean_dark - master_bias
        master_dark.sigma = compute_err(mean_dark, master_bias)
    # condition to display the images/plots
    if display_plots:
        # if master_dark.sigma is not None:
        #     plt.figure()
        #     plt.title('Sigma Dark')
        #     plt.imshow(master_dark.sigma)
        #     plt.colorbar()
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
    if master_bias is not None: 
        flat.data  = flat - master_bias
        flat.sigma =  compute_err(flat, master_bias)
    # correct by dark, if any
    if master_dark is not None: 
        # check the exposure times
        fl_exp = flat.get_exposure()
        dk_exp = master_dark.get_exposure()
        if fl_exp != dk_exp:
            master_dark.data  = master_dark.data / dk_exp * fl_exp
            master_dark.sigma = master_dark.sigma / dk_exp * fl_exp
        flat.data  = flat - master_dark 
        flat.sigma = compute_err(flat, master_dark)

    master_flat = flat.copy()
    master_flat.data = flat.data / np.mean(flat.data)
    if flat.sigma is not None: 
        master_flat.sigma /= np.mean(flat.data)
        # plt.figure()
        # plt.imshow(master_flat.sigma)
        # plt.colorbar()
    master_flat.name = 'Master Flat'
    # condition to display the images/plots
    if display_plots:
        _ = show_fits(master_flat,show=True)
    return master_flat

def get_target_data(ch_obs: str, ch_obj: str, selection: int | Literal['mean'], cut: bool = True, angle: float | None = 0, spec_plot: bool = True, display_plots: bool = False,**kwargs) -> tuple[Spectrum, Spectrum]:
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
    if display_plots: 
        show_fits(target, title='Light Frame',**kwargs)
        if lamp.name != 'empty':
            show_fits(lamp, title='Light Frame',**kwargs)
        plt.show()
    
    ## Calibration correction
    calibration = extract_cal_data(ch_obs)
    if len(calibration) > 1:
        if len(calibration) == 3:   #: in this case `calibration = [flat, dark, bias]` 
            # compute master dark
            calibration[1] = compute_master_dark(*calibration[1:],display_plots=display_plots,**kwargs)
            # bias correction
            bias = calibration[2]
            target.data = target - bias
            target.sigma = compute_err(target, bias)
            if lamp.name != 'empty':
                lamp.data = lamp - bias
                lamp.sigma = compute_err(lamp, bias)
        # check the exposure times
        tg_exp = target.get_exposure()
        dk_exp = calibration[1].get_exposure()
        if tg_exp != dk_exp:
            calibration[1].data = calibration[1].data  / dk_exp * tg_exp
            calibration[1].sigma = calibration[1].sigma / dk_exp * tg_exp
        # dark correction
        dark = calibration[1].copy()
        target.data = target - dark
        target.sigma = compute_err(target, dark)
        if lamp.name != 'empty':
            lamp.data = lamp - dark
            lamp.sigma = compute_err(lamp, dark)
    # compute master flat
    master_flat = compute_master_flat(*calibration,display_plots=display_plots,**kwargs)
    print('MIN',master_flat.data.min())
    flat_err = lambda data : 0 if master_flat.sigma is None else (data*master_flat.sigma / master_flat.data**2)**2
    # flat correction
    err = lambda sigma : 0 if sigma is None else (sigma / master_flat.data)**2
    target_err = flat_err(target.data) + err(target.sigma)
    target.data = target.data / master_flat.data
    target.sigma = None if isinstance(target_err, int) else np.sqrt(target_err)

    if lamp.name != 'empty':
        lamp_err = flat_err(lamp.data) + err(lamp.sigma)
        lamp.data = lamp.data / master_flat.data  
        lamp.sigma = None if isinstance(lamp_err, int) else np.sqrt(lamp_err)
        #? CHECK
        pos = np.where(lamp.sigma < 0)
        if len(pos[0]) != 0:
            plt.figure()
            plt.title('HELP')
            plt.imshow(lamp.sigma,cmap='gray')
            plt.plot(pos[1],pos[0],'.',color='red')
            plt.show()
            raise
    exit_cond = False
    norm = 'linear'
    if cut:    
        target, angle = target.angle_correction(angle=angle)      
        if np.all(target.lims == [0, None, 0, None]): 
            exit_cond = True
            norm = 'log'
        target.cut_image()
        if lamp.name != 'empty':
            lamp, _ = lamp.angle_correction(angle=angle)      
            lamp.cut_image()
    if spec_plot:
        show_fits(target, title='Science Frame',norm=norm,**kwargs)
        # if target.sigma is not None: 
        #     plt.figure()
        #     plt.title('sigma targ')
        #     plt.imshow(target.sigma)
        #     plt.colorbar()
        if lamp.name != 'empty':
            show_fits(lamp, title='Science Frame',norm=norm,**kwargs)
            # if lamp.sigma is not None: 
            #     plt.figure()
            #     plt.title('sigma lamp')
            #     plt.imshow(lamp.sigma)
            #     plt.colorbar()
        plt.show()
    if exit_cond: exit()
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
    print('LAMP2',lamp2)
    edge = edge1 - edge2
    print('EDGE',edge1)
    print('EDGE',edge2)
    print('EDGE',edge)
    edge1 = [0 if edge[0] > 0 else -edge[0], -edge[1] if edge[1] > 0 else None]
    edge2 = [edge[0] if edge[0] > 0 else 0 , None if edge[1] > 0 else edge[1]]
    print('EDGEs',edge1, edge2)
    lamp1 = lamp1.spec[slice(*edge1)].copy()
    lamp2 = lamp2.spec[slice(*edge2)].copy()

    ## Correlation procedure
    print('LAMP2',lamp2)
    # normalize the data
    lamp1 = lamp1/lamp1.max()
    lamp2 = lamp2/lamp2.max()
    # compute cross-correlation
    from scipy.signal import correlate
    corr = correlate(lamp1,lamp2)
    print('MAX POS: ',np.argmax(corr)/len(corr)*100, len(corr) // np.argmax(corr))
    # quickplot(corr,title=f'Cross corr -> {np.argmax(corr)/len(corr)*100} : {len(corr) // np.argmax(corr)}')
    # plt.show()
    # compute the lag
    shift = np.argmax(corr) - (len(corr)/2)  
    return shift

def calibration(ch_obs: str, ch_obj: str, selection: int | Literal['mean'], angle: float | None = None, height: int | None = None, other_lamp: Spectrum | None = None, ord: int = 2, initial_values: Sequence[float] | None = None, spec_plot: bool = False, display_plots: bool = False, **kwargs) -> tuple[Spectrum, Spectrum]:
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
    target, lamp = get_target_data(ch_obs, ch_obj, selection, angle=angle, spec_plot=spec_plot, display_plots=display_plots,**kwargs)
    # average along the y axis
    data = target.data.mean(axis=1)
    data = np.array([ target.data[i,:] / data[i] for i in range(len(data)) ])
    # plt.figure(); plt.imshow(data); plt.colorbar(); plt.show()
    target.spec, target.std = mean_n_std(data, axis=0)
    # compute the height for the lamp
    if height is None: height = int(len(lamp.data)/2)
    # take lamp spectrum at `height` 
    lamp.spec = lamp.data[height]
    if lamp.sigma is not None: 
        lamp.std = lamp.sigma[height]
        pos = np.where(lamp.std < 0)[0]
        if len(pos) != 0:
            plt.figure()
            plt.plot(lamp.std)
            plt.plot(pos,lamp.std[pos],'.')
            plt.show()
            raise
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
    else:
        # compute the lag between the two lamps
        shift = lamp_correlation(lamp, other_lamp)
        # store the functions
        lamp.func = [*other_lamp.func]
        target.func = [*other_lamp.func]
        # compute the values in armstrong
        lamp.compute_lines(shift=shift)
        target.compute_lines(shift=shift)
    
    ## Plot
    if spec_plot:
        quickplot((lamp.lines,lamp.spec),labels=('$\\lambda$ [$\\AA$]','I [a.u.]'),numfig=3,**kwargs)
        if lamp.sigma is not None:
            plt.errorbar(lamp.lines, lamp.spec,yerr=lamp.std,xerr=lamp.errs,fmt='.')
        else:
            plt.errorbar(lamp.lines, lamp.spec,xerr=lamp.errs,fmt='.')
        quickplot((target.lines,target.spec),labels=('$\\lambda$ [$\\AA$]','I [a.u.]'),numfig=4,**kwargs)
        plt.errorbar(target.lines,target.spec,target.std,target.errs,fmt='.')
        plt.yscale('log')
        plt.show()        
    return target, lamp






    




