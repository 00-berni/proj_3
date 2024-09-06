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
from .data import extract_data, extract_cal_data, get_cal_lines, get_standard, store_results
from .stuff import FuncFit, compute_err, mean_n_std

def compute_master_dark(mean_dark: Spectrum | None, master_bias: Spectrum | None = None, diagn_plots: bool = False, **figargs) -> Spectrum:
    """To compute master dark

    Parameters
    ----------
    mean_dark : Spectrum | None
        averaged dark data
    master_bias : Spectrum | None, optional
        averaged bias data, by default `None`
    diagn_plots : bool, optional
        to plot figures, by default `False`
    **figargs
        parameters for the images, see `display.showfits()`

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
    if diagn_plots:
        if master_dark.sigma is not None:
            plt.figure()
            plt.title('Sigma Dark')
            plt.imshow(master_dark.sigma)
            plt.colorbar()
        show_fits(master_dark,show=True,**figargs)
    return master_dark

def compute_master_flat(flat: Spectrum, master_dark: Spectrum | None = None, master_bias: Spectrum | None = None, diagn_plots: bool = True, **figargs) -> Spectrum:
    """To estimate the flat gain

    Parameters
    ----------
    flat : Spectrum
        flat data
    master_dark : Spectrum | None, optional
        master dark if any, by default `None`
    master_bias : Spectrum | None, optional
        master bias if any, by default `None`
    diagn_plots : bool, optional
        to plot figures, by default `False`
    **figargs
        parameters for the images, see `display.showfits()`

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
    if diagn_plots:
        _ = show_fits(master_flat,show=True,**figargs)
    return master_flat

def get_target_data(ch_obs: str, ch_obj: str, selection: int | Literal['mean'], cut: bool = True, angle: float | None = 0, display_plots: bool = True, diagn_plots: bool = False,**figargs) -> tuple[Spectrum, Spectrum]:
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
        to display images and plots, by default `True`
    diagn_plots : bool, optional
        to plot figures, by default `False`
    **figargs
        parameters for the images, see `display.showfits()`

    Returns
    -------
    target : Spectrum 
        science frame of the target
    lamp : Spectrum
        science frame of its calibration lamp
    """
    ## Data extraction
    # get the light frames
    target, lamp = extract_data(ch_obs,ch_obj,selection,diagn_plots=diagn_plots,**figargs)
    if diagn_plots: 
        show_fits(target, title='Light Frame',**figargs)
        if lamp.name != 'empty':
            show_fits(lamp, title='Light Frame',**figargs)
        plt.show()
    
    ## Calibration correction
    calibration = extract_cal_data(ch_obs)
    if len(calibration) > 1:
        if len(calibration) == 3:   #: in this case `calibration = [flat, dark, bias]` 
            # compute master dark
            calibration[1] = compute_master_dark(*calibration[1:], diagn_plots=diagn_plots,**figargs)
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
    master_flat = compute_master_flat(*calibration,diagn_plots=diagn_plots,**figargs)
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
        target, angle = target.angle_correction(angle=angle, diagn_plots=diagn_plots)      
        if np.all(target.lims == [0, None, 0, None]): 
            exit_cond = True
            norm = 'log'
        target.cut_image()
        if lamp.name != 'empty':
            lamp, _ = lamp.angle_correction(angle=angle, diagn_plots=diagn_plots)      
            lamp.cut_image()
    if display_plots:
        show_fits(target, title='Science Frame',norm=norm,**figargs)
        if diagn_plots and target.sigma is not None: 
            plt.figure()
            plt.title('sigma targ')
            plt.imshow(target.sigma)
            plt.colorbar()
        if lamp.name != 'empty':
            show_fits(lamp, title='Science Frame',norm=norm,**figargs)
            if diagn_plots and lamp.sigma is not None: 
                plt.figure()
                plt.title('sigma lamp')
                plt.imshow(lamp.sigma)
                plt.colorbar()
        plt.show()
    if exit_cond: exit()
    return target, lamp

def lines_calibration(ch_obs: str, ch_obj: str, trsl: int, ord: int = 2, initial_values: Sequence[float] | None = None, display_plots: bool = True) -> tuple[Callable[[ndarray], ndarray], Callable[[ndarray],ndarray]]:
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
    display_plots : bool, optional
        to display images and plots, by default `True`

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
    if display_plots:
        plt.figure()
        pp = np.linspace(pxs.min(),pxs.max(),200)
        plt.errorbar(pxs,lines,Dlines,errs,'.',color='orange')
        plt.plot(pp,px_to_arm(pp))
        plt.figure()
        plt.errorbar(pxs,lines-fit.res['func'](pxs,*pop),Dlines,errs,'.',color='orange')
        plt.axhline(0,0,1)
        plt.show()
    return px_to_arm, err_func

def lamp_correlation(lamp1: Spectrum, lamp2: Spectrum, display_plots: bool = True, **pltargs) -> float:
    """To compute the lag to pass from a lamp to another

    Parameters
    ----------
    lamp1 : Spectrum
        first lamp
    lamp2 : Spectrum
        second lamp
    display_plots : bool, optional
        to display images and plots, by default `True`
    **pltargs
        parameters for the plots, see `display.quickplot()`

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
    if display_plots:
        quickplot(corr,title=f'Cross corr -> {np.argmax(corr)/len(corr)*100} : {len(corr) // np.argmax(corr)}',**pltargs)
        plt.show()
    # compute the lag
    shift = np.argmax(corr) - (len(corr)/2)  
    return shift

def calibration(ch_obs: str, ch_obj: str, selection: int | Literal['mean'], angle: float | None = None, height: int | None = None, other_lamp: Spectrum | None = None, ord: int = 2, initial_values: Sequence[float] | None = None, save_data: bool = True, txtkw: dict = {}, display_plots: bool = True, diagn_plots: bool = False, figargs: dict = {}, pltargs: dict = {}) -> tuple[Spectrum, Spectrum]:
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
        to display images and plots, by default `True`
    diagn_plots : bool, optional
        if it is `True` diagnostic images/plots are displayed, by default `False`

    Returns
    -------
    target : Spectrum 
        calibrated science frame of the target
    lamp : Spectrum
        calibrated science frame of its lamp
    """    
    ## Data
    # extract data
    target, lamp = get_target_data(ch_obs, ch_obj, selection, angle=angle, display_plots=display_plots, diagn_plots=diagn_plots,**figargs)
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
    if diagn_plots:
        quickplot(target.spec,title='Uncalibrated spectrum of '+target.name,labels=('x [a.u.]','I [a.u.]'),numfig=1,**pltargs)
        quickplot(lamp.spec,title='Uncalibrated spectrum of its lamp',labels=('x [a.u.]','I [a.u.]'),numfig=2,**pltargs)
        plt.show()
    
    ## Calibration
    if other_lamp is None:      
        # compute the calibration function via fit
        cal_func, err_func = lines_calibration(ch_obs, ch_obj, trsl=lamp.lims[2], ord=ord, display_plots=display_plots)
        # store results
        lamp.func = [cal_func, err_func]
        target.func = [*lamp.func]
        # compute the values in armstrong
        lamp.compute_lines()
        target.compute_lines()
    else:
        # compute the lag between the two lamps
        shift = lamp_correlation(lamp, other_lamp, display_plots=display_plots, **pltargs)
        # store the functions
        lamp.func = [*other_lamp.func]
        target.func = [*other_lamp.func]
        # compute the values in armstrong
        lamp.compute_lines(shift=shift)
        target.compute_lines(shift=shift)
    
    ## Plot
    if display_plots:
        quickplot((lamp.lines,lamp.spec),labels=('$\\lambda$ [$\\AA$]','I [a.u.]'),numfig=3,**pltargs)
        if lamp.sigma is not None:
            plt.errorbar(lamp.lines, lamp.spec,yerr=lamp.std,xerr=lamp.errs,fmt='.')
        else:
            plt.errorbar(lamp.lines, lamp.spec,xerr=lamp.errs,fmt='.')
        quickplot((target.lines,target.spec),labels=('$\\lambda$ [$\\AA$]','I [a.u.]'),numfig=4,**pltargs)
        plt.errorbar(target.lines,target.spec,target.std,target.errs,fmt='.')
        plt.yscale('log')
        plt.show()        
    
    ## Store
    if save_data:
        header = 'lambda [A]\tDlambda [A]\tspecval [counts]\tDspecval [counts]'
        # target
        file_name = ch_obj.lower() + '_' + str(selection)
        data = [target.lines, target.errs, target.spec, target.std]
        store_results(file_name, data, ch_obs, ch_obj, header=header, **txtkw)
        # lamp
        file_name = 'lamp-' + ch_obj.lower()
        data = [lamp.lines, lamp.errs, lamp.spec, lamp.std]
        store_results(file_name, data, ch_obs, ch_obj, header=header, **txtkw)
    return target, lamp


def atm_transfer(airmass: tuple[ndarray, ndarray], wlen: tuple[ndarray, ndarray], data: tuple[ndarray, ndarray], bins: ndarray, display_plots: bool = True, diagn_plot: bool = False) -> tuple[tuple[ndarray,ndarray], tuple[ndarray, ndarray]]:
    """To compute the optical depth and the 0 airmass spectrum

    Parameters
    ----------
    airmass : tuple[ndarray, ndarray]
        values of airmass and the corresponding uncertainties
    wlen : tuple[ndarray, ndarray]
        values of wavelengths and the corresponding uncertainties
    data : tuple[ndarray, ndarray]
        spectrum and the corresponding uncertainties
    bins : ndarray
        wavelengths bin values
    display_plots : bool, optional
        to display images and plots, by default `True`
    diagn_plot : bool, optional
        to display diagnostic images and plots, by default `False`

    Returns
    -------
    (I0, DI0) : tuple[ndarray, ndarray]
        0 airmass spectrum and uncertainty
    (tau, Dtau) : tuple[ndarray,ndarray]
        optical depth and uncertainty
    """
    # load data
    x, Dx = airmass
    l_data, Dl_data = wlen
    y_data, Dy_data = data
    bins = np.copy(bins)
    # prepare arrays to collect values of I0 and tau
    a_I0   = np.empty((0,2))
    a_tau  = np.empty((0,2))
    if display_plots:
        plt.figure()
        plt.title('Binned data for different airmass')
        for i in range(l_data.shape[0]):
            plt.errorbar(l_data[i],y_data[i],Dy_data[i],Dl_data[i], label=f'$X = ${x[i]:.2}')
            plt.xticks(bins[i],bins[i],rotation=45)
        plt.xlabel('$\\lambda$ [$\\AA$]')
        plt.ylabel('Norm. Data [counts/s]')
        plt.grid(True,which='both',axis='x')
        plt.legend()
        plt.show()
    if diagn_plot:    
        fig1, ax1 = plt.subplots(1,1)
        fig2, ax2 = plt.subplots(1,1)
        ax2.axhline(0, 0, 1, color='black')
    for i in range(l_data.shape[1]):
        # select data
        y = y_data[:,i]
        Dy = Dy_data[:,i]
        # fit routine
        #.. Assuming N/t = exp(-tau*a)*I0 then 
        #.. log(N/t) = - tau * a + log(I0)
        initial_values = [-0.5, np.log(y).max()]
        fit = FuncFit(x, np.log(y), xerr=Dx, yerr=Dy/y)
        fit.linear_fit(initial_values, names=('tau','ln(I0)'))
        pop, Dpop = fit.results()
        I0  = np.exp(pop[1])
        DI0 = Dpop[1] * I0
        # store the results
        a_I0  = np.append(a_I0,  [ [I0, DI0] ], axis=0)
        a_tau = np.append(a_tau, [ [-pop[0], Dpop[0]] ], axis=0)
        # plot them
        if diagn_plot:
            func = fit.res['func']
            xx = np.linspace(x.min(),x.max(),50)
            color = (0.5,i/l_data.shape[1],1-i/l_data.shape[1])
            ax1.errorbar(x, np.log(y), xerr=Dx, yerr=Dy/y, fmt='.', color=color)
            ax1.plot(xx, func(xx,*pop), color=color)
            ax2.errorbar(x, np.log(y) - func(x,*pop), xerr=Dx, yerr=Dy/y, fmt='.', color=color)
    print('DIFF',np.diff(bins,axis=0), np.diff(Dl_data,axis=0))
    # select a row
    l_data, Dl_data = l_data[0], Dl_data[0] 
    bins = bins[0]
    # collect data
    I0, DI0 = a_I0[:,0], a_I0[:,1]
    tau, Dtau = a_tau[:,0], a_tau[:,1]
    # plot
    if display_plots:
        plt.figure()
        plt.title('Estimated Optical Depth')
        plt.errorbar(l_data,tau,Dtau,Dl_data,'.',linestyle='dashed')
        plt.xlabel('$\\lambda$ [$\\AA$]')
        plt.ylabel('$\\tau$')
        plt.show()

        plt.figure()
        plt.title('Estimated Data at 0 airmass')
        plt.errorbar(l_data, I0, DI0, Dl_data)
        plt.grid(True,which='both',axis='x')
        plt.xticks(bins,bins,rotation=45)
        plt.xlabel('$\\lambda$ [$\\AA$]')
        plt.ylabel('$I_0$ [counts/s]')
        plt.show()
    return (I0, DI0), (tau, Dtau)

def ccd_response(altitude: tuple[ndarray, ndarray], tg_obs: list[Spectrum], ends_wlen: list[list[float]],  bin_width: float | int = 50, std_name: str = 'Vega', selection: int = 0, display_plots: bool = True, diagn_plots: bool = False) -> tuple[tuple[ndarray,ndarray],tuple[ndarray,ndarray], tuple[ndarray, ndarray]]:
    """To estimate instrument response function

    Parameters
    ----------
    altitude : tuple[ndarray, ndarray]
        different altitudes values and the corresponding uncertainties
    tg_obs : list[Spectrum]
        spectrum data of target for different altitudes
    ends_wlen : list[list[float]]
        list of extremes of each wavelengths range acquired at different altitudes
    bin_width : float | int, optional
        the width of each bin, by default `50`
    std_name : str, optional
        the name of the standard used to calibrate, by default `'Vega'`
    selection : int, optional
        the kind of chosen standard data, by default `0`
    display_plots : bool, optional
        to display images and plots, by default `True`
    diagn_plot : bool, optional
        to display diagnostic images and plots, by default `False`

    Returns
    -------
    (l_data, Dl_data) : tuple[ndarray, ndarray]
        _description_
    (R, DR) : tuple[ndarray,ndarray]
        _description_
    (op_dep, Dop_dep) : tuple[ndarray,ndarray]
        _description_
    """
    ## Data Collection
    alt, Dalt = altitude
    # airmass
    x  = 1/np.sin(alt*np.pi/180)
    Dx = Dalt * np.cos(alt*np.pi/180) * x**2 * np.pi/180 
    # ends of the wavelengths range
    min_line = np.max(ends_wlen,axis=0)[0]
    max_line = np.min(ends_wlen,axis=0)[1]
    # define variables to collect values
    l_data = []     #: central values of binned wavelengths
    y_data = []     #: binned spectrum data
    a_bin  = []     #: bins values
    for obs in tg_obs:
        # normalize spectrum data by exposure time
        obs.spec = obs.spec / obs.get_exposure()
        # bin the data
        l, y, bins = obs.binning(bin=bin_width,edges=(min_line,max_line))    
        if diagn_plots:
            plt.figure()
            plt.title('Normalized Binned Data')
            plt.plot(l[0],y[0],'.-')
            plt.grid(True,which='both',axis='x')
            plt.xticks(bins,bins,rotation=45)
            plt.xlabel('$\\lambda$ [$\\AA$]')
            plt.ylabel('$N/t_{exp}$ [counts/s]')
            plt.show()
        # store the results     
        l_data +=  [[*l]]
        y_data +=  [[*y]]
        a_bin  +=  [bins]
    # from list to array
    l_data, Dl_data = np.array(l_data).transpose((1,0,2))
    y_data, Dy_data = np.array(y_data).transpose((1,0,2))
    a_bin = np.array(a_bin)
    
    ## Atmospheric Transfer Function
    # estimate 0 airmass spectrum
    (I0, DI0), (op_dep, Dop_dep) = atm_transfer((x,Dx), (l_data,Dl_data), (y_data,Dy_data), a_bin, display_plots=display_plots, diagn_plot=diagn_plots)

    ## Response Function
    # select a row
    l_data, Dl_data = l_data[0], Dl_data[0] 
    bins = a_bin[0]
    # set ends of wavelengths range
    min_line, max_line = bins[0], bins[-1]
    # get the data of the standard
    std_wlen, std_data = get_standard(name=std_name, sel=selection, diagn_plots=diagn_plots)
    std = Spectrum.empty()      #: variable to collect standard spectrum data
    # store the ends of wavelengths range of the standard 
    start, end = std_wlen[0], std_wlen[-1]
    # check the length
    if start > bins[1]: 
        pos = np.argmin(np.abs(bins-std_wlen[0]))
        min_line = bins[pos]
        bins = bins[pos:]
        l_data = l_data[pos:]
        print('Less')
        print('\t',len(I0))
        I0 = I0[pos:]
        DI0 = DI0[pos:]
        print('\t',min_line,len(I0))
    if end < bins[-2]: 
        pos = np.argmin(np.abs(bins-std_wlen[-1]))
        max_line = bins[pos]
        bins = bins[:pos+1]
        l_data = l_data[:pos]
        print('More')
        print('\t',len(I0))
        I0 = I0[:pos]
        DI0 = DI0[:pos]
        print('\t',max_line,len(I0))
    # store data
    std.lines = std_wlen
    std.spec  = std_data
    if display_plots:
        plt.figure()
        plt.suptitle('Spectra of Standard and Target after resizing')
        plt.subplot(2,1,1)
        plt.title('Standard')
        plt.plot(std.lines,std.spec,'.-')
        plt.ylabel('$S_{std}$ [erg/(s cm$^2$ $\\AA$)]')
        plt.subplot(2,1,1)
        plt.title('Target')
        plt.plot(l_data,I0,'.-')
        plt.xlabel('$\\lambda$ [$\\AA$]')
        plt.ylabel('$I_0$ [counts/s]')
    # bin data
    bstd_wlen, bstd_s, _ = std.binning(bin=bins)
    print('SEE',len(I0),len(bstd_wlen[0]),len(l_data))

    if diagn_plots:
        plt.figure()
        plt.title('Binned Standard Data')
        plt.errorbar(bstd_wlen[0],bstd_s[0],bstd_s[1],bstd_wlen[1],'.',linestyle='dashed')
        plt.grid(True,which='both',axis='x')
        plt.xticks(bins,bins,rotation=45)
        plt.ylabel('$S_{std}$ [erg/(s cm$^2$ $\\AA$)]')
        plt.xlabel('$\\lambda$ [$\\AA$]')

    if display_plots:
        plt.figure()
        plt.title('Response Function')
        plt.plot(l_data,I0/bstd_s[0],'.-')
        plt.grid(True,which='both',axis='x')
        plt.xticks(bins,bins,rotation=45)
        plt.xlabel('$\\lambda$ [$\\AA$]')
        plt.ylabel('$R$ [counts cm$^2$ $\\AA$ / ergs]')
    
    plt.show()

    stand, Dstand = bstd_s
    R = I0/stand
    DR = R * np.sqrt((DI0/I0)**2 + (Dstand/stand)**2)
    return (l_data, Dl_data), (R, DR), (op_dep, Dop_dep)


    




