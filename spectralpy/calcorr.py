"""
    # Calibration module
"""
import os
import numpy as np
from typing import Callable
from scipy import odr
from .display import *
from .stuff import angle_correction, fit_routine
from .data import get_data_fit, extract_data, get_data



def compute_flat(ch_obs: int, display_plots: bool = False) -> np.ndarray:
    """Evaluating the flat gain
    After extracting the data of the flat acquisition, the function finds the
    maximum and computes the cumulative spectrum over the y axis. Then the 
    gain is extimated for each x coordinate by normalization through the 
    maximum and returned.

    :param ch_obs: chosen observation night
    :type ch_obs: int
    :param display_plots: if it is True images/plots are displayed, defaults to False
    :type display_plots: bool, optional
    
    :return: flat gain for each x coordinate
    :rtype: np.ndarray
    """
    # name of the target for which flat was acquired
    if ch_obs == 0:
        flat_obj = 'giove'  
    elif ch_obs == 1:
        flat_obj = 'arturo'
    # collecting informations for the target and the flat
    obj_fit, lims_fit, obj_flat, lims_flat = extract_data(ch_obs, flat_obj,sel=['obj','flat'])
    # extracting target spectrum data and evaluating the inclination angle
    _, _, angle = get_data(flat_obj,obj_fit,lims_fit, display_plots=display_plots)
    # extracting flat spectrum data and correcting for inclination
    _, sp_flat, _ = get_data(flat_obj,obj_flat,lims_flat, angle=angle, display_plots=display_plots)
    
    # flat gain estimation
    # finding the x coordinate for the maximum
    _, x_max = np.unravel_index(np.argmax(sp_flat), sp_flat.shape)
    # computing the cumulative value over the column
    tot_flat = sp_flat[:,x_max].sum()
    # computing the normalized flat depending on position x
    flat_value = sp_flat.sum(axis=0)/tot_flat
    # condition to display the images/plots
    if display_plots == True:
        fastplot(np.arange(len(flat_value)), flat_value, title='Flat spectrum', labels=['x','norm counts'])
        plt.show()

    return flat_value


def calibration(cal_file: str, obj_lamp: str, lims_lamp: list, angle: float, initial_values: list[float] = [3600, 2.6,0.], display_plots: bool = False) -> Callable[[np.ndarray],np.ndarray]:
    """Evaluating the calibration function to pass from x axis units to Armstrong
    From spectrum data of the selected calibration lamp and corrisponding detected lines (obtained from `SpecrteArNeLISA.pdf` file) the function to map positions
    along the x axis in wavelengths is computed. To estimate the parameters a fit for a linear function is implemented.

    :param cal_file: file with calibration lines sampled from the `SpecrteArNeLISA.pdf` file
    :type cal_file: str
    :param obj_lamp: path for the lamp file
    :type obj_lamp: str
    :param lims_lamp: extrema of the lamp image
    :type lims_lamp: list
    :param angle: inclination angle
    :type angle: float
    :param height: y coorfinate at which the spectrum is taken
    :type height: int
    :param initial_values: initial values for the fit, defaults to [3600, 2.6]
    :type initial_values: list[float], optional
    :param display_plots: if it is True images/plots are displayed, defaults to False
    :type display_plots: bool, optional
    
    :return: the calibration function
    :rtype: FUNC_TYPE
    """
    # extracting lamp spectrum and correcting for inclination
    _, sp_lamp = get_data_fit(obj_lamp,lims=lims_lamp, title='Row spectrum lamp', n=1, display_plots=display_plots)
    _, sp_lamp = angle_correction(sp_lamp, angle=angle, display_plots=display_plots)
    height = int((np.argmax(sp_lamp,axis=0)).sum()/sp_lamp.shape[1])
    # condition to display the images/plots
    if display_plots == True:
        showfits(sp_lamp, title='Rotated lamp image')
        plt.hlines(height,0,len(sp_lamp[0])-1,color='blue')
        plt.show()
    # taking the spectrum of the lamp at `height`
    spectrum_lamp = sp_lamp[height]
    # condition to display the images/plots
    if display_plots == True:
        fastplot(np.arange(len(spectrum_lamp)), spectrum_lamp, title=f'Lamp spectrum at y = {height}',labels=['x','counts'],grid=True)
        plt.show()

    # fit method
    # defining the linear function for the fit
    def fit_func(param,x):
        p0,p1,p2 = param
        return p0 + p1*x + p2*x**2
    # extracting the data for the calibration from `cal_file`
    lines, x, Dx = np.loadtxt(cal_file, unpack=True)
    Dy = np.full(lines.shape,3.63)

    pop, perr, pcov = fit_routine(x,lines,initial_values=initial_values,fit_func=fit_func,xerr=Dx,yerr=Dy,display_res=display_plots,return_res=['pcov'])
    _,p1,p2 = pop
    Dp0,Dp1,Dp2 = perr

    # defining the calibration function
    cal_func = lambda x : fit_func(pop,x)
    
    def err_func(x: np.ndarray, dx: np.ndarray, all_res: bool = False):
        dfdx = p1 + p2*2*x
        err = (dfdx*dx)**2
        if all_res:
            dfdp1 = x
            dfdp2 = x**2
            err += Dp0**2 + (dfdp1*Dp1)**2 + (dfdp2*Dp2)**2 + 2*( dfdp1*pcov[0,1] + dfdp2*pcov[0,2] + dfdp1*dfdp2*pcov[1,2]) 
        return np.sqrt(err)
        
    # condition to display the images/plots
    if display_plots == True:
        sigma = np.sqrt(Dy**2 + (p1*Dx + p2*x*Dx*2)**2)
        fig = plt.figure('Calibration',figsize=[8,7])
        fig.suptitle('Fit for lamp calibration')
        ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
        # plt.title('Calibration fit')
        ax1.errorbar(x,lines,xerr=Dx,yerr=Dy,fmt='.',color='green',label='data')
        ax1.plot(x,cal_func(x),color='orange',label='Best-fit')
        ax1.set_ylabel('$\lambda$ [$\AA$]')
        ax1.legend(numpoints=1)

        # plt.figure(figsize=[10,7])
        # plt.title('Residuals')
        ax2.axhline(0,xmin=0,xmax=1,linestyle='-.',color='black',alpha=0.5)
        ax2.errorbar(x,lines-cal_func(x),xerr=Dx,yerr=sigma,fmt='v',linestyle=':',color='green')
        ax2.set_xlabel('x [px]')
        ax2.set_ylabel('Residuals [$\AA$]')

        plt.show()

    return cal_func, err_func




def calibrated_spectrum(ch_obs: int, ch_obj: str, flat: None | np.ndarray = None, cal_func: Callable[[np.ndarray],np.ndarray] | None = None, err_func: Callable[[np.ndarray,np.ndarray,bool],np.ndarray] | None = None, display_plots: bool = False, initial_values: list[float] | tuple[float] = [3600, 2.6,0.], ret_values: str = 'few') -> list[np.ndarray | dict[str,np.ndarray] | dict[str,float | np.ndarray | Callable[[np.ndarray],np.ndarray]] | Callable[[np.ndarray,np.ndarray,bool],np.ndarray] | float]:
    """Getting the spectrum of a selceted target for a chosen observation night
    The function extracts fits data for a target and evaluates inclination correction, flat gain and calibration function to return the
    calibrated spectrum. If the flat gain or the calibration function are already computed then one can pass them to the function, 
    avoiding an other estimation.
    
    The function returns fits informations (`hdul`), spectrum image data (`sp_data`), cumulative spectrum (`spectrum`), corrisponding 
    wavelengths (`lenghts`), flat gain (`flat_value`) and calibration function (`cal_func`).
    
    It calls the functions:
      - `extract_data()`
      - `get_data()`
      - `showfits()`
      - `fastplot()`
      - `compute_flat()`
      - `calibration()`
    
    :param ch_obs: chosen observation night
    :type ch_obs: int
    :param ch_obj: chosen ob
    :type ch_obj: str
    :param flat: if the flat gain is not evaluated yet, the flat target name is passed
    :type flat: str | np.ndarray
    :param cal_func: if it is None, the calibration function will be computed, defaults to None
    :type cal_func: FUNC_TYPE | None, optional
    :param display_plots: if it is True images/plots are displayed, defaults to False
    :type display_plots: bool, optional
    
    :return: fits informations, spectrum image data, cumulative spectrum, corrisponding wavelenghts, flat gain and calibration function
    :rtype: tuple
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
        fastplot(np.arange(len(spectrum)), spectrum, title='Cumulative spectrum', labels=['x','counts'],grid=True)
        plt.show()

    ret_cond = (flat is None) and (cal_func is None) and (err_func is None)

    # estimating the flat gain
    flat_value = compute_flat(ch_obs, display_plots=display_plots) if flat is None else flat
    # correcting for the flat gain
    spectrum = spectrum / flat_value[:len(spectrum)]
    if display_plots == True:
        fastplot(np.arange(len(spectrum)), spectrum, title='Cumulative spectrum corrected by flat', labels=['x','counts'],grid=True)    
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
    fastplot(lengths, spectrum,title='Corrected and calibrated spectrum of ' + ch_obj,labels=['$\lambda$ [$\AA$]','counts'],dim=[17,9],grid=True)
    plt.show()

    if (ret_values == 'all') or (ret_values == 'few') or (ret_values == 'data')or (ret_values == 'calibration'):
        results = [spectrum, lengths]
    
    if (ret_values == 'all') or (ret_values == 'data'):
        data = { 'hdul' : hdul,
                 'sp_data' : sp_data }
        results += [data]
    if (ret_values == 'all') or (ret_values == 'calibration'):
        if ret_cond:
            cal_data = { 'angle' : angle,
                         'flat' : flat_value,
                         'func' : cal_func,
                         'err' : err_func }
        else:
            cal_data = angle
        results += [cal_data]
    
    return results


def mean_line(peaks: np.ndarray, spectrum: np.ndarray, dist: int = 3, height: int | float = 800) -> tuple[np.ndarray,np.ndarray]:
    """

    :param peaks: peaks
    :type peaks: np.ndarray
    :param spectrum: spectrum
    :type spectrum: np.ndarray
    :param dist: maximum distance between peaks, defaults to 3
    :type dist: int, optional
    :param height: the maximum value to start interpolation, defaults to 800
    :type height: int | float, optional

    :return: average peaks and the corresponding value of the spectrum 
    :rtype: tuple[np.ndarray,np.ndarray]
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

    print(height_diff)
    print(f'height -> {height}')

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
    """_summary_

    :param nights: selected observation nights (e.g. `(0,1)` or `0`)
    :type nights: tuple[int,int] | list[int] | int
    :param objs_name: names of the selected targets (e.g. `('betaLyr','vega')`)
    :type objs_name: tuple[str,str] | list[str]
    :param angles: the corresponding angles for image correction (if any), defaults to None
    :type angles: tuple[float  |  None, float  |  None] | list[float  |  None, float  |  None] | float | None, optional
    :param height: the smallest height of peaks, defaults to 2000
    :type height: int | float, optional
    :param display_plots: if `True` functional plots are displayed, defaults to False
    :type display_plots: bool, optional
    
    :return: max lag
    :rtype: float
    """
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
    
    plots = False

    obj1, cut1, lamp1, lims1 = extract_data(obs1,obj1,sel=['obj','lamp'])
    obj2, cut2, lamp2, lims2 = extract_data(obs2,obj2,sel=['obj','lamp'])

    _, _, angle1 = get_data('',obj1,cut1,display_plots=plots)
    _, _, angle2 = get_data('',obj2,cut2,display_plots=plots)

    _, lamp1 = get_data_fit(lamp1,lims1,title='lamp1',display_plots=plots)
    _, lamp2 = get_data_fit(lamp2,lims2,title='lamp2',display_plots=plots)

    _, lamp1 = angle_correction(lamp1,angle=angle1,display_plots=plots)
    _, lamp2 = angle_correction(lamp2,angle=angle2,display_plots=plots)

    plt.show()


    sel_height1 = int((np.argmax(lamp1,axis=0)).sum()/lamp1.shape[1])
    sel_height2 = int((np.argmax(lamp2,axis=0)).sum()/lamp2.shape[1])

    print(f'Sel heightt 1: {sel_height1}')
    print(f'Sel heightt 2: {sel_height2}')


    lamp1 = lamp1[sel_height1]
    lamp2 = lamp2[sel_height2]

    maxlamp1 = lamp1.max()
    maxlamp2 = lamp2.max()

    fact = min(maxlamp1,maxlamp2)/max(maxlamp1,maxlamp2)

    if maxlamp1 > maxlamp2:
        height1 = height
        height2 = height*fact
    elif maxlamp1 < maxlamp2:
        height1 = height*fact
        height2 = height
    else:
        height1, height2 = height, height

    print(height1,height2)

    ## Correlation
    from scipy.signal import find_peaks
    # finding the positions of the peaks in lamp spectra
    pkslamp1, _ = find_peaks(lamp1,height=height1)
    pkslamp2, _ = find_peaks(lamp2,height=height2)

    pkslamp1 = pkslamp1.astype(int)
    pkslamp2 = pkslamp2.astype(int)

    mpks1, mline1 = mean_line(pkslamp1,lamp1)
    mpks2, mline2 = mean_line(pkslamp2,lamp2)
    print('0: ',len(mpks1),len(mpks2))
    
    cnt = 0
    dim_diff = len(mpks1)-len(mpks2)
    while(dim_diff != 0):
        cnt += 1
        if dim_diff > 0: height1 += 100
        else: height2 += 100

        pkslamp1, _ = find_peaks(lamp1,height=height1)
        pkslamp2, _ = find_peaks(lamp2,height=height2)

        pkslamp1 = pkslamp1.astype(int)
        pkslamp2 = pkslamp2.astype(int)

        mpks1, mline1 = mean_line(pkslamp1,lamp1)
        mpks2, mline2 = mean_line(pkslamp2,lamp2)
        
        print(f'{cnt}: ',len(mpks1),len(mpks2))
        dim_diff = len(mpks1)-len(mpks2)

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

    print(len(pkslamp1),len(pkslamp2))
    print(f'Max diff: {abs(mpks1-mpks2).max()}')

    # computing the correlation and the autocorrelation
    corr = np.correlate(mpks1.astype(float),mpks2.astype(float),'full')
    autocorr = np.correlate(mpks2.astype(float),mpks2.astype(float),'full')
    # corr = np.correlate(pkslamp1.astype(float),pkslamp2.astype(float),'full')
    # autocorr = np.correlate(pkslamp2.astype(float),pkslamp2.astype(float),'full')
    

    # computing the max lag
    maxlag = np.abs(corr/max(corr)-autocorr/max(autocorr)).max()
    # checking the max lag
    print(f'max(|auto_corr - corr|) = {maxlag}')
    # if maxlag < 1:
    #     print(f'max(|auto_corr - corr|) = {maxlag}')
    # else:
    #     raise Exception(f'\nPeaks positions in lamps have to be shifted!\nMAXLAG = {maxlag}')

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

    




