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
    obj_fit, lims_fit, _, _, obj_flat, lims_flat = extract_data(ch_obs, flat_obj)
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


def calibration(cal_file: str, obj_lamp: str, lims_lamp: list, angle: float, high: int, initial_values: list[float] = [3600, 2.6,0.], display_plots: bool = False) -> Callable[[np.ndarray],np.ndarray]:
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
    :param high: y coorfinate at which the spectrum is taken
    :type high: int
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
    # condition to display the images/plots
    if display_plots == True:
        showfits(sp_lamp, title='Rotated lamp image')
        plt.hlines(high,0,len(sp_lamp[0])-1,color='blue')
        plt.show()
    # taking the spectrum of the lamp at `high`
    spectrum_lamp = sp_lamp[high]
    # condition to display the images/plots
    if display_plots == True:
        fastplot(np.arange(len(spectrum_lamp)), spectrum_lamp, title=f'Lamp spectrum at y = {high}',labels=['x','counts'])
        plt.show()

    # fit method
    # defining the linear function for the fit
    def fit_func(param,x):
        p0,p1,p2 = param
        return p0 + p1*x + p2*x**2
    # extracting the data for the calibration from `cal_file`
    lines, x, Dx = np.loadtxt(cal_file, unpack=True)
    Dy = np.full(lines.shape,3.63)

    pop, _ = fit_routine(x,lines,initial_values=initial_values,fit_func=fit_func,xerr=Dx,yerr=Dy,display_res=display_plots)
    _,p1,p2 = pop

    # defining the calibration function
    cal_func = lambda x : fit_func(pop,x)
    
    # condition to display the images/plots
    if display_plots == True:
        sigma = np.sqrt(Dy**2 + (p1*Dx)**2 + (p2*x*Dx*2)**2)
        fig = plt.figure('Calibration',figsize=[8,7])
        fig.suptitle('Fit for lamp calibration')
        ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
        # plt.title('Calibration fit')
        ax1.errorbar(x,lines,xerr=Dx,yerr=Dy,fmt='.',color='violet',label='data')
        ax1.plot(x,cal_func(x),label='Best-fit')
        ax1.set_ylabel('$\lambda$ [$\AA$]')
        ax1.legend(numpoints=1)

        # plt.figure(figsize=[10,7])
        # plt.title('Residuals')
        ax2.axhline(0,xmin=0,xmax=1,linestyle='-.',alpha=0.5)
        ax2.errorbar(x,lines-cal_func(x),xerr=Dx,yerr=sigma,fmt='v',linestyle=':',color='violet')
        ax2.set_xlabel('x [px]')
        ax2.set_ylabel('Residuals [$\AA$]')

        plt.show()

    return cal_func




def calibrated_spectrum(ch_obs: int, ch_obj: str, flat: list[None] | np.ndarray = [], cal_func: Callable[[np.ndarray],np.ndarray] | None = None, display_plots: bool = False, initial_values: list[float] | tuple[float] = [3600, 2.6,0.]) -> tuple:
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
    obj_fit, lims_fit, obj_lamp, lims_lamp = extract_data(ch_obs, ch_obj)
    # extracting fits data and correcting for inclination
    hdul, sp_data, angle = get_data(ch_obj,obj_fit,lims_fit, display_plots=display_plots)
    if display_plots == False: 
        showfits(sp_data, title='Spectrum of '+ ch_obj)
        plt.show() 

    # computing the cumulative spectrum over columns
    spectrum = sp_data.sum(axis=0)
    if display_plots == True:
        fastplot(np.arange(len(spectrum)), spectrum, title='Cumulative spectrum', labels=['x','counts'])
        plt.show()

    # estimating the flat gain
    flat_value = compute_flat(ch_obs, display_plots=display_plots) if len(flat) == 0 else flat
    # correcting for the flat gain
    spectrum = spectrum / flat_value[:len(spectrum)]
    if display_plots == True:
        fastplot(np.arange(len(spectrum)), spectrum, title='Cumulative spectrum corrected by flat', labels=['x','counts'])    
        plt.show()
        
    # condition to compute the calibration function
    if cal_func == None:
        # getting the path of the calibration file
        cal_file = os.path.join(os.path.split(obj_fit)[0],'calibration_lines.txt')
        #! CONDIZIONE DA CONTROLLARE 
        high = int((685.7+26.8) / 2)
        # estimating the calibration function
        cal_func = calibration(cal_file=cal_file, obj_lamp=obj_lamp, lims_lamp=lims_lamp, angle=angle, high=high,initial_values=initial_values,display_plots=display_plots)
    # getting the corrisponding wavelengths
    lengths = cal_func(np.arange(len(spectrum)))
    # displaying the calibrated spectrum
    fastplot(lengths, spectrum,title='Corrected and calibrated spectrum of ' + ch_obj,labels=['$\lambda$ [$\AA$]','counts'],dim=[17,9])
    plt.show()

    return hdul, sp_data, spectrum, lengths, flat_value, cal_func
