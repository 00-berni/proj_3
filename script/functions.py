"""
SCRIPT FOR FUNCTIONS IMPLEMENTATION

These are the functions used in the `fr_spectrum1.py` script

The implemented functions are:
 - `fastplot` 	:	Makes a simple plot
 - `hotpxRemove` : 	Removes hotpx [taken by internet]
 - `showfits` 	:	Prints fits image
 - `targetDatafit` :	Extracts data from fits and print a row image
 - `RotCor`	:	Corrects the inclination of the spectrum, making a linear fit
 - `initialization`

"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.ndimage as ndimage
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
# from scipy.signal import correlate

# definition of function type to use in docstring of functions
FUNC_TYPE = type(abs)


def data_extraction(path_file: str) -> dict:
    """Extracting data from a .json file

    :param path_file: path of the data file
    :type path_file: str
    
    :return: data organized into nights of aquisition and objects
    :rtype: dict
    """
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
def fastplot(x: np.ndarray, y: np.ndarray, numfig: int = None, fmt: str = '-', title: str = '', labels: list[str] = ['',''], dim: list[int] = [10,7]) -> None:
    """Function to display a plot quickly.
    You can choose to make a simple plot or adding some stuff.

    (I wrote it only because of my laziness in writing code).

    :param x: Data on x axis
    :type x: np.ndarray
    :param y: Data on y axis
    :type y: np.ndarray
    :param numfig: figure number, defaults to None
    :type numfig: int, optional
    :param title: title of the figure. default to ' '
    :type title: str, optional
    :param labels: axes label [x,y] format, defaults to ['','']
    :type labels: list[str], optional
    :param dim: figure size, defaults to [10,7] 
    :type dim: list[int], optional
    """
    xl,yl = labels
    plt.figure(numfig,figsize=dim)
    plt.title(title)
    plt.plot(x,y,fmt)
    plt.xlabel(xl)
    plt.ylabel(yl)



##*
def showfits(data: np.ndarray, v: int = -1, title: str = '', n: int = None, dim: list[int] = [10,7]) -> None:
    """Function to display the fits image.
    You can display simply the image or set a figure number and a title.

    :param data: image matrix of fits file
    :type data: np.ndarray
    :param v: cmap parameter: 1 for false colors, 0 for grayscale, -1 for reversed grayscale; defaults to -1
    :type v: int, optional
    :param title: title of the image, defaults to ''
    :type title: str, optional
    :param n: figure number, defaults to None
    :type n: int, optional
    :param dim: figure size, defaults to [10,7]
    :type dim: list[int], optional
    """
    plt.figure(n,figsize=dim)
    plt.title(title)
    if v == 1 : color = 'viridis'
    elif v == 0 : color = 'gray'
    else : color = 'gray_r'
    plt.imshow(data, cmap=color)
    plt.colorbar(orientation='horizontal')
##*


def hotpx_remove(data: np.ndarray) -> np.ndarray:
    """Removing hot pixels from the image
    The function replacing `NaN` values from
    the image if there are.
    I did not implement this function, I
    took it from [*astropy documentation*](https://docs.astropy.org/en/stable/convolution/index.html)

    :param data: spectrum data
    :type data: np.ndarray
    
    :return: spectrum data without `NaN` values
    :rtype: np.ndarray
    """
    # checking the presence of `NaNs`
    if True in np.isnan(data):
        # building a gaussian kernel for the interpolation
        kernel = Gaussian2DKernel(x_stddev=1)
        # removing the `NaNs`
        data = interpolate_replace_nans(data, kernel)
    return data

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
    if display_plots == True: showfits(data, v=v,title=title,n=n,dim=dim) 
    return hdul,data
##*

##*
def angle_correction(data: np.ndarray, init: list[float] = [0.9, 0.], angle: float | None = None, display_plots: bool = True) -> tuple[float, np.ndarray]:
    """Function to correct the inclination, rotating the image.
    It takes the maximum of each column and does a fit to find 
    the angle with the horizontal. The the image is rotated.

    :param data: image matrix
    :type data: np.ndarray
    :param init: initial values for the fit, defaults to [0.9,0.]
    :type init: list[float], optional

    :return: inclination angle and the corrected data
    :rtype: tuple[float, np.ndarray]
    """
    if angle == None:
        y_pos = np.argmax(data, axis=0)
        x_pos = np.arange(len(y_pos))
        def fitlin(x,m,q):
            return x*m+q
        initial_values = init
        for i in range(3):
            pop, pcov = curve_fit(fitlin,x_pos,y_pos,initial_values)
            m, q = pop
            Dm, Dq = np.sqrt(pcov.diagonal())
            if m == initial_values[0] and q == initial_values[1]: break
            initial_values = pop

        angle = np.arctan(m)*180/np.pi   # degrees
        Dangle = 180/np.pi * Dm/(1+m**2)
        if display_plots == True:
            print(f'\n- Fit results -\nm =\t{m} +- {Dm}\nq =\t{q} +- {Dq}\ncor =\t{pcov[0,1]/Dm/Dq}\nangle =\t{angle} +- ({Dangle/angle*100:.2f} %) deg')
            fastplot(x_pos,y_pos,2,'+')
            fastplot(x_pos,fitlin(x_pos,m,q),2,'-',labels=['x','y'])

    data_rot  = ndimage.rotate(data, angle, reshape=False)
    return angle, data_rot


def extract_data(ch_obs: int, ch_obj: str, flat: bool = False) -> list:
    """Collecting data from data files
    Given a selected observation night and object, the function 
    returns both the paths of the target and the calibration lamp
    (also the flat if there is) and the corrisponding values for 
    the edges of the images. 

    :param ch_obs: chosen obeservation night
    :type ch_obs: int
    :param ch_obj: chosen object name
    :type ch_obj: str
    :param flat: if also the flat is present, the parameter is True, defaults to False
    :type flat: bool, optional
    
    :return: the list with paths and extrema
    :rtype: list
    """
    # extracting informations
    obj, lims = collect_fits(ch_obs, ch_obj)
    # condition for the presence of flat
    if flat == False:
        # collecting in different variables
        obj_fit, obj_lamp = obj 
        lims_fit, lims_lamp = lims

        # appending the path
        obj_fit = data_file_path(ch_obs, ch_obj, obj_fit)
        obj_lamp = data_file_path(ch_obs, ch_obj, obj_lamp)
        
        # storing in `data` list
        #   format is [obj, its extrema, ...]
        data = [obj_fit, lims_fit, obj_lamp, lims_lamp]         
    else:
        # collecting in different variables
        obj_fit, obj_lamp, obj_flat = obj 
        lims_fit, lims_lamp, lims_flat = lims

        # appending the path
        obj_fit = data_file_path(ch_obs, ch_obj, obj_fit)
        obj_lamp = data_file_path(ch_obs, ch_obj, obj_lamp)
        obj_flat = data_file_path(ch_obs, ch_obj, obj_flat)

        # storing in `data` list
        #   format is [obj, its extrema, ...]
        data = [obj_fit, lims_fit, obj_lamp, lims_lamp, obj_flat, lims_flat]
    return data



def get_data(ch_obj: str, obj_fit: str, lims_fit: list[int], angle: float | None = None, display_plots: bool = False) -> tuple:
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



def compute_flat(ch_obs: int, flat_obj: str, display_plots: bool = False) ->  np.ndarray:
    """Evaluating the flat gain
    After extracting the data of the flat acquisition, the function finds the
    maximum and computes the cumulative spectrum over the y axis. Then the 
    gain is extimated for each x coordinate by normalization through the 
    maximum and returned.

    :param ch_obs: chosen observation night
    :type ch_obs: int
    :param flat_obj: name of the target for which flat was acquired
    :type flat_obj: str
    :param display_plots: if it is True images/plots are displayed, defaults to False
    :type display_plots: bool, optional
    
    :return: flat gain for each x coordinate
    :rtype: np.ndarray
    """
    # collecting informations for the target and the flat
    obj_fit, lims_fit, _, _, obj_flat, lims_flat = extract_data(ch_obs, flat_obj, flat=True)
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


def calibration(cal_file: str, obj_lamp: str, lims_lamp: list, angle: float, high: int, initial_values: list[float] = [3600, 2.6], display_plots: bool = False) -> FUNC_TYPE:
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
    def fit_func(x,p0,p1):
        return p0 + p1*x
    # extracting the data for the calibration from `cal_file`
    lines, x, Dx = np.loadtxt(cal_file, unpack=True)
    # running the fit method three times to increase accuracy
    pop, pcov = curve_fit(fit_func,x,lines,initial_values)
    for _ in range(2):
        initial_values = pop
        # esitmating the error over y variable
        Dy = Dx*pop[1]
        pop, pcov = curve_fit(fit_func,x,lines,initial_values, sigma=Dy)
    # extracting values of parameters
    p0,p1 = pop
    Dp0,Dp1 = np.sqrt(pcov.diagonal())
    #! DA CONTROLLARE
    # computing the sigma
    sigma = Dy**2 + Dp0**2 + (Dp1*x)**2 + pcov[0][1]
    chiq = sum((lines - fit_func(x,p0,p1))**2/sigma)
    #!
    # printing the results
    print(f'Fit results for 2 params\np0 =\t{p0} +- {Dp0}\np1 =\t{p1} +- {Dp1}\ncor =\t{pcov[0][1]/Dp0/Dp1*100} %')
    print(f'Chi_red =\t{chiq/(len(x)-2)} +- {np.sqrt(len(x)-2)}')
    
    # defining the calibration function
    cal_func = lambda x : fit_func(x,p0,p1)
    
    print('Lengths\nl\t\tDl\tl\t\tDl')
    for i in range(len(spectrum_lamp)//2):
        print(f'{cal_func(i):.0f}\t{np.sqrt(Dp0**2 + (Dp1*i)**2 + pcov[0][1]):.0f}\t{cal_func(i+len(spectrum_lamp)//2):.0f}\t{np.sqrt(Dp0**2 + (Dp1*(i+len(spectrum_lamp)//2))**2 + pcov[0][1]):.0f}')
    
    # condition to display the images/plots
    if display_plots == True:
        plt.figure(figsize=[10,7])
        plt.title('Calibration fit')
        plt.errorbar(x,lines,xerr=Dx,fmt='x',color='violet')
        plt.plot(x,cal_func(x))
        plt.xlabel('x [px]')
        plt.ylabel('$\lambda$ [$\AA$]')

        plt.figure()
        plt.title('Residuals')
        plt.hlines(0,min(x),max(x),linestyles='-.',alpha=0.5)
        plt.plot(x,(lines-cal_func(x))/np.sqrt(sigma),'v:',color='violet')
        plt.xlabel('x [px]')
        plt.ylabel('$(y-f(x))/\sigma$')

        plt.show()

    return cal_func



def calibrated_spectrum(ch_obs: int, ch_obj: str, flat: str | np.ndarray, cal_func: FUNC_TYPE | None = None, display_plots: bool = False) -> tuple:
    """Getting the spectrum of a selected target for a chosen observation night
    The function extracts fits data for a target and evaluates inclination correction, flat gain and calibration function to return the
    calibrated spectrum. If the flat gain or the calibration function are already computed then one can pass them to the function, 
    avoiding an other estimation.
    The function returns fits informations (`hdul`), spectrum image data (`sp_data`), cumulative spectrum (`spectrum`), corrisponding 
    wavelengths (`lenghts`), flat gain (`flat_value`) and calibration function (`cal_func`).
    

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
    flat_value = compute_flat(ch_obs,flat, display_plots=display_plots) if type(flat) == str else flat
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
        cal_func = calibration(cal_file, obj_lamp, lims_lamp, angle, high)
    # getting the corrisponding wavelengths
    lengths = cal_func(np.arange(len(spectrum)))
    # displaying the calibrated spectrum
    fastplot(lengths, spectrum,title='Corrected and calibrated spectrum of ' + ch_obj,labels=['$\lambda$ [$\AA$]','counts'],dim=[17,9])
    plt.show()

    return hdul, sp_data, spectrum, lengths, flat_value, cal_func


