import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from test_func import collect_fits, data_file_path, fastplot, showfits


##* 
def get_data_fit(path: str, lims: list = [0,-1,0,-1], hotpx: bool = False, v: int = -1, title: str = '', n: int = None, dim: list[int] = [10,7]) -> tuple:
    """Function to open fits file and extract data.
    It brings the path and extracts the data, giving a row image.
    You can set a portion of image and also the correction for hotpx.

    It calls the functions `hotpx_remove` and `showfits`.

    :param path: path of the fits file
    :type path: str
    :param lims: edges of the fits, defaults to [0,-1,0,-1]
    :type lims: list, optional
    :param hotpx: parameter to remove or not the hot pixels, defaults to False
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

    # data extraction
    # format -> data[Y,X]
    data = hdul[0].data
    ly,ry,lx,rx = lims
    data = data[ly:ry,lx:rx]
    # hot px correction
    if hotpx == True:
        data = hotpx_remove(data)
    # Spectrum image
    showfits(data, v=v,title=title,n=n,dim=dim) 
    return hdul,data
##*

##*
def angle_correction(data: np.ndarray, init: list[float] = [0.9, 0.], angle: float | None = None) -> tuple[float, np.ndarray]:
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
        print(f'\n- Fit results -\nm =\t{m} +- {Dm}\nq =\t{q} +- {Dq}\ncor =\t{pcov[0,1]/Dm/Dq}\nangle =\t{angle} +- ({Dangle/angle*100:.2f} %) deg')
        fastplot(x_pos,y_pos,2,'+')
        fastplot(x_pos,fitlin(x_pos,m,q),2,'-',labels=['x','y'])

    data_rot  = ndimage.rotate(data, angle, reshape=False)
    return angle, data_rot
##*





if __name__ == '__main__':
    # selecting the observation night
    sel_obs = 1
    # choosing the object
    # sel_obj = 'betaLyr'
    sel_obj = 'chcygni'
    # collecting data fits for that object
    obj, lims = collect_fits(sel_obs, sel_obj)
    if sel_obj != 'giove':
        obj_fit, obj_lamp = obj
    else:
        obj_fit, obj_lamp, obj_flat = obj
    print(obj_fit)
    obj_fit = obj_fit[0]
    # appending the path
    obj_fit = data_file_path(sel_obs, sel_obj, obj_fit)
    obj_lamp = data_file_path(sel_obs, sel_obj,obj_lamp)
    if sel_obj == 'giove':
        obj_flat = data_file_path(sel_obs, sel_obj,obj_flat)

    hdul, spectrum = get_data_fit(obj_fit)
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
    hdul_lamp, lamp = get_data_fit(obj_lamp)
    if sel_obj == 'giove':
        hdul_flat, flat = get_data_fit(obj_flat)
    lamp = lamp[640:1373, 750:-1]
    showfits(spectrum,n=3)
    showfits(lamp,n=4)
    plt.show()