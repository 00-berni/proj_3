"""
STUFF PACKAGE
=============

***

::METHODS::
-----------

***

!TO DO!
-------
    - [] **Update `fit_routine`**
    - [] **Change `angle_correction`**


***
    
?WHAT ASK TO STEVE?
-------------------
"""

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from typing import Callable, Sequence
from .display import fastplot
from astropy.io.fits import HDUList


class Spectrum():
    
    @staticmethod
    def empty():
        return Spectrum([],[],[],False)

    def __init__(self, hdul: HDUList | Sequence[HDUList] | None, data: NDArray | None, lims: NDArray | None, hotpx: bool = True) -> None:
        self.hdul = hdul
        self.data = hotpx_remove(data) if hotpx else data 
        self.lims = lims

    def print_header(self) -> None:
        # print header
        hdr = self.hdul[0].header
        print(' - HEADER -')
        for parameter in hdr:
            hdr_info = f'{parameter} =\t{hdr[parameter]}' 
            comm = hdr.comments[parameter]
            if comm != '': hdr_info = hdr_info + ' \ ' + comm 
            print(hdr_info)
        print()
    
    def get_exposure(self) -> float:
        header = self.hdul[0].header
        return header['EXPOSURE']
    
    def copy(self):
        target = Spectrum(self.hdul, self.data, self.lims, hotpx=False)
        return target


# definition of function type to use in docstring of functions
FUNC_TYPE = type(abs)

def make_cut_indicies(file_path: str, lines_num: int) -> NDArray:
    cut = np.array([[0,-1,0,-1]]*lines_num,dtype=int)
    content = "#\tThe section of image to display\n#\n#\tThe first row is for the target acquisition\n#\tThe last one is for the lamp\n#\n#yl\tyu\txl\txu"
    for line in cut.astype(str):
        content = content + '\n' + '\t'.join(line)
    f = open(file_path, "w")
    f.write(content)
    f.close()
    return cut

def hotpx_remove(data: NDArray) -> NDArray:
    """Removing hot pixels from the image

    Parameters
    ----------
    data : NDArray
        spectrum data

    Returns
    -------
    data : NDArray
        spectrum data without `NaN` values
    
    Notes
    -----
    The function replacing `NaN` values from the image if there are.
    I did not implement this function, I took it from [*astropy documentation*](https://docs.astropy.org/en/stable/convolution/index.html)

    """
    from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
    # checking the presence of `NaNs`
    if True in np.isnan(data):
        # building a gaussian kernel for the interpolation
        kernel = Gaussian2DKernel(x_stddev=1)
        # removing the `NaNs`
        data = interpolate_replace_nans(data, kernel)
    return data


def fit_routine(xdata: NDArray, ydata: NDArray, initial_values: list[float], fit_func: Callable[[NDArray],NDArray], xerr: NDArray | float | int | None = None, yerr: NDArray | float | None = None, iter: int | None = None, return_res: list[str] | tuple[str] | None = None, display_res: bool = False) -> list[NDArray]:
    """_summary_

    Parameters
    ----------
    xdata : NDArray
        _description_
    ydata : NDArray
        _description_
    initial_values : list[float]
        _description_
    fit_func : Callable[[NDArray],NDArray]
        _description_
    xerr : NDArray | float | int | None, optional
        _description_, by default None
    yerr : NDArray | float | None, optional
        _description_, by default None
    iter : int | None, optional
        _description_, by default None
    return_res : list[str] | tuple[str] | None, optional
        _description_, by default None
    display_res : bool, optional
        _description_, by default False

    Returns
    -------
    list[NDArray]
        _description_
    """
    if isinstance(xerr,(float,int)):
        xerr = np.full(xdata.shape,xerr)
    if isinstance(yerr,(float,int)):
        yerr = np.full(ydata.shape,yerr)

    if xerr is not None:
        from scipy import odr
        data = odr.RealData(xdata,ydata,sx=xerr,sy=yerr)
        model = odr.Model(fit_func)
        fit = odr.ODR(data,model,beta0=initial_values)
        out = fit.run()
        pop = out.beta
        pcov = out.cov_beta
        perr = np.sqrt(pcov.diagonal())
        chisq = out.sum_square
        free = len(xdata) - len(pop)

        results = [pop,perr]
        if return_res is not None:
            if 'pcov' in return_res:
                results += [pcov]
            if 'chisq' in return_res:
                results += [chisq]
            if 'free' in return_res:
                results += [free]
            if 'out' in return_res:
                results += [out]
            if 'fit' in return_res:
                results += [fit]
        
    else:
        if iter is None:
            iter = 2
        from scipy.optimize import curve_fit
        pop, pcov = curve_fit(fit_func,xdata,ydata,initial_values)
        for i in range(iter):
            initial_values = pop
            pop, pcov = curve_fit(fit_func,xdata,ydata,initial_values,sigma=yerr)
        perr = np.sqrt(pcov.diagonal())
        chisq = (((ydata-fit_func(xdata,*pop))/yerr)**2).sum()
        free = len(xdata) - len(pop)
        
        results = [pop,perr]
        if return_res is not None:
            if 'pcov' in return_res:
                results += [pcov]
            if 'chisq' in return_res:
                results += [chisq]
            if 'free' in return_res:
                results += [free]


    if display_res:

        str_res = '\n'.join([f'p{i} = {pop[i]:e} +- {perr[i]:e}\t-> {perr[i]/pop[i]*100:.2f} %' for i in range(len(pop))])
        if len(pop) > 1:
            str_corr = []
            for i in range(pcov.shape[0]):
                str_corr += [f'corr_{i}{j} =\t {pcov[i,j]/np.sqrt(pcov[i,i]*pcov[j,j])*100:.2f} %' for j in range(i+1,pcov.shape[1])]
            str_corr = '\n'.join(str_corr)
            str_res += '\n' + str_corr
        
        print('\n--- Results of the fit ---\n' + str_res + f'\n\u03C7\u00b2_red =\t{chisq/free:.2f} +- {np.sqrt(2/free):.2f}\n----------------------\n')

    return results




def angle_correction(data: NDArray, init: list[float] = [0.9, 0.], angle: float | None = None, display_plots: bool = True) -> tuple[float, NDArray]:
    """Function to correct the inclination, rotating the image.
    
    It takes the maximum of each column and fit in order to find 
    the angle with the horizontal. The the image is rotated.

    Parameters
    ----------
    data : NDArray
        image matrix
    init : list[float], optional
        initial values for the fit, by default `[0.9, 0.]`
    angle : float | None, optional
        _description_, by default `None`
    display_plots : bool, optional
        _description_, by default `True`

    Returns
    -------
    angle : float
        inclination angle
    data_rot : NDArray
        the corrected data
    """
    if angle == None:
        y_pos = np.argmax(data, axis=0)
        x_pos = np.arange(len(y_pos))
        
        Dy = 1

        def fitlin(x,m,q):
            return x*m+q

        pop, perr = fit_routine(x_pos,y_pos,init,fitlin,display_res=display_plots,yerr=Dy)
        m, q = pop
        Dm, _ = perr

        angle = np.arctan(m)*180/np.pi   # degrees
        Dangle = 180/np.pi * Dm/(1+m**2)

        if display_plots == True:
            print(f'Estimated Angle:\ntheta = {angle:e} +- {Dangle:e} deg\t-> {Dangle/angle*100} %')
            fastplot(x_pos,y_pos,2,'+')
            fastplot(x_pos,fitlin(x_pos,*pop),2,'-',labels=['x','y'])

    data_rot  = ndimage.rotate(data, angle, reshape=False)
    return angle, data_rot
