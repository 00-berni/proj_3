import numpy as np
from scipy import ndimage
from typing import Callable
from .display import fastplot


# definition of function type to use in docstring of functions
FUNC_TYPE = type(abs)

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
    from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
    # checking the presence of `NaNs`
    if True in np.isnan(data):
        # building a gaussian kernel for the interpolation
        kernel = Gaussian2DKernel(x_stddev=1)
        # removing the `NaNs`
        data = interpolate_replace_nans(data, kernel)
    return data


def fit_routine(xdata: np.ndarray, ydata: np.ndarray, initial_values: list[float], fit_func: Callable[[np.ndarray],np.ndarray], xerr: np.ndarray | float | int | None = None, yerr: np.ndarray | float | None = None, iter: int | None = None, return_res: list[str] | tuple[str] | None = None, display_res: bool = False) -> list:
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
            fastplot(x_pos,fitlin(x_pos,m,q),2,'-',labels=['x','y'])

    data_rot  = ndimage.rotate(data, angle, reshape=False)
    return angle, data_rot
