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
from numpy import ndarray
from typing import Callable, Sequence, Any
from astropy.io.fits import HDUList
import matplotlib.pyplot as plt


class Spectrum():
    """To store spectrum information

    Attributes
    ----------
    name : str
        the name of the object
    hdul : HDUList | Sequence[HDUList] | None
        hdul information
    data : ndarray | None
        spectrum values
    lims : ndarray | None
        edges of cutted image
    spec : ndarray | None
        _description_
    """
    @staticmethod
    def empty():
        """To generate an empty object

        Returns
        -------
        Spectrum
            empty object
        """
        return Spectrum([],[],[],False,name='empty')

    def __init__(self, hdul: HDUList | Sequence[HDUList] | None, data: ndarray | None, lims: ndarray | None, hotpx: bool = True, name: str = '') -> None:
        """Constructor of the class

        Parameters
        ----------
        hdul : HDUList | Sequence[HDUList] | None
            hdul information
        data : ndarray | None
            spectrum values
        lims : ndarray | None
            edges of cutted image
        hotpx : bool, optional
            to filter and remove hot pixels, by default `True`
        name : str, optional
            the name of the object, by default `''`
        """
        self.name = name
        self.hdul = hdul
        self.data = hotpx_remove(data) if hotpx else data 
        self.lims = lims
        self.spec = None

    def print_header(self) -> None:
        """To print the header of fits file"""
        # take the header
        hdr = self.hdul[0].header
        print(' - HEADER -')
        for parameter in hdr:
            hdr_info = f'{parameter} =\t{hdr[parameter]}' 
            comm = hdr.comments[parameter]
            if comm != '': hdr_info = hdr_info + ' \ ' + comm 
            print(hdr_info)
        print()
    
    def get_exposure(self) -> float:
        """To get the exposure time

        Returns
        -------
        exp_time : float
            exposure time
            For different acquisitions the mean is computed
        """
        if isinstance(self.hdul[0], HDUList):
            exp_time = []
            for hd in self.hdul:
                header = hd[0].header
                exp_time += [header['EXPOSURE']]
            exp_time = np.mean(exp_time)
        else:
            header = self.hdul[0].header
            exp_time = header['EXPOSURE']
        return exp_time
    
    def cut_image(self) -> None:
        """To cut the image"""
        lims = (slice(*self.lims[:2]),slice(*self.lims[2:]))
        self.data = self.data[lims]

    def angle_correction(self, angle: float | None = None, init: list[float] = [0.9, 0.]) -> tuple:
        """To correct the inclination of spectrum, rotating the image.
        
        Parameters
        ----------
        angle : float | None, optional
            inclination angle of the slit, by default `None`
            If `angle is None` then the value is estimated by a 
            fitting rountine
        init : list[float], optional
            initial values for the fit, by default `[0.9, 0.]`

        Returns
        -------
        target : Spectrum
            the corrected data
        angle : float
            inclination angle
        
        Notes
        -----
        The method consists of two steps:
          1. Take the maximum of each column and compute a linear fit in order to find 
          the inclination angle
          2. Fit each column with a gaussian profile, get the mean position and repeat
          the linear fit to estimate the angle
        After then the image is rotated
        """
        target = self.copy()
        data = target.data
        from scipy import ndimage
        if angle == None:
            y_pos = np.argmax(data, axis=0)
            x_pos = np.arange(len(y_pos))
            
            Dy = 0.5
            Dx = 0.5

            def fitlin(x,m,q):
                return x*m+q

            fit = FuncFit(xdata=x_pos, ydata=y_pos, yerr=Dy, xerr=Dx)
            fit.pipeline(fitlin,init,names=['m','q'])
            pop, perr = fit.results()
            m, q = pop
            Dm, _ = perr
            angle = np.arctan(m)*180/np.pi   # degrees
            Dangle = 180/np.pi * Dm/(1+m**2)
            data = ndimage.rotate(data, angle, reshape=False).copy()

            fig, ax = plt.subplots(1,1) 
            y_pos, Dy = np.array([]), np.array([])
            x_pos = x_pos[::50]
            for i in x_pos:
                values = data[:,i]
                y = np.arange(len(values))
                hwhm = abs(np.argmax(values) - np.argmin(abs(values - max(values)/2)))
                initial_values = [max(values),np.argmax(values),hwhm]
                pos = np.where(values > np.mean(values)*2)
                values = values[pos]
                y = y[pos]
                fit = FuncFit(xdata=y, ydata=values,xerr=0.5)
                fit.gaussian_fit(initial_values)
                pop, perr = fit.results()
                y_pos = np.append(y_pos,pop[1])
                Dy = np.append(Dy,pop[2])
                method = fit.res['func']
                color = (1-i/max(x_pos),i/max(x_pos),i/(2*max(x_pos))+0.5)
                ax.plot(data[:,i],color=color,label='fit')
                ax.plot(y, method(y,*pop), '--',color=color)
            ax.legend()
            
            print(len(x_pos),len(y_pos))

            fit = FuncFit(xdata=x_pos, ydata=y_pos, yerr=Dy)
            fit.pipeline(fitlin,init,names=['m','q'])
            pop, perr = fit.results()
            m  = pop[0]
            Dm = perr[0]

            angle = np.arctan(m)*180/np.pi   # degrees
            Dangle = 180/np.pi * Dm/(1+m**2)

        data_rot = ndimage.rotate(data, angle, reshape=False)
        target.data = data_rot
        return target, angle

    def copy(self):
        """To make an identical copy of a Spectrum object

        Returns
        -------
        target : Spectrum
            the copy
        """
        target = Spectrum([*self.hdul], self.data.copy(), self.lims.copy(), hotpx=False, name=self.name)
        return target

    def __add__(self, spec: Any) -> ndarray:
        if isinstance(spec, Spectrum):
            return self.data + spec.data
        else:
            return self.data + spec

    def __radd__(self, other):
            return self.__add__(other)

    def __sub__(self, spec: Any) -> ndarray:
        if isinstance(spec, Spectrum):
            return self.data - spec.data
        else:
            return self.data - spec

    def __rsub__(self, other):
            return -self.__sub__(other)

    def __mul__(self, spec: Any) -> ndarray:
        if isinstance(spec, Spectrum):
            return self.data * spec.data
        else:
            return self.data * spec

class FuncFit():
    """To compute the fit procedure of some data

    Attributes
    ----------
    data : list[ndarray | None]
        the x and y data and (if there are) their uncertanties
    fit_par : ndarray | None
        fit estimated parameters
    fit_err : ndarray | None
        uncertanties of `fit_par`
    res : dict
        it collects all the results

    Examples
    --------
    Simple use:
    >>> def lin_fun(x,m,q):
    ...     return m*x + q
    >>> initial_values = [1,1]
    >>> fit = FuncFit(xdata=xfit, ydata=yfit, yerr=yerr)
    >>> fit.pipeline(lin_fun, initial_values, names=['m','q'])
    Fit results:
        m = 3 +- 1
        q = 0.31 +- 0.02
        red_chi = 80 +- 5 %
    >>> pop, Dpop = fit.results()

    A method provides the gaussian fit:
    >>> fit = FuncFit(xdata=xfit, ydata=yfit, yerr=errfit)
    >>> fit.gaussian_fit(initial_values, names=['k','mu','sigma'])
    Fit results:
        k = 10 +- 1
        mu = 0.01 +- 0.003
        sigma = 0.20 +- 0.01
        red_chi = 72 +- 15 %
    """
    def __init__(self, xdata: Any, ydata: Any, yerr: Any = None, xerr: Any = None) -> None:
        """Constructor of the class

        Parameters
        ----------
        xdata : Any
            x data points
        ydata : Any
            y data points
        yerr : Any, default None
            if there is, the uncertainties of `ydata` 
        xerr : Any, default None
            if there is, the uncertainties of `xdata` 
        """
        self.data = [xdata, ydata, yerr, xerr]
        self.fit_par: ndarray | None = None
        self.fit_err: ndarray | None = None
        self.res = {}


    def fit(self, method: Callable[[Any,Any],Any], initial_values: Sequence[Any], **kwargs) -> None:
        """To compute the fit

        Parameters
        ----------
        method : Callable[[Any,Any],Any]
            the fit function
        initial_values : Sequence[Any]
            initial values
        """
        # importing the function
        xdata, ydata = self.data[:2]
        sigma = self.data[2]
        Dx = self.data[3]
        self.res['func'] = method
        from scipy import odr
        def fit_model(pars, x):
            return method(x, *pars)
        model = odr.Model(fit_model)
        data = odr.RealData(xdata,ydata,sx=Dx,sy=sigma)
        alg = odr.ODR(data, model, beta0=initial_values)
        out = alg.run()
        pop = out.beta
        pcov = out.cov_beta
        Dpop = np.sqrt(pcov.diagonal())
        self.fit_par = pop
        self.fit_err = Dpop
        self.res['cov'] = pcov
        if sigma is not None or Dx is not None:
            chisq = out.sum_square
            chi0 = len(ydata) - len(pop)
            self.res['chisq'] = (chisq, chi0)
    
    def infos(self, names: Sequence[str] | None = None) -> None:
        """To plot information about the fit

        Parameters
        ----------
        names : list[str] | None, default None
            list of fit parameters names
        """
        pop  = self.fit_par
        Dpop = self.fit_err
        print('\nFit results:')
        if names is None:
            names = [f'par{i}' for i in range(len(pop))]
        for name, par, Dpar in zip(names,pop,Dpop):
            print(f'\t{name}: {par:.2} +- {Dpar:.2}  -->  {Dpar/par*100:.2} %')
        if 'chisq' in self.res:
            chisq, chi0 = self.res['chisq']
            print(f'\tred_chi = {chisq/chi0*100:.2f} +- {np.sqrt(2/chi0)*100:.2f} %')

    def results(self) -> tuple[ndarray, ndarray] | tuple[None, None]:
        return self.fit_par, self.fit_err

    def pipeline(self,method: Callable[[Any,Any],Any], initial_values: Sequence[Any], names: Sequence[str] | None = None,**kwargs) -> None:
        self.fit(method=method,initial_values=initial_values,**kwargs)
        self.infos(names=names)
    
    def gaussian_fit(self, initial_values: Sequence[float], names: Sequence[str] = ('k','mu','sigma'),**kwargs) -> None:
        """To fit with a Gaussian

        Parameters
        ----------
        intial_values : Sequence[Any]
            k, mu, sigma
        names : list[str] | None, optional
            names, by default None
        """
        def gauss_func(data: float | ndarray, *args) -> float | ndarray:
            k, mu, sigma = args
            z = (data - mu) / sigma
            return k * np.exp(-z**2/2)

        self.pipeline(method=gauss_func,initial_values=initial_values,names=names,**kwargs)

    def pol_fit(self, ord: int, initial_values: Sequence[float], names: Sequence[str] | None = None) -> None:
        def pol_func(x, *args):
            poly = [ args[i] * x**(ord - i) for i in range(len(args))]
            return np.sum(poly,axis=0)
        self.pipeline(pol_func,initial_values=initial_values,names=names)

    def linear_fit(self, initial_values: Sequence[float], names: Sequence[str] = ('m','q')) -> None:
        self.pol_fit(ord=1, initial_values=initial_values, names=names)

def make_cut_indicies(file_path: str, lines_num: int) -> ndarray:
    """To make a `cut_indicies.txt` file when it is not

    Parameters
    ----------
    file_path : str
        path of the chosen target directory
    lines_num : int
        number of acquisitions fot that target

    Returns
    -------
    cut : ndarray
        array of the edges of each acquisition 
    """
    cut = np.array([[0,-1,0,-1]]*lines_num,dtype=int)   #: array of the edges of each acquisition
    # compute the string to print in the file
    content = "#\tThe section of image to display\n#\n#\tThe first row is for the target acquisition\n#\tThe last one is for the lamp\n#\n#yl\tyu\txl\txu"
    for line in cut.astype(str):
        content = content + '\n' + '\t'.join(line)
    f = open(file_path, "w")
    f.write(content)
    f.close()
    return cut

def hotpx_remove(data: ndarray) -> ndarray:
    """To remove hot pixels from the image

    Parameters
    ----------
    data : ndarray
        spectrum data

    Returns
    -------
    data : ndarray
        spectrum data without `NaN` values
    
    Notes
    -----
    The function replacing `NaN` values from the image, if there are.
    I did not implement this function, I took it from [*astropy documentation*](https://docs.astropy.org/en/stable/convolution/index.html)

    """
    from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
    # check the presence of `NaNs`
    if True in np.isnan(data):
        # build a gaussian kernel for the interpolation
        kernel = Gaussian2DKernel(x_stddev=1)
        # remove the `NaNs`
        data = interpolate_replace_nans(data, kernel)
    return data