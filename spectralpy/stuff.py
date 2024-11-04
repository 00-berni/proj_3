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
from numpy.typing import ArrayLike
from typing import Callable, Sequence, Any, Literal
from astropy.io.fits import HDUList
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from astropy.units import Quantity

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
    def check_edges(lims: ArrayLike, shape: tuple[int, int]) -> ndarray:
        """To compute exact pixel value for cut image edges

        Parameters
        ----------
        lims : ArrayLike
            edges values
        shape : tuple[int, int]
            sizes of the image

        Returns
        -------
        lims : ndarray
            checked edges values
        """
        if lims is not None and 0 not in shape:
            lims = np.copy(lims)
            for i in [0,1]:
                print('Start',lims)
                n = shape[i]    #: size of image
                print('N =',n)
                k = 2*i+1       #: position of upper end 
                print('K =',k)
                if lims[k] is None: lims[k] = n
                elif lims[k] < 0: lims[k] = n + lims[k]
                print(lims)
        return lims

    @staticmethod
    def empty() -> 'Spectrum':
        """To generate an empty object

        Returns
        -------
        Spectrum
            empty object
        """
        return Spectrum([],[],[],[],False,hotpx=False,name='empty',check_edges=False)

    def __init__(self, hdul: HDUList | Sequence[HDUList] | None, data: ndarray | None, lims: ArrayLike | None, cut: ArrayLike | None, sigma: ndarray | None = None, hotpx: bool = True, name: str = '', check_edges: bool = True) -> None:
        """Constructor of the class

        Parameters
        ----------
        hdul : HDUList | Sequence[HDUList] | None
            hdul information
        data : ndarray | None
            spectrum values
        lims : ArrayLike | None
            edges of cut image
        cut : ArrayLike | None
            cut image edges for inclination correction
        hotpx : bool, optional
            to filter and remove hot pixels, by default `True`
        name : str, optional
            the name of the object, by default `''`
        check_edges: bool, optional
            parameter to check edges values, by default `True`
        """
        if check_edges:
            lims = Spectrum.check_edges(lims, shape=data.shape)
            cut  = Spectrum.check_edges(cut , shape=data.shape)
        self.name = name
        self.hdul = hdul.copy()
        self.data  = hotpx_remove(data) if hotpx else data 
        self.sigma = np.copy(sigma) if sigma is not None else None
        self.lims = lims
        self.cut  = cut 
        self.angle : None | tuple[float, float] = None
        self.sldata  : None | ndarray = None
        self.slsigma : None | ndarray = None
        self.spec  : None | ndarray = None
        self.std   : None | ndarray = None
        self.lines : None | ndarray = None
        self.errs  : None | ndarray = None
        self.func : None | tuple[Callable, Callable]  = None
        print(self.cut)
        print(self.lims)

    def print_header(self) -> None:
        """To print the header of fits file"""
        # take the header
        hdr = self.hdul[0].header
        print(' - HEADER -')
        print(hdr.tostring(sep='\n'))
        print()
    
    def format_ends(self) -> None:
        self.lims = Spectrum.check_edges(self.lims,self.data.shape)
        self.cut  = Spectrum.check_edges(self.cut ,self.data.shape)
    
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
        # self.format_ends()
        lims = (slice(*self.lims[:2]),slice(*self.lims[2:]))
        self.data = self.data[lims]
        if self.sigma is not None:
            self.sigma = self.sigma[lims]

    def rotate_target(self, angle: float = 0, **imagepar) -> 'Spectrum':
        if 'reshape' not in imagepar.keys():
            imagepar['reshape'] = False
        from scipy import ndimage
        target = self.copy()
        target.data = ndimage.rotate(target.data,angle=angle,**imagepar)
        if target.sigma is not None:
            target.sigma = ndimage.rotate(target.sigma, angle=angle, **imagepar)
        return target


    def angle_correction(self, angle: float | None = None, Dangle: float | None = None, lim_width: Sequence[int | Sequence[int]] | None = None, init: list[float] | None = None, lag: int = 10, gauss_corr: bool = True, fit_args: dict = {}, diagn_plots: bool = False, **pltargs) -> tuple['Spectrum',float]:
        """To correct the inclination of spectrum, rotating the image.
        
        Parameters
        ----------
        angle : float | None, optional
            inclination angle of the slit, by default `None`
            If `angle is None` then the value is estimated by a 
            fitting rountine
        init : list[float], optional
            initial values for the fit, by default `[0.9, 0.]`
        diagn_plots : bool, optional
            to plot figures, by default `False`

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
          2. Fit a column for every 50 points with a gaussian profile, estimate the mean 
          for each one and repeat the linear fit to estimate the angle
        After then the image is rotated (and its uncertainties too, if any)
        """
        target = self.copy()
        if angle is None:
            lims = target.cut       #: cut image edges to correct inclination 
            ylim = lims[:2]
            xlim = lims[2:]
            data = target.data[slice(*ylim), slice(*xlim)].copy()
            if not gauss_corr:
                if lim_width is None:
                    print('CUT',lims)
                    # cut image
                    plt.figure()
                    plt.imshow(data,cmap='gray_r',norm='log')
                    plt.show()
                    exit()
                xx = np.array([*lim_width[0]])
                yy = np.array([*lim_width[1]])
                m1 = np.diff(yy[0])/np.diff(xx)
                m2 = np.diff(yy[1])/np.diff(xx)
                print(m1,m2)
                print(yy)
                yc = np.mean(yy,axis=0)
                mean_width = abs(np.diff(yy,axis=0).mean()/4).astype(int)
                
                m12 = np.diff(yc)/np.diff(xx)

            ## Fit of Maxima 
            # compute maxima coordinates
            x_pos = np.arange(0,data.shape[1],lag)
            y_pos = np.argmax(data[:,x_pos], axis=0)
            if not gauss_corr:
                pos = np.where((y_pos <= x_pos*m12[0] + yc[0] - mean_width) | (y_pos >= x_pos*m12[0] + yc[0] + mean_width))[0]
                if diagn_plots:
                    plt.figure()
                    plt.imshow(data,cmap='gray_r',norm='log',origin='lower')
                    plt.plot(xx,xx*m1[0]+yy[0,0],color='violet')
                    plt.plot(xx,xx*m2[0]+yy[1,0],color='orange')
                    plt.plot(xx,xx*m12[0]+yc[0]+mean_width,'--',color='green')
                    plt.plot(xx,xx*m12[0]+yc[0]-mean_width,'--',color='green')
                    plt.plot(np.delete(x_pos,pos),np.delete(y_pos,pos),'.')
                    plt.show()
                print(mean_width)
                if len(pos) >= len(x_pos)-1: 
                    print('NO GOOD')
                    pos = []
                r_p = (x_pos[pos],y_pos[pos])
                x_pos = np.delete(x_pos,pos) 
                y_pos = np.delete(y_pos,pos) 

            Dx = 0
            Dy = 3

            if init is None: init = [np.mean([m1,m2,m12]),0.] if not gauss_corr else [0.9,ylim[0]]
            fit = FuncFit(xdata=x_pos, ydata=y_pos, yerr=Dy, xerr=Dx)
            fit.linear_fit(init,mode='curve_fit',absolute_sigma=True)
            pop, perr = fit.results()
            m = pop[0]
            Dm = perr[0]
            # compute the angle in the degrees from the angular coefficient
            angle1 = np.arctan(m)*180/np.pi  
            Dangle1 = 180/np.pi * Dm/(1+m**2)

            ## Fit of Gaussians
            if gauss_corr:
                from scipy import ndimage
                # correct data
                data = ndimage.rotate(data, angle1, reshape=False).copy()
                print('\nGAUSSIAN CORRECTION')
                if diagn_plots: fig, ax = plt.subplots(1,1)
                # prepare data 
                x_pos = x_pos#[::50]
                y_pos, Dy = np.array([]), np.array([])
                # fit colums with a gaussian
                for i in x_pos:
                    values = data[:,i]
                    Dvalues = target.sigma[:,i].copy() if target.sigma is not None else None
                    y = np.arange(len(values))
                    # estimate HWHM for the initial value of the sigma
                    hwhm = abs(np.argmax(values) - np.argmin(abs(values - max(values)/2)))
                    initial_values = [max(values),np.argmax(values),hwhm]
                    # take only values greater than the double of the mean
                    pos = np.where(values > np.mean(values)*2)
                    if len(pos[0]) <= 3: pos = np.where(values > np.mean(values))
                    values = values[pos]
                    y = y[pos]
                    # compute the fit
                    fit = FuncFit(xdata=y, ydata=values,xerr=0.5,yerr=Dvalues)
                    fit.gaussian_fit(initial_values,**fit_args)
                    pop, perr = fit.results()
                    # store results
                    y_pos = np.append(y_pos,pop[1])
                    Dy = np.append(Dy,pop[2])
                    method = fit.res['func']
                    color = (1-i/max(x_pos),i/max(x_pos),i/(2*max(x_pos))+0.5)
                    if diagn_plots:
                        ax.plot(data[:,i],color=color,label='fit',**pltargs)
                        ax.plot(y, method(y,*pop), '--',color=color,**pltargs)
                if diagn_plots: ax.legend()            
                print(len(x_pos),len(y_pos))
                # compute the fit to get inclination angle
                fit = FuncFit(xdata=x_pos, ydata=y_pos, yerr=Dy)
                fit.linear_fit(init,**fit_args)
                pop, perr = fit.results()
                m  = pop[0]
                Dm = perr[0]
                # compute the angle in the degrees from the angular coefficient
                angle2 = np.arctan(m)*180/np.pi   
                Dangle2 = 180/np.pi * Dm/(1+m**2)
                # compute the total angle
                angle  = angle1 + angle2
                Dangle = np.sqrt(Dangle1**2 + Dangle2**2)
            else: 
                angle  = angle1
                Dangle = Dangle1
                if diagn_plots:
                    plt.figure()
                    plt.imshow(data,cmap='gray_r',origin='lower')
                    plt.plot(x_pos, y_pos,'.')
                    plt.plot(r_p[0], r_p[1],'.',color='red')
                    fit.plot(points_num=3)
                    plt.show()
            fmt = unc_format(angle,Dangle)
            str_res ='Inclination Angle : {angle:' + fmt[0][1:] + '} +/- {Dangle:' + fmt[1][1:] + '} deg  -->  ' + f'{Dangle/angle*100:.2f} %'
            print(str_res.format(angle=angle,Dangle=Dangle))
        # rotate the image
        target = target.rotate_target(angle)
        target.angle = (angle,Dangle)
        print(target.angle)
        return target, angle
    
    def compute_lines(self, shift: float = 0, Dx: ArrayLike = 1/np.sqrt(12)) -> None:
        """To compute and store calibrated values in Armstrong 
        and their uncertainties 

        Parameters
        ----------
        shift : float, optional
            lag between two different lamps obtained from the
            cross-correlation of them, by default `0`
        """
        pxs = np.arange(self.lims[2],self.lims[3]) + shift
        cal_func, err_func = self.func
        self.lines = cal_func(pxs)
        self.errs  = err_func(pxs,Dx)

    def binning(self, bin: ArrayLike = 50, edges: None | Sequence[float] = None) -> tuple[tuple[ndarray,ndarray], tuple[ndarray,ndarray], ndarray]:
        """To bin spectrum data

        Parameters
        ----------
        bin : ArrayLike, optional
            The width of each bin if a `float` or a `int` is passed, by default `50`
            It is possible to pass instead the array of specific values of the bins
        edges : None | Sequence[float], optional
            the ends of the wavelengths range to bin, by default `None` 
            If `edges is None` then the extremes of `self.lines` array are taken
            Values are approximated to the nearest multiple of the bin width
            If `bin` is an array then this parameter is ignored
                          

        Returns
        -------
        (bin_lines, err_lines) : tuple[ndarray, ndarray]
            central values of bins and their uncertainties
            Each value has an uncertainty equal to `bin / 2`
        (bin_spect, err_spect) : tuple[ndarray, ndarray]
            binned spectrum and the uncertainty
            Each value is the average over the values in the bin
            and the relative uncertainty is the STD
        bins : ndarray
            bins values
        """
        # store data
        spectrum = self.spec.copy()
        lines = self.lines.copy()
        if isinstance(bin,(float,int)):     #: `bin` is the bin width
            bin_width = bin
            # compute and approximate the ends of wavelengths range
            if edges is None: edges = (lines[0], lines[-1])
            appr = lambda l : np.rint(l / bin_width) * bin_width
            edges = (appr(edges[0]), appr(edges[1])+1)
            if isinstance(bin_width,int):
                bins = np.arange(*edges,bin_width) 
            else:
                num = int((edges[1] - edges[0]) // bin_width + 1)
                bins = np.linspace(*edges,num)
        else:
            bins = np.copy(bin)
            bin_width = np.diff(bin).astype(int)[0]
        # define some useful quantities
        half_bin = bin_width / 2
        bin_num = len(bins) - 2 
        # define array of the central value in each bin
        bin_lines = bins[:-1] + half_bin
        # set the every uncertainties to the half width
        err_lines = np.full(bin_lines.shape, bin_width / 2)
        # average over the values in each bin
        pos = lambda i : np.where((bin_lines[i] - half_bin <= lines) & (lines < bin_lines[i] + half_bin))[0]
        bin_spect, err_spect = np.array([ [*mean_n_std(spectrum[pos(i)])] for i in range(bin_num+1)]).transpose()
        return (bin_lines, err_lines), (bin_spect, err_spect), bins

    def spectral_data(self, plot_format: bool = False) -> ndarray[float,ndarray]:
        """To get spectral data

        Parameters
        ----------
        plot_format : bool, optional
            if `True` data are returned to be passed directly to 
            `matplotlib.pyplot.errorbar()` function as
            `[xdata, ydata, yerr, xerr]`, by default `False`
        
        Returns
        -------
        ndarray
            an array that collects wavelengths with uncentanties and
            the corresponding counts with their uncentanties
            If `plot_format == True` data are returned like
            `[xdata, ydata, yerr, xerr]`
        """
        if plot_format:
            spectral_data = np.array([self.lines, self.spec, self.std, self.errs])
        else:
            spectral_data = np.array([self.lines, self.errs, self.spec, self.std])
        return spectral_data

    def copy(self) -> 'Spectrum':
        """To make an identical copy of a Spectrum object

        Returns
        -------
        target : Spectrum
            the copy
        """
        target = Spectrum([*self.hdul], self.data.copy(), np.copy(self.lims), cut=np.copy(self.cut), sigma=self.sigma, hotpx=False, name=self.name,check_edges=False)
        target.spec  = self.spec.copy()  if self.spec  is not None else None
        target.lines = self.lines.copy() if self.lines is not None else None
        target.errs  = self.errs.copy()  if self.errs  is not None else None
        target.func  = [*self.func]      if self.func  is not None else None
        return target

    def __add__(self, spec: Any) -> ndarray:
        if isinstance(spec, Spectrum):
            return self.data + spec.data
        else:
            return self.data + spec

    def __radd__(self, other) -> ndarray:
            return self.__add__(other)

    def __sub__(self, spec: Any) -> ndarray:
        if isinstance(spec, Spectrum):
            return self.data - spec.data
        else:
            return self.data - spec

    def __rsub__(self, other) -> ndarray:
            return -self.__sub__(other)

    def __mul__(self, spec: Any) -> ndarray:
        if isinstance(spec, Spectrum):
            return self.data * spec.data
        else:
            return self.data * spec
    
    def __getitem__(self, index: int | Sequence[int | slice] | slice) -> ndarray:
        return self.data[index]

    # def __setitem__(self, index: int | Sequence[int | slice] | slice, value: ArrayLike) -> None:
    #     self.data[index] = value


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
    def __init__(self, xdata: ArrayLike, ydata: ArrayLike, yerr: ArrayLike | None = None, xerr: ArrayLike | None = None) -> None:
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
        if np.all(xerr == 0): xerr = None
        elif isinstance(xerr,(int,float)): xerr = np.full(xdata.shape,xerr)
        if np.all(yerr == 0): yerr = None
        elif isinstance(yerr,(int,float)): yerr = np.full(ydata.shape,yerr)
        xdata = np.copy(xdata)
        ydata = np.copy(ydata)
        xerr  = np.copy(xerr) if xerr is not None else None
        yerr  = np.copy(yerr) if yerr is not None else None
        self.data = [xdata, ydata, yerr, xerr]
        self.fit_par: ndarray | None = None
        self.fit_err: ndarray | None = None
        self.res = {}

    def odr_routine(self, **odrargs) -> None:
        xdata, ydata, yerr, xerr = self.data
        beta0 = self.res['init']
        from scipy import odr
        method = self.res['func']
        def fit_model(pars, x):
            return method(x, *pars)
        model = odr.Model(fit_model)
        data = odr.RealData(xdata,ydata,sx=xerr,sy=yerr)
        alg = odr.ODR(data, model, beta0=beta0,**odrargs)
        out = alg.run()
        pop = out.beta
        pcov = out.cov_beta
        Dpop = np.sqrt(pcov.diagonal())
        self.fit_par = pop
        self.fit_err = Dpop
        self.res['cov'] = pcov
        if yerr is not None or xerr is not None:
            chisq = out.sum_square
            chi0 = len(ydata) - len(pop)
            self.res['chisq'] = (chisq, chi0)

    def chi_routine(self, err_func: Callable[[ArrayLike,ArrayLike,ArrayLike], ArrayLike] | None = None, iter: int = 3, **chiargs) -> None:
        xdata, ydata, yerr, xerr = self.data
        initial_values = self.res['init']
        method = self.res['func']
        from scipy.optimize import curve_fit
        if yerr is None: 
            chiargs['absolute_sigma'] = False
        pop, pcov = curve_fit(method, xdata, ydata, initial_values, sigma=yerr, **chiargs)
        sigma = yerr
        print('XERR',xerr)
        if xerr is not None:
            if yerr is None: yerr = 0
            initial_values = [*pop]
            sigma = np.sqrt(yerr**2 + err_func(xdata,xerr,pop))
            for _ in range(iter):
                pop, pcov = curve_fit(method, xdata, ydata, initial_values, sigma=sigma, **chiargs)
                initial_values = [*pop]
                sigma = np.sqrt(yerr**2 + err_func(xdata,xerr,pop))
        Dpop = np.sqrt(pcov.diagonal())
        self.fit_par = pop
        self.fit_err = Dpop
        self.res['cov'] = pcov
        if sigma is not None:
            chisq = np.sum(((ydata - method(xdata,*pop)) / sigma)**2)
            chi0 = len(ydata) - len(pop)
            self.res['chisq'] = (chisq, chi0)        
            

    def fit(self, method: Callable[[Any,Any],Any], initial_values: Sequence[Any], mode: Literal['odr','curve_fit'] = 'odr', **fitargs) -> None:
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
        if len(xdata) != len(ydata): return IndexError(f'different arrays length:\nxdata : {len(xdata)}\nydata : {len(ydata)}')
        self.res['func'] = method
        self.res['init'] = np.copy(initial_values)
        # if Dx is None: mode = 'curve_fit'
        self.res['mode'] = mode
        if mode == 'curve_fit':
            self.chi_routine(**fitargs)
        elif mode == 'odr':
            self.odr_routine(**fitargs)
        else: return ValueError(f'mode = `{mode}` is not accepted')


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
        print('\tmode : '+self.res['mode'])
        initial_values = self.res['init']        
        if names is None:
            names = [f'par{i}' for i in range(len(pop))]
        for name, par, Dpar, init in zip(names,pop,Dpop,initial_values):
            fmt_m, fmt_u = unc_format(par,Dpar)
            fmt_measure = '\t{name}: {par:' + fmt_m[1:] + '} +/- {Dpar:' + fmt_u[1:] + '}  -->  {relerr:.2f} %\tinit : {init:.2}'
            # if Dpar != 0. : fmt_measure = fmt_measure + 
            try:
                info_str = fmt_measure.format(name=name, par=par, Dpar=Dpar,relerr=abs(Dpar/par)*100,init=init)
            except ValueError:
                info_str = f'\t{name}: {par} +/- {Dpar}  -->  {abs(Dpar/par)*100.0:.2f} %\tinit : {init*1.0:.2}'
            print(info_str)
        cov = self.res['cov']
        corr = np.array([ cov[i,j]/np.sqrt(cov[i,i]*cov[j,j]) for i in range(cov.shape[0]) for j in range(i+1,cov.shape[1])])
        names = np.array([ names[i] + '-' + names[j] for i in range(cov.shape[0]) for j in range(i+1,cov.shape[1])])
        for c, name in zip(corr,names):
            print(f'\tcorr_{name}\t = {c:.2}')
        if 'chisq' in self.res:
            chisq, chi0 = self.res['chisq']
            Dchi0 = np.sqrt(2*chi0)
            res = '"OK"' if abs(chisq-chi0) <= Dchi0 else '"NO"'
            if chi0 == 0:
                print('! ERROR !')
                print('\t',chisq,chi0)
                raise ValueError('Null degrees of freedom. Overfitting!')
            print(f'\tchi_sq = {chisq:.2f}  -->  red = {chisq/chi0*100:.2f} %')
            print(f'\tchi0 = {chi0:.2f} +/- {Dchi0:.2f}  -->  '+res)

    def results(self) -> tuple[ndarray, ndarray] | tuple[None, None]:
        return np.copy(self.fit_par), np.copy(self.fit_err)
    
    def method(self, xdata: ArrayLike) -> ArrayLike:
        func = self.res['func']
        pop = np.copy(self.fit_par)
        return func(xdata,*pop)

    def fvalues(self) -> ArrayLike:
        return self.method(self.data[0])
    
    def residuals(self) -> ArrayLike:
        return self.data[1] - self.fvalues()

    def pipeline(self,method: Callable[[Any,Any],Any], initial_values: Sequence[Any], names: Sequence[str] | None = None, mode: Literal['odr','curve_fit'] = 'odr',**fitargs) -> None:
        self.fit(method=method,initial_values=initial_values,mode=mode,**fitargs)
        self.infos(names=names)
    
    def gaussian_fit(self, initial_values: Sequence[float], names: Sequence[str] = ('k','mu','sigma'),mode: Literal['odr','curve_fit'] = 'odr',**fitargs) -> None:
        """To fit with a Gaussian

        Parameters
        ----------
        intial_values : Sequence[Any]
            k, mu, sigma
        names : list[str] | None, optional
            names, by default None
        """
        def gauss_func(data: ArrayLike, *args) -> ArrayLike:
            k, mu, sigma = args
            z = (data - mu) / sigma
            return k * np.exp(-z**2/2)
        
        def err_der(x: ArrayLike, Dx: ArrayLike, par: ArrayLike) -> ArrayLike:
            coeff = gauss_func(x,*par) 
            return np.abs(coeff * (x-par[1]) / par[2]**2 * Dx)

        if mode == 'curve_fit': 
            fitargs['err_func'] = err_der
        self.pipeline(method=gauss_func,initial_values=initial_values,names=names,mode=mode,**fitargs)
        
        def err_func(xdata: ArrayLike, Dx: ArrayLike | None, par: ArrayLike, cov: ArrayLike) -> ArrayLike:
            coeff = gauss_func(xdata,*par) 
            der = np.array([np.ones(xdata.shape)/par[0],
                            (xdata-par[1])/par[2]**2,
                            (xdata-par[1])**2/par[2]**3])
            der *= coeff
            err = np.sum([ der[i]*der[j] * cov[i,j] for i in range(3) for j in range(3)],axis=0)
            if Dx is not None:
                err += ( err_der(xdata, Dx, par))**2
            err = np.sqrt(err)
            return err
        error_function = lambda x, Dx : err_func(x,Dx,self.fit_par,self.res['cov'])
        self.res['errfunc'] = error_function

    def voigt_fit(self, initial_values: Sequence[float], names: Sequence[str] = ['sigma','gamma','k','median'], mode: Literal['odr','curve_fit'] = 'odr',**fitargs):
        # def voigt_func(data: ArrayLike, *args) -> ArrayLike:
        #     sigma, gamma, k, mu_g, mu_l = args
        #     gauss = np.exp(-((data-mu_g)/sigma)**2/2) / np.sqrt(2*np.pi*sigma**2)
        #     loren = gamma / ((data-mu_l)**2 + gamma**2) / np.pi
        #     voigt = np.convolve(gauss,loren,mode='same')
        #     return k * voigt / voigt.mean()
        def voigt_func(data: ArrayLike, *args) -> ArrayLike:
            from scipy.special import voigt_profile
            sigma, gamma, k, mu = args
            return k*voigt_profile(data-mu,sigma,gamma)
        
        self.pipeline(method=voigt_func,initial_values=initial_values,names=names,mode=mode, **fitargs)


    def pol_fit(self, ord: int, initial_values: Sequence[float], names: Sequence[str] | None = None, mode: Literal['odr','curve_fit'] = 'odr',**fitargs) -> None:
        def pol_func(x: ArrayLike, *args) -> ArrayLike:
            poly = [ args[i] * x**(ord - i) for i in range(ord+1)]
            return np.sum(poly,axis=0)
        
        def err_der(x: ArrayLike, Dx: ArrayLike, par: ArrayLike) -> ArrayLike:
            return np.abs(np.sum([ par[i] * (ord-i) * x**(ord-i-1) for i in range(ord)],axis=0) * Dx)

        if mode == 'curve_fit': 
            fitargs['err_func'] = err_der
        self.pipeline(pol_func,initial_values=initial_values,names=names, mode=mode, **fitargs)
        
        def err_func(xdata: ArrayLike, Dx: ArrayLike | None, par: ArrayLike, cov: ArrayLike) -> ArrayLike:
            der = lambda i : xdata**(ord-i)
            err = np.sum([ der(i)*der(j) * cov[i,j] for i in range(ord+1) for j in range(ord+1)],axis=0)
            if Dx is not None:
                err += (err_der(xdata,Dx,par))**2
            print(err[err<0])
            if len(np.where(err<0)[0]) != 0: exit()
            err = np.sqrt(err)
            return err
        error_function = lambda x, Dx : err_func(x, Dx, self.fit_par, self.res['cov'])
        self.res['errfunc'] = error_function

    def linear_fit(self, initial_values: Sequence[float], names: Sequence[str] = ('m','q'), mode: Literal['odr','curve_fit'] = 'odr',**fitargs) -> None:
        self.pol_fit(ord=1, initial_values=initial_values, names=names, mode=mode, **fitargs)
    
    def sigma(self) -> ArrayLike:
        err_func = self.res['errfunc']
        xdata, _, Dy, Dx = self.data
        err = err_func(xdata,Dx)
        if Dy is not None:
            err = np.sqrt(Dy**2 + err**2)
        return err

    def data_plot(self, ax: Axes, points_num: int = 200, grid: bool = True, pltarg1: dict = {}, pltarg2: dict = {}) -> None:
        xdata = self.data[0]
        xx = np.linspace(xdata.min(),xdata.max(),points_num)
        if 'fmt' not in pltarg1.keys():
            pltarg1['fmt'] = '.'        
        ax.errorbar(*self.data,**pltarg1)
        ax.plot(xx,self.method(xx),**pltarg2)
        if grid: ax.grid(color='lightgray', ls='dashed')

    def residuals_plot(self, ax: Axes, grid: bool = True, **pltargs) -> None:
        xdata = self.data[0]
        if 'fmt' not in pltargs.keys():
            pltargs['fmt'] = '.'
        ax.errorbar(xdata,self.residuals(),self.sigma(),**pltargs)
        ax.axhline(0,0,1,color='black')
        if grid: ax.grid(color='lightgray', ls='dashed')

    def plot(self, sel: Literal['data','residuals','all'] = 'all', mode: Literal['plots', 'subplots'] = 'plots', points_num: int = 200, fig: None | Figure | tuple[Figure,Figure] = None, grid: bool = True, plot1: dict = {}, plot2: dict = {}, **resarg) -> None:
        if fig is None:
            if sel in ['data','all']: 
                fig = plt.figure()
            if sel in ['residuals', 'all'] and mode != 'subplots': 
                fig2 = plt.figure()
        elif isinstance(fig,tuple):
            fig, fig2 = fig
        if sel == 'all' and mode == 'subplots':
            ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))        
            self.data_plot(ax=ax1,points_num=points_num,grid=grid,pltarg1=plot1,pltarg2=plot2)
            self.residuals_plot(ax=ax2,**resarg)
        else:
            if sel in ['data','all']:
                ax = fig.add_subplot(111)
                self.data_plot(ax=ax, points_num=points_num,grid=grid,pltarg1=plot1,pltarg2=plot2)
            if sel in ['residuals', 'all']:
                ax2 = fig2.add_subplot(111)
                self.residuals_plot(ax=ax2,**resarg)

def mean_n_std(data: ArrayLike, axis: int | None = None, weights: Sequence[Any] | None = None) -> tuple[float, float]:
    """To compute the mean and standard deviation from it

    Parameters
    ----------
    data : Sequence[Any]
        values of the sample
    axis : int | None, default None
        axis over which averaging
    weights : Sequence[Any] | None, default None
        array of weights associated with data

    Returns
    -------
    mean : float
        the mean of the data
    std : float
        the STD from the mean
    """
    dim = len(data)     #: size of the sample
    # data = np.array(data)
    # compute the mean
    mean = np.average(data,axis=axis,weights=weights)
    # compute the STD from it
    if weights is None:
        std = np.sqrt( ((data-mean)**2).sum(axis=axis) / (dim*(dim-1)) )
    else:
        std = np.sqrt(np.average((data-mean)**2, axis=axis, weights=weights) / (dim-1) * dim)
    return mean, std

def compute_err(*args: Spectrum) -> ndarray | None:
    sigma = 0
    for elem in args:
        elem = elem.copy()
        if elem.sigma is None: elem.sigma = 0
        sigma += elem.sigma**2
    sigma = np.sqrt(sigma)
    if isinstance(sigma,int) and sigma == 0: sigma = None
    return sigma

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
    cut = np.array([[0,-1,0,-1]*2]*lines_num,dtype=int)   #: array of the edges of each acquisition
    # compute the string to print in the file
    content = "#\tThe section of image to display\n#\n#\tThe first row is for the target acquisition\n#\tThe last one is for the lamp\n#\n#\tThe first four columns are the edges of the\n#\tinclinated image\n#\tThe others are the edges of the corrected image\n#\n#cyl\tcyu\tcxl\tcxu\tyl\tyu\txl\txu"
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

def magnitude_order(number: ArrayLike) -> ArrayLike:
    number = np.abs(number) 
    order = np.floor(np.log10(number)).astype(int)
    return order

def unc_format(value: ArrayLike, err: ArrayLike) -> list[str]:
    err_ord = magnitude_order(err).min()
    val_ord = magnitude_order(value).max()
    order = val_ord - err_ord + 1
    fmt = [f'%.{order:d}e',r'%.1e']
    return fmt
    
def print_measure(value: float | Quantity, err: float | Quantity, name: str = 'value', unit: str | None = None) -> None:
    from spectralpy.stuff import unc_format
    if isinstance(value, Quantity) and isinstance(err, Quantity):
        unit = value.unit.to_string()
        value = value.value
        err = err.value
    fmt = unc_format(value,err)
    if value != 0:
        fmt = name + ' = {value:' + fmt[0][1:] + '} +/- {err:' + fmt[1][1:] + '} ' + unit + ' ---> {perc:.2%}'
        print(fmt.format(value=value,err=err,perc=err/value))
    else:
        fmt = name + ' = {value:' + fmt[0][1:] + '} +/- {err:' + fmt[1][1:] + '} ' + unit 
        print(fmt.format(value=value,err=err))


def argmax(arr: ndarray, out: Literal['ndarray','tuple'] = 'tuple' , **maxkw) -> ndarray:
    maxpos = np.unravel_index(np.argmax(arr, **maxkw), arr.shape)
    if out == 'tuple':
        return maxpos
    elif out == 'ndarray':
        return np.array([*maxpos])

def argmin(arr: ndarray, out: Literal['ndarray','tuple'] = 'tuple', **minkw) -> tuple[ndarray,ndarray] | ndarray:
    minpos = np.unravel_index(np.argmin(arr, **minkw), arr.shape)
    if out == 'tuple':
        return minpos
    elif out == 'ndarray':
        return np.array([*minpos])
