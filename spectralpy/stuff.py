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
from astropy.io.fits import HDUList, PrimaryHDU, Header
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from astropy.units import Quantity
from statistics import pvariance

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

    @staticmethod
    def get_header(hdul: HDUList | PrimaryHDU | list[HDUList]) -> Header | list[Header]:
        if hdul == []:
            header = []
        elif isinstance(hdul,PrimaryHDU):
            header = hdul.header
        elif isinstance(hdul,HDUList):
            header = hdul[0].header
        elif isinstance(hdul,list):
            header = []
            for hh in hdul:
                if isinstance(hh, HDUList): 
                    header += [hh[0].header]
                elif isinstance(hh,PrimaryHDU):
                    header += [hh.header]
            if len(header)==1: header = header[0]
        else:
            print(hdul)
            raise TypeError(f'Wrong type {type(hdul)}')
        return header

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
        self.header = Spectrum.get_header(hdul)
        self.data  = hotpx_remove(data) if hotpx else data 
        self.sigma = np.copy(sigma) if sigma is not None else None
        self.lims = lims
        self.cut  = cut 
        self.angle : None | tuple[float, float] = None
        self.sldata  : None | ndarray = None
        self.slsigma : None | ndarray = None
        self.cen  : None | int = None
        self.span : None | int = None
        self.spec  : None | ndarray = None
        self.std   : None | ndarray = None
        self.lines : None | ndarray = None
        self.errs  : None | ndarray = None
        self.func : None | tuple[Callable, Callable]  = None

    def print_header(self) -> None:
        """To print the header of fits file"""
        header = self.header
        if isinstance(header,Header):
            header = [header]
        for h in header:
            print(' - HEADER -')
            print(h.tostring(sep='\n'))
            print()

    def update_header(self) -> None:
        self.header = Spectrum.get_header(self.hdul)

    
    def format_ends(self) -> None:
        self.lims = Spectrum.check_edges(self.lims,self.data.shape)
        self.cut  = Spectrum.check_edges(self.cut ,self.data.shape)

    def get_exposure(self, key: Literal['EXPTIME','EXPOSURE'] = 'EXPTIME') -> float:
        """To get the exposure time

        Returns
        -------
        exp_time : float
            exposure time
            For different acquisitions the mean is computed
        """
        header = self.header
        if isinstance(header, list):
            for hd in header:
                print(self.name + ':\t',hd[key])
            exp_time = np.mean([hd[key] for hd in header])
        else:
            exp_time = header[key]
        return exp_time
    
    def cut_image(self) -> None:
        """To cut the image"""
        # self.format_ends()
        lims = (slice(*self.lims[:2]),slice(*self.lims[2:]))
        self.data = self.data[lims]
        if self.sigma is not None:
            self.sigma = self.sigma[lims]

    def nan_remove(self) -> None:
        self.data = hotpx_remove(self.data)

    def rotate_target(self, angle: float = 0, **imagepar) -> 'Spectrum':
        if 'reshape' not in imagepar.keys():
            imagepar['reshape'] = False
        if 'order' not in imagepar.keys():
            imagepar['order'] = 1
        from scipy import ndimage
        target = self.copy()
        target.data = ndimage.rotate(target.data,angle=angle,**imagepar)
        target.angle = np.array([angle,0]) if target.angle is None else target.angle + angle
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
        if target.sigma is not None and np.any(target.sigma<0):
            plt.figure()
            plt.title('Before inclination correction')
            plt.imshow(target.sigma)
            plt.plot(*np.where(target.sigma<0)[::-1],'.',color='red')
            plt.colorbar()
            plt.show()
            exit()
        if angle is None:
            lims = target.cut       #: cut image edges to correct inclination 
            ylim = lims[:2]
            xlim = lims[2:]
            data  = target.data[slice(*ylim), slice(*xlim)].copy()
            Ddata = target.sigma[slice(*ylim), slice(*xlim)].copy() if target.sigma is not None else None 
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
            hwhm = []
            for x,y in zip(x_pos,y_pos):
                hm = data[y,x]/2
                up_data = data[y+1:,x].copy()
                down_data = data[:y,x].copy()
                up_hwhm = np.argmin(abs(up_data-hm))/2
                down_hwhm = (y-np.argmin(abs(down_data-hm)))/2
                m_hwhm = (up_hwhm+down_hwhm)/2
                hwhm += [m_hwhm]
            Dx = 0
            Dy = hwhm
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
                Dy = np.delete(Dy,pos) 

            if diagn_plots:
                plt.figure()
                plt.imshow(data,cmap='gray_r')
                plt.errorbar(x_pos,y_pos,Dy,fmt='.')
                plt.show()

            if init is None: init = [np.mean([m1,m2,m12]),0.] if not gauss_corr else [0.9,ylim[0]]
            fit = FuncFit(xdata=x_pos, ydata=y_pos, yerr=Dy, xerr=Dx)
            fit.linear_fit(init,mode='curve_fit',absolute_sigma=True)
            pop, perr = fit.results()
            m = pop[0]
            Dm = perr[0]
            # compute the angle in the degrees from the angular coefficient
            angle1 = np.arctan(m)*180/np.pi  
            Dangle1 = 180/np.pi * Dm/(1+m**2)
            if diagn_plots:
                fit.plot(mode='subplots')
            fmt = unc_format(angle1,Dangle1)
            str_res ='First Inclination Angle : {angle:' + fmt[0][1:] + '} +/- {Dangle:' + fmt[1][1:] + '} deg  -->  ' + f'{Dangle1/angle1*100:.2f} %'
            print(str_res.format(angle=angle1,Dangle=Dangle1))

            ## Fit of Gaussians
            if gauss_corr:
                if diagn_plots:
                    fig0, axes = plt.subplots(3,1,sharex=True)
                    fig0.suptitle(self.name+': inclination correction',fontsize=20)
                    axes[0].set_title('The brightest pixels',fontsize=20)
                    axes[0].imshow(data,cmap='gray_r')
                    axes[0].errorbar(x_pos,y_pos,Dy,fmt='.',label='Brightest values',color='green',capsize=3)
                    axes[0].legend()
                    axes[0].set_ylabel('y [px]',fontsize=18)

                from scipy import ndimage
                # correct data
                # data = ndimage.rotate(data, angle1, reshape=False).copy()
                # xlim[1] -= 200
                # ylim[1] += 5
                data = ndimage.rotate(target.data, angle1, reshape=False,order=1).copy()[slice(*ylim), slice(*xlim)]
                Ddata = ndimage.rotate(target.sigma, angle1, reshape=False,order=1).copy()[slice(*ylim), slice(*xlim)] if target.sigma is not None else None
                if Ddata is not None and np.any(Ddata<0):
                    plt.figure()
                    plt.title('After first rotation')
                    plt.imshow(Ddata)
                    plt.plot(*np.where(Ddata<0)[::-1],'.',color='red')
                    plt.colorbar()
                    plt.show()
                    exit()
                # plt.figure()
                # plt.imshow(data,norm='log')
                x_pos = x_pos[x_pos<=data.shape[1]]
                # for x in x_pos:
                #     plt.axvline(x,0,1,color='blue')
                # plt.show()
                # exit()
                print('\nGAUSSIAN CORRECTION')
                if diagn_plots: 
                    fig, ax = plt.subplots(1,1)
                    fig2, ax2 = plt.subplots(1,1)
                # prepare data 
                # x_pos = x_pos[::10]
                y_pos, Dy = np.array([]), np.array([])
                stop_val = x_pos[-1]+1
                # fit colums with a gaussian
                for i in x_pos:
                    values = data[:,i]
                    Dvalues = Ddata[:,i].copy() if Ddata is not None else None
                    if Dvalues is not None and np.any(Dvalues<0):
                        plt.figure()
                        plt.title('Gaussian inclination')
                        plt.imshow(Ddata)
                        plt.plot(*np.where(Ddata<0)[::-1],'.',color='red')
                        plt.colorbar()
                        plt.show()
                        exit()
                    y = np.arange(len(values))
                    # estimate HWHM for the initial value of the sigma
                    hwhm = abs(np.argmax(values) - np.argmin(abs(values - max(values)/2)))
                    initial_values = [max(values),np.argmax(values),hwhm]
                    # take only values greater than the double of the mean
                    pos = np.where(values > np.mean(values)*2)
                    if len(pos[0]) <= 3: pos = np.where(values > np.mean(values))
                    # values = values[pos]
                    # Dvalues = Dvalues[pos]
                    # y = y[pos]
                    # plt.figure()
                    # plt.errorbar(y,values,Dvalues,fmt='.',linestyle='dashed')
                    # plt.show()
                    # compute the fit
                    fit = FuncFit(xdata=y, ydata=values,xerr=0.5,yerr=Dvalues)
                    fit.gaussian_fit(initial_values,**fit_args)
                    pop, perr = fit.results()
                    if pop[2] < 0:
                        stop_val = i
                        break
                    # store results
                    y_pos = np.append(y_pos,pop[1])
                    Dy = np.append(Dy,pop[2])
                    method = fit.res['func']
                    color = (1-i/max(x_pos),i/max(x_pos),i/(2*max(x_pos))+0.5)
                    if diagn_plots:
                        fit.data_plot(ax)
                        ax2.plot(data[:,i],color=color,label='fit',**pltargs)
                        ax2.plot(y, method(y,*pop), '--',color=color,**pltargs)
                x_pos = x_pos[x_pos<stop_val]
                if diagn_plots: 
                    ax.legend()
                    axes[1].set_title('Gaussian means',fontsize=20)
                    axes[1].imshow(data,cmap='gray_r')
                    axes[1].errorbar(x_pos,y_pos,Dy,fmt='.',label='Gaussian means',color='green',capsize=3)
                    axes[1].legend()
                    axes[1].set_ylabel('y [px]',fontsize=18)
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
                if diagn_plots:
                    data = ndimage.rotate(target.data, angle, reshape=False,order=1).copy()[slice(*ylim), slice(*xlim)]
                    axes[2].set_title('Rotated image',fontsize=20)
                    axes[2].imshow(data,cmap='gray_r')
                    axes[2].set_ylabel('y [px]',fontsize=18)
                    axes[2].set_xlabel('x [px]',fontsize=18)
                    plt.show()
            else: 
                angle  = angle1
                Dangle = Dangle1
                if diagn_plots:
                    # fig,ax = plt.subplots(1,1)
                    # fit.data_plot(ax)
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
        (bin_lines, err_lines), (bin_spect, err_spect), bins = binning(spectrum,lines,bin,edges)
        # if isinstance(bin,(float,int)):     #: `bin` is the bin width
        #     bin_width = bin
        #     # compute and approximate the ends of wavelengths range
        #     if edges is None: edges = (lines[0], lines[-1])
        #     appr = lambda l : np.rint(l / bin_width) * bin_width
        #     edges = (appr(edges[0]), appr(edges[1])+1)
        #     if isinstance(bin_width,int):
        #         bins = np.arange(*edges,bin_width) 
        #     else:
        #         num = int((edges[1] - edges[0]) // bin_width + 1)
        #         bins = np.linspace(*edges,num)
        # else:
        #     bins = np.copy(bin)
        #     bin_width = np.diff(bin).astype(int)[0]
        # # define some useful quantities
        # half_bin = bin_width / 2
        # bin_num = len(bins) - 2 
        # # define array of the central value in each bin
        # bin_lines = bins[:-1] + half_bin
        # # set the all uncertainties to the half width
        # err_lines = np.full(bin_lines.shape, bin_width / 2)
        # # average over the values in each bin
        # pos = lambda i : np.where((bins[i] <= lines) & (lines < bins[i+1]))[0]
        # print('EDGES',bins[[0,-1]])
        # bin_spect, err_spect = np.array([ [*mean_n_std(spectrum[pos(i)])] for i in range(bin_num+1)]).transpose()
        # # plt.figure()
        # # plt.plot(bin_lines,[np.mean(spectrum[pos(i)]) for i in range(len(bin_lines))], '.-')
        # # plt.plot(bin_lines, bin_spect,'x-')
        # # plt.show()
        return (bin_lines, err_lines), (bin_spect, err_spect), bins

    def spectral_data(self, plot_format: bool = False) -> list[ndarray | None]:
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
            spectral_data = [self.lines, self.spec, self.std, self.errs]
        else:
            spectral_data = [self.lines, self.errs, self.spec, self.std]
        if (self.errs is not None) and (self.std is not None): spectral_data = np.array(spectral_data)
        return spectral_data

    def copy(self) -> 'Spectrum':
        """To make an identical copy of a Spectrum object

        Returns
        -------
        target : Spectrum
            the copy
        """
        target = Spectrum([*self.hdul], self.data.copy(), np.copy(self.lims), cut=np.copy(self.cut), sigma=self.sigma, hotpx=False, name=self.name,check_edges=False)
        target.angle = self.angle  
        target.cen  = self.cen
        target.span = self.span
        target.sldata  = self.sldata.copy()  if self.sldata  is not None else None
        target.slsigma = self.slsigma.copy() if self.slsigma is not None else None
        target.spec  = self.spec.copy()  if self.spec  is not None else None
        target.std   = self.std.copy()   if self.std   is not None else None
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
        print(type(index))
        return self.data[index]

    def __getslice__(self, pr: int , af: int ) -> ndarray:
        return self.data[pr:af]

    # def __setitem__(self, index: int | Sequence[int | slice] | slice, value: ArrayLike) -> None:
    #     self.data[index] = value


# class PolyMod():
#     @staticmethod
#     def func(xdata: ArrayLike, *pars) -> ArrayLike:
#         ord = len(pars)-1
#         poly = [ pars[i] * xdata**(ord - i) for i in range(ord+1)]
#         return np.sum(poly,axis=0)

#     def error(self, xdata: ArrayLike, Dxdata: ArrayLike | None, par: ArrayLike, cov: ArrayLike | None = None) -> ArrayLike:
#         ord = self.ord
#         if self.par is None: self.par = np.copy(par)
#         err = 0
#         if Dxdata is not None:
#             err += (np.sum([ par[i] * (ord-i) * xdata**(ord-i-1) for i in range(ord)],axis=0) * Dxdata)**2
#         if cov is not None:
#             der = lambda i : xdata**(ord-i)
#             err += np.sum([ der(i)*der(j) * cov[i,j] for i in range(ord+1) for j in range(ord+1)],axis=0)
#         if len(np.where(err<0)[0]) != 0: 
#             print(err[err<0])
#             raise ValueError("Negative value(s) in uncertainty estimation")
#         err = np.sqrt(err)
#         return err

#     def __init__(self, par: ArrayLike, cov: ArrayLike) -> None:
#         self.ord = len(par) - 1 
#         self.par = np.copy(par)
#         self.cov = np.copy(cov)
    

        

#     def value(self, xdata: ArrayLike) -> ArrayLike:
#         return self.func(xdata,*self.par)

#     def uncert(self, xdata: ArrayLike, Dxdata: ArrayLike | None = None) -> ArrayLike:
#         return self.error(xdata,Dxdata,self.par,self.cov)

# class GaussMod():
#     def __init__(self):
#         self.par = None
#         self.cov = None

#     def func(self,)

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
    @staticmethod
    def poly_func(xdata: ArrayLike, *pars) -> ArrayLike:
        ord = len(pars)-1
        poly = [ pars[i] * xdata**(ord - i) for i in range(ord+1)]
        return np.sum(poly,axis=0)

    @staticmethod
    def poly_error(xdata: ArrayLike, Dxdata: ArrayLike | None, par: ArrayLike, *errargs) -> ArrayLike:
        ord = len(par)-1
        err = 0
        if Dxdata is not None:
            err += (np.sum([ par[i] * (ord-i) * xdata**(ord-i-1) for i in range(ord)],axis=0) * Dxdata)**2
        if len(errargs) != 0:
            err += np.sum(np.square(errargs))
        err = np.sqrt(err)
        return err

    @staticmethod
    def gauss_func(xdata: ArrayLike, *args) -> ArrayLike:
        k, mu, sigma = args
        z = (xdata - mu) / sigma
        return k * np.exp(-z**2/2)

    @staticmethod
    def err_func(xdata: ArrayLike, Dxdata: ArrayLike | None, par: ArrayLike, *errargs) -> ArrayLike:
        coeff = FuncFit.gauss_func(xdata,*par) 
        err = 0
        if Dxdata is not None:
            err += (coeff * (xdata-par[1]) / par[2]**2 * Dxdata)**2
        if len(errargs) != 0:
            err += np.sum(np.square(errargs))
        err = np.sqrt(err)
        return err


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
        self.errvar: float | None = None
        self.res: dict = {}

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
        # if yerr is not None or xerr is not None:
        #     chisq = out.sum_square
        #     chi0 = len(ydata) - len(pop)
        #     self.res['chisq'] = (chisq, chi0)

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
        # if sigma is not None:
        #     chisq = np.sum(((ydata - method(xdata,*pop)) / sigma)**2)
        #     chi0 = len(ydata) - len(pop)
        #     self.res['chisq'] = (chisq, chi0)        
            

    def fit(self, method: Callable[[Any,Any],Any], initial_values: Sequence[Any], mode: Literal['odr','curve_fit'] = 'curve_fit', **fitargs) -> None:
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
        print(self.res['mode'])
        if mode == 'curve_fit':
            self.chi_routine(**fitargs)
        elif mode == 'odr':
            self.odr_routine(**fitargs)
        else: raise ValueError(f'mode = `{mode}` is not accepted')
        self.errvar = np.sqrt(pvariance(self.residuals()))


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

    def pipeline(self,method: Callable[[Any,Any],Any], initial_values: Sequence[Any], names: Sequence[str] | None = None, mode: Literal['odr','curve_fit'] = 'curve_fit',**fitargs) -> None:
        self.fit(method=method,initial_values=initial_values,mode=mode,**fitargs)
        self.infos(names=names)
    
    def gaussian_fit(self, initial_values: Sequence[float] | None = None, names: Sequence[str] = ('k','mu','sigma'),mode: Literal['odr','curve_fit'] = 'curve_fit',**fitargs) -> None:
        """To fit with a Gaussian

        Parameters
        ----------
        intial_values : Sequence[Any]
            k, mu, sigma
        names : list[str] | None, optional
            names, by default None
        """
        if initial_values is None:
            xdata, ydata = self.data[0:2]
            maxpos = ydata.argmax()
            hm = ydata[maxpos]/2
            hm_pos = np.argmin(abs(hm-ydata))
            hwhm = abs(xdata[maxpos]-xdata[hm_pos])
            initial_values = [ydata[maxpos],maxpos,hwhm]

        if mode == 'curve_fit': 
            fitargs['err_func'] = FuncFit.err_func
        self.pipeline(method=FuncFit.gauss_func,initial_values=initial_values,names=names,mode=mode,**fitargs)
        
        # error_function = lambda x, Dx : FuncFit.err_func(x,Dx,self.fit_par,self.res['cov'])
        error_function = lambda x, Dx : FuncFit.err_func(x,Dx,self.fit_par,self.errvar)
        self.res['errfunc'] = error_function


    def pol_fit(self, ord: int | None = None, initial_values: Sequence[float] | None = None, names: Sequence[str] | None = None, mode: Literal['odr','curve_fit'] = 'curve_fit',**fitargs) -> None:
        if initial_values is None:
            if ord is None: raise ValueError('You have to set the grade of the polynomial')
            xdata, ydata = self.data[0:2]
            xtmp = np.copy(xdata)
            ytmp = np.copy(ydata)
            initial_values = [0]
            for _ in range(1,ord+1):
                ytmp = np.diff(ytmp)/np.diff(xtmp)
                initial_values += [np.mean(ytmp)]
                xtmp = (xtmp[1:] + xtmp[:-1])/2
            initial_values = initial_values[::-1] 
            initial_values[-1] = ydata[0] - FuncFit.poly_func(xdata[0],*initial_values)
            del xtmp,ytmp
        elif ord is None: ord = len(initial_values)-1
        if names is None:
            names = []
            for i in range(ord+1):
                names += [f'par_{ord-i}']
        if mode == 'curve_fit': 
            fitargs['err_func'] = FuncFit.poly_error
        self.pipeline(FuncFit.poly_func,initial_values=initial_values,names=names, mode=mode, **fitargs)
        print(self.res['cov'])
        self.res['errfunc'] = lambda x,Dx: FuncFit.poly_error(x,Dx,self.fit_par,self.errvar)
        # self.res['errfunc'] = lambda x,Dx: FuncFit.poly_error(x,Dx,self.fit_par)

    def linear_fit(self, initial_values: Sequence[float] | None = None, names: Sequence[str] = ('m','q'), mode: Literal['odr','curve_fit'] = 'curve_fit',**fitargs) -> None:
        self.pol_fit(ord=1, initial_values=initial_values, names=names, mode=mode, **fitargs)
    
    def sigma(self) -> ArrayLike:
        err = 0
        xdata, _, Dy, Dx = self.data
        if 'errfunc' in self.res.keys() and Dx is not None:
            err_func = self.res['errfunc']
            err += err_func(xdata,Dx)
        if Dy is not None:
            err = np.sqrt(Dy**2 + err**2)
        return err

    def data_plot(self, ax: Axes, points_num: int = 200, grid: bool = True, pltarg1: dict = {}, pltarg2: dict = {},**pltargs) -> None:
        if 'title' not in pltargs.keys():
            pltargs['title'] = 'Fit of the data'
        if 'fontsize' not in pltargs.keys():
            pltargs['fontsize'] = 18
        if 'xlabel' not in pltargs.keys():
            pltargs['xlabel'] = ''
        if 'ylabel' not in pltargs.keys():
            pltargs['ylabel'] = ''
        title = pltargs['title']
        fontsize = pltargs['fontsize']
        ylabel = pltargs['ylabel']
        xlabel = pltargs['xlabel']
        xdata = self.data[0]
        xx = np.linspace(xdata.min(),xdata.max(),points_num)
        if 'fmt' not in pltarg1.keys():
            pltarg1['fmt'] = '.'        
        ax.errorbar(*self.data,**pltarg1)
        ax.plot(xx,self.method(xx),**pltarg2)
        ax.set_xlabel(xlabel,fontsize=fontsize)
        ax.set_ylabel(ylabel,fontsize=fontsize)
        ax.set_title(title,fontsize=fontsize+2)
        if grid: ax.grid(color='lightgray', ls='dashed')
        if 'label' in pltarg1.keys() or 'label' in pltarg2.keys():
            ax.legend(fontsize=fontsize)

    def residuals_plot(self, ax: Axes, grid: bool = True, **pltargs) -> None:
        if 'title' not in pltargs.keys():
            pltargs['title'] = 'Residuals of the data'
        if 'fontsize' not in pltargs.keys():
            pltargs['fontsize'] = 18
        if 'xlabel' not in pltargs.keys():
            pltargs['xlabel'] = ''
        if 'ylabel' not in pltargs.keys():
            pltargs['ylabel'] = 'residuals'
        title = pltargs['title']
        fontsize = pltargs['fontsize']
        ylabel = pltargs['ylabel']
        xlabel = pltargs['xlabel']
        pltargs.pop('title')
        pltargs.pop('fontsize')
        pltargs.pop('ylabel')
        pltargs.pop('xlabel')
        if 'fmt' not in pltargs.keys():
            pltargs['fmt'] = 'o'
        if 'linestyle' not in pltargs.keys():
            pltargs['linestyle'] = 'dashed'
        if 'capsize' not in pltargs.keys():
            pltargs['capsize'] = 3
        xdata = self.data[0]
        ax.errorbar(xdata,self.residuals(),self.sigma(),**pltargs)
        ax.axhline(0,0,1,color='black')
        ax.set_xlabel(xlabel,fontsize=fontsize)
        ax.set_ylabel(ylabel,fontsize=fontsize)
        ax.set_title(title,fontsize=fontsize+2)
        if grid: ax.grid(color='lightgray', ls='dashed')
        if 'label' in pltargs.keys():
            ax.legend(fontsize=fontsize)

    def plot(self, sel: Literal['data','residuals','all'] = 'all', mode: Literal['plots', 'subplots'] = 'plots', points_num: int = 200, fig: None | Figure | tuple[Figure,Figure] = None, grid: bool = True, plot1: dict = {}, plot2: dict = {}, plotargs: dict = {}, **resargs) -> None:
        if fig is None:
            if sel in ['data','all']: 
                fig = plt.figure()
            if sel in ['residuals', 'all'] and mode != 'subplots': 
                fig2 = plt.figure()
        elif isinstance(fig,tuple):
            fig, fig2 = fig
        if sel == 'all' and mode == 'subplots':
            ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
            plotargs['xlabel'] = ''
            resargs['title'] = ''        
            self.data_plot(ax=ax1,points_num=points_num,grid=grid,pltarg1=plot1,pltarg2=plot2,**plotargs)
            self.residuals_plot(ax=ax2,**resargs)
        else:
            if sel in ['data','all']:
                ax = fig.add_subplot(111)
                self.data_plot(ax=ax, points_num=points_num,grid=grid,pltarg1=plot1,pltarg2=plot2,**plotargs)
            if sel in ['residuals', 'all']:
                ax2 = fig2.add_subplot(111)
                self.residuals_plot(ax=ax2,**resargs)

def binning(spectrum: ArrayLike, lines: ArrayLike, bin: float | int | ArrayLike = 50, edges: None | Sequence[float] = None):
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
        and the relative uncertainty is the semi-dispersion
    bins : ndarray
        bins values
    """
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
    print(bins)
    # define some useful quantities
    half_bin = bin_width / 2
    bin_num = len(bins) - 2 
    # define array of the central value in each bin
    bin_lines = bins[:-1] + half_bin
    # set the all uncertainties to the half width
    err_lines = np.full(bin_lines.shape, bin_width / 2)
    # average over the values in each bin
    pos = lambda i : np.where((bins[i] <= lines) & (lines < bins[i+1]))[0]
    print('EDGES',bins[[0,-1]])
    bin_spect = np.array([ np.mean(spectrum[pos(i)]) if len(spectrum[pos(i)]) != 0 else 0 for i in range(bin_num+1)])
    err_spect = np.array([ (spectrum[pos(i)].max()-spectrum[pos(i)].min())/2 if len(spectrum[pos(i)]) != 0 else 0 for i in range(bin_num+1)])
    return (bin_lines, err_lines), (bin_spect, err_spect), bins


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
    
def print_measure(value: float | Quantity, err: float | Quantity, name: str = 'value', unit: str = '') -> None:
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
    