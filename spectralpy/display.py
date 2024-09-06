"""
DISPLAY PACKAGE
===============

***

::METHODS::
-----------

***

!TO DO!
-------


***
    
?WHAT ASK TO STEVE?
-------------------
"""
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from .stuff import Spectrum
from typing import Sequence, Any

##*
def quickplot(data: Sequence[ndarray] | ndarray, numfig: int = None, fmt: str = '-', title: str = '', labels: Sequence[str] = ('',''), dim: list[int] = [10,7], grid: bool = False,**pltargs) -> None:
    """Function to display a plot quickly.

    Parameters
    ----------
    x : ndarray
        Data on x axis
    y : ndarray
        Data on y axis
    numfig : int, optional
        figure number, by default `None`
    fmt : str, optional
        _description_, by default `'-'`
    title : str, optional
        title of the figure, by default `''`
    labels : list[str], optional
        axes label [x,y] format, by default `['','']`
    dim : list[int], optional
        figure size, by default `[10,7]`
    grid : bool, optional
        _description_, by default `False`
    
    Notes
    -----
    You can choose to make a simple plot or adding some stuff.

    (I wrote it only because of my laziness in writing code).

    """
    xl,yl = labels
    plt.figure(numfig,figsize=dim)
    plt.title(title)
    if isinstance(data,ndarray): data = [data]
    plt.plot(*data,fmt,**pltargs)
    plt.xlabel(xl)
    plt.ylabel(yl)
    if grid: plt.grid(which='both',linestyle='--',alpha=0.2,color='grey')


def fits_image(fig: Figure, ax: Axes, data: Spectrum, v: int = -1, subtitle: str = '', **kwargs) -> None:
    """To plot fits images

    Parameters
    ----------
    fig : Figure
        figure variable
    ax : Axes
        axes variable
    data : Spectrum
        target
    v : int, optional
        color-code, by default `-1`
    subtitle : str 
        subtitle, by default `''`
    **kwargs
        parameters of `matplotlib.pyplot.imshow()`
    """
    subtitle = data.name + f', exposure time: {data.get_exposure()} s ' + subtitle
    ax.set_title(subtitle, fontsize=18)
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
    if v == 1 : color = 'viridis'
    elif v == 0 : color = 'gray'
    else : color = 'gray_r'
    image = ax.imshow(data.data, cmap=color,**kwargs)
    cbar = fig.colorbar(image, ax=ax, cmap=color, orientation='horizontal')
    cbar.set_label('intensity [a.u.]')

##*
def show_fits(data: Spectrum, num_plots: Sequence[int] = (1,1), dim: Sequence[int] = (10,7), title: str = '', show: bool = False, **kwargs) -> tuple[Figure, Axes | ndarray]:
    """To plot quickly one or a set of fits pictures
    
    Parameters
    ----------
    data : Spectrum
        target
    num_plots : Sequence[int], optional
        shape of grid of plots, by default `(1,1)`
    dim : Sequence[int], optional
        figure size, by default `(10,7)`
    title : str, optional
        title of the image, by default `''`
    show : bool, optional
        if `True` it displays the figure, by default `False`
    **kwargs
        parameters of `fits_image()` and `matplotlib.pyplot.imshow()`
    Returns
    -------
    fig : Figure
        figure
    axs : Axes | ndarray
        axes
    """
    fig, axs = plt.subplots(*num_plots, figsize=dim) 
    fig.suptitle(title, fontsize=20)
    try:
        for ax in axs:
            fits_image(fig,ax,data,**kwargs)
    except TypeError:
            fits_image(fig,axs,data,**kwargs)
    if show: plt.show()
    return fig, axs
##*
