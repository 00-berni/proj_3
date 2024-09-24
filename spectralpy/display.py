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
from numpy.typing import ArrayLike
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
    data : Sequence[ndarray] | ndarray
        one can pass a single array of data (on y axis) or 
        x data and y data or in addition the corresponding uncertainty/ies
    numfig : int, optional
        figure number, by default `None`
    fmt : str, optional
        makers format, by default `'-'`
    title : str, optional
        title of the figure, by default `''`
    labels : list[str], optional
        axes label [x,y] format, by default `['','']`
    dim : list[int], optional
        figure size, by default `[10,7]`
    grid : bool, optional
        to plot the grid, by default `False`
    **pltargs
        parameters of `matplotlib.pyplot.errorbar()`
    
    Notes
    -----
    You can choose to make a simple plot or adding some stuff.

    (I wrote it only because of my laziness in writing code).

    """
    xl,yl = labels
    plt.figure(numfig,figsize=dim)
    plt.title(title)
    if isinstance(data,ndarray): data = [np.arange(len(data)),data]
    plt.errorbar(*data,fmt=fmt,**pltargs)
    plt.xlabel(xl)
    plt.ylabel(yl)
    if grid: plt.grid(which='both',linestyle='--',alpha=0.2,color='grey')


def fits_image(fig: Figure, ax: Axes, data: Spectrum, v: int = -1, subtitle: str = '', **figargs) -> None:
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
    **figargs
        parameters of `matplotlib.pyplot.imshow()`
    """
    subtitle = data.name + f', exposure time: {data.get_exposure()} s ' + subtitle
    ax.set_title(subtitle, fontsize=18)
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
    if v == 1 : color = 'viridis'
    elif v == 0 : color = 'gray'
    else : color = 'gray_r'
    image = ax.imshow(data.data, cmap=color,**figargs)
    cbar = fig.colorbar(image, ax=ax, cmap=color, orientation='horizontal')
    cbar.set_label('intensity [a.u.]')

##*
def show_fits(data: Spectrum, num_plots: Sequence[int] = (1,1), dim: Sequence[int] = (10,7), title: str = '', show: bool = False, **figargs) -> tuple[Figure, Axes | ndarray]:
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
    **figargs
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
            fits_image(fig,ax,data,**figargs)
    except TypeError:
            fits_image(fig,axs,data,**figargs)
    if show: plt.show()
    return fig, axs
##*


def display_line(line: ArrayLike, name: str | Sequence[str], err: ArrayLike | None = None, minpos: float = 0, edges: tuple[float,float] = (0,np.inf), label : str | Sequence[str] = '', **pltargs) -> None:
    try:
        for i in range(len(line)):
            if edges[0] <= line[i] <= edges[1]:
                l = line[i]
                n = name[i]
                plt.axvline(l,0,1, label=label, **pltargs)
                plt.annotate(n,(l,minpos),(l+10,minpos))
                if err is not None:
                    plt.axvspan(l-err[i],l+err[i],0,1,alpha=0.3,**pltargs)
    except TypeError:
        if edges[0] <= line <= edges[1]:
            plt.axvline(line,0,1, label=label, **pltargs)
            plt.annotate(name,(line,minpos),(line+10,minpos))
            if err is not None:
                plt.axvspan(line-err,line+err,0,1,alpha=0.3,**pltargs)
