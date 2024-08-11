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
from numpy.typing import NDArray
import matplotlib.pyplot as plt

##*
def fastplot(x: NDArray, y: NDArray, numfig: int = None, fmt: str = '-', title: str = '', labels: list[str] = ['',''], dim: list[int] = [10,7], grid: bool = False) -> None:
    """Function to display a plot quickly.
    You can choose to make a simple plot or adding some stuff.

    (I wrote it only because of my laziness in writing code).

    Parameters
    ----------
    x : NDArray
        Data on x axis
    y : NDArray
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
    """
    xl,yl = labels
    plt.figure(numfig,figsize=dim)
    plt.title(title)
    plt.plot(x,y,fmt)
    plt.xlabel(xl)
    plt.ylabel(yl)
    if grid:
        plt.grid(which='both',linestyle='--',alpha=0.2,color='grey')


##*
def showfits(data: NDArray, v: int = -1, title: str = '', subtitle: str = '', n: int = None, dim: list[int] = [10,7]) -> None:
    """Function to display the fits image.
    You can display simply the image or set a figure number and a title.

    Parameters
    ----------
    data : NDArray
        image matrix of fits file
    v : int, optional
        cmap parameter, by default `-1`
        -  `1` for false colors 
        -  `0` for grayscale 
        - `-1` for reversed grayscale
    title : str, optional
        title of the image, by default `''`
    n : int, optional
        figure number, by default `None`
    dim : list[int], optional
        figure size, by default `[10,7]`
    """
    plt.figure(n,figsize=dim)
    plt.suptitle(title, fontsize=20)
    plt.title(subtitle, fontsize=18)
    if v == 1 : color = 'viridis'
    elif v == 0 : color = 'gray'
    else : color = 'gray_r'
    plt.imshow(data, cmap=color)
    plt.colorbar(orientation='horizontal')
##*
