"""
    # Display module
"""
import numpy as np
import matplotlib.pyplot as plt

##*
def fastplot(x: np.ndarray, y: np.ndarray, numfig: int = None, fmt: str = '-', title: str = '', labels: list[str] = ['',''], dim: list[int] = [10,7], grid: bool = False) -> None:
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
    if grid:
        plt.grid(which='both',linestyle='--',alpha=0.2,color='grey')


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
