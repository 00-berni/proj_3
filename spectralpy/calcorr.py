"""
    # Calibration module
"""
import numpy as np
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans



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
    # checking the presence of `NaNs`
    if True in np.isnan(data):
        # building a gaussian kernel for the interpolation
        kernel = Gaussian2DKernel(x_stddev=1)
        # removing the `NaNs`
        data = interpolate_replace_nans(data, kernel)
    return data
