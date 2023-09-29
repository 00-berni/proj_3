import numpy as np
# from .spectralpy import data
from spectralpy.calcorr import calibrated_spectrum

if __name__ == '__main__':
    hdul, sp_data, spectrum, lengths, flat_value, cal_func = calibrated_spectrum(0,'betaLyr',display_plots=True)
    