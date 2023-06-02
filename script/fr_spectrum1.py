import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from functions import *

if __name__ == '__main__':
    hdul, sp_data, spectrum, lengths, flat_value, cal_func = calibrated_spectrum(0,'betaLyr','giove')
