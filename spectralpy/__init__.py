__author__  = '00-berni'
__version__ = '0.0.0'

from .data import *
from .display import *
from .stuff import Spectrum, FuncFit, print_measure,binning
from .calcorr import get_target_data, calibration, vega_std

TARGETS = open_targets_list()
