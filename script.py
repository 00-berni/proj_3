import numpy as np
# from .spectralpy import data
from spectralpy.calcorr import calibrated_spectrum, lamp_corr
            

if __name__ == '__main__':

    ### DATA
    display_plots = False
    night = 0
    obj_name1 = 'betaLyr'

    ### ANALYSIS
    spectrum, lengths, cal_res =  calibrated_spectrum(night,obj_name1,display_plots=display_plots, ret_values='calibration')
    
    angle1 = cal_res['angle']
    flat_value = cal_res['flat']
    cal_func = cal_res['func']
    err_func = cal_res['err']

    x = np.arange(len(spectrum))
    Dx = np.full(x.shape,1)
    print(err_func(x,Dx))
    print(err_func(x,Dx,True))

    obj_name2 = 'vega'
    spectrum, lengths, angle2 = calibrated_spectrum(night,obj_name2,flat=flat_value,cal_func=cal_func,err_func=err_func,display_plots=display_plots,ret_values='calibration')

    lag = lamp_corr(night,(obj_name1,obj_name2),(angle1,angle2),display_plots=True)

    obj_name2 = 'gammaCygni'
    spectrum, lengths, angle2 = calibrated_spectrum(night,obj_name2,flat=flat_value,cal_func=cal_func,err_func=err_func,display_plots=display_plots,ret_values='calibration')

    lag = lamp_corr(night,(obj_name1,obj_name2),(angle1,angle2),display_plots=True)