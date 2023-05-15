import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from test_func import collect_fits, data_file_path, fastplot, showfits
from test_extraction import get_data_fit, angle_correction



if __name__ == '__main__':
    # selecting the observation night
    sel_obs = 0
    # choosing the object
    sel_obj = 'betaLyr'
    # collecting data fits for that object
    obj, lims = collect_fits(sel_obs, sel_obj)
    obj_fit, obj_lamp = obj
    lims_fit, lims_lamp = lims

    # appending the path
    obj_fit = data_file_path(sel_obs, sel_obj, obj_fit)
    obj_lamp = data_file_path(sel_obs, sel_obj,obj_lamp)


    hdul, sp_data = get_data_fit(obj_fit, lims=lims_fit, title='Row spectrum of '+sel_obj, n=1)

    # correcting for inclination angle
    angle, sp_data = angle_correction(sp_data)

    showfits(sp_data, title='Rotated image')

    # plt.show()

    # taking the cumulative and plot the spectrum
    spectrum = sp_data.sum(axis=0)

    fastplot(np.arange(len(spectrum)), spectrum, title='Cumulative spectrum', labels=['x','counts'])

    # plt.show()

    hdul_lamp, sp_lamp = get_data_fit(obj_lamp,lims=lims_lamp, title='Row spectrum lamp',n=1)

    _, sp_lamp = angle_correction(sp_lamp, angle=angle)

    showfits(sp_lamp, title='Rotated lamp image')
    
    sel_high = int((685.7+26.8) / 2)

    plt.hlines(sel_high,0,len(sp_lamp[0])-1)

    # plt.show()

    spectrum_lamp = sp_lamp[sel_high]
    fastplot(np.arange(len(spectrum_lamp)), spectrum_lamp, title='Lamp spectrum',labels=['x','counts'])

    plt.show()

    # lamps correlation
    # storing the data of the previous lamp
    prev_lamp = spectrum_lamp


    # choosing the object
    sel_obj = 'gammaCygni'
    # collecting data fits for that object
    obj, lims = collect_fits(sel_obs, sel_obj)
    obj_fit, obj_lamp = obj
    lims_fit, lims_lamp = lims

    # appending the path
    obj_fit = data_file_path(sel_obs, sel_obj, obj_fit)
    obj_lamp = data_file_path(sel_obs, sel_obj,obj_lamp)


    hdul, sp_data = get_data_fit(obj_fit, lims=lims_fit, title='Row spectrum of '+sel_obj, n=1)

    # correcting for inclination angle
    angle, sp_data = angle_correction(sp_data)

    showfits(sp_data, title='Rotated image')

    # plt.show()

    # taking the cumulative and plot the spectrum
    spectrum = sp_data.sum(axis=0)

    fastplot(np.arange(len(spectrum)), spectrum, title='Cumulative spectrum', labels=['x','counts'])

    # plt.show()

    hdul_lamp, sp_lamp = get_data_fit(obj_lamp,lims=lims_lamp, title='Row spectrum lamp',n=1)

    _, sp_lamp = angle_correction(sp_lamp, angle=angle)

    showfits(sp_lamp, title='Rotated lamp image')
    
    sel_high = int((685.7+26.8) / 2)

    plt.hlines(sel_high,0,len(sp_lamp[0])-1)

    # plt.show()

    spectrum_lamp = sp_lamp[sel_high]
    fastplot(np.arange(len(spectrum_lamp)), spectrum_lamp, title='Lamp spectrum',labels=['x','counts'])
    plt.plot(prev_lamp)

    plt.show()
    
    corr = correlate(spectrum_lamp,prev_lamp,'same')

    plt.figure(figsize=[15,7])
    plt.plot(corr,'v',color='red')
    plt.plot(corr,':',alpha=0.7)
    plt.plot(correlate(spectrum_lamp,spectrum_lamp,'same'),'v',color='green')
    plt.show()

    prev_x_lamp = (np.argsort(prev_lamp)[::-1])[:38]
    x_lamp = (np.argsort(spectrum_lamp)[::-1])[:38]

    plt.figure()
    plt.plot((prev_x_lamp-x_lamp), 'v--',color='violet')
    
    plt.figure()
    plt.plot(x_lamp,spectrum_lamp[x_lamp], 'v',color='blue')

    plt.show()
