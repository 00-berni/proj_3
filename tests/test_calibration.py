import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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

    cal_file = os.path.join(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], 'data_files', '01_night', 'betaLyr', 'calibration_lines.txt')

    lines, x, Dx = np.loadtxt(cal_file, unpack=True)

    plt.figure()
    plt.errorbar(x,lines,xerr=Dx,fmt='.')

    def fit_func3(x,p0,p1,p2):
        return p0 + x*p1 + x**2*p2
    
    initial_values = [1400,2.5,1e-4]
    pop, pcov = curve_fit(fit_func3,x,lines,initial_values)
    for i in range(2):
        initial_values = pop
        Dy = np.sqrt((Dx*pop[1])**2 + (x*Dx*pop[2]/2)**2)
        pop, pcov = curve_fit(fit_func3,x,lines,initial_values, sigma=Dy)
    p0,p1,p2 = pop
    Dp0,Dp1,Dp2 = np.sqrt(pcov.diagonal())

    print(f'Fit results 3 params\np0 = {p0} +- {Dp0}\np1 = {p1} +- {Dp1}\np2 = {p2} +- {Dp2}\n')
    
    plt.plot(np.arange(max(x)),fit_func3(np.arange(max(x)),p0,p1,p2))

    plt.show()    
    

    plt.figure()
    plt.errorbar(x,lines,xerr=Dx,fmt='.')

    def fit_func2(x,p0,p1):
        return p0 + x*p1
    
    initial_values = [3600,2.65]
    pop, pcov = curve_fit(fit_func2,x,lines,initial_values)
    for i in range(2):
        initial_values = pop
        Dy = Dx*pop[1]
        pop, pcov = curve_fit(fit_func2,x,lines,initial_values, sigma=Dy)
    p0,p1 = pop
    Dp0,Dp1 = np.sqrt(pcov.diagonal())

    print(f'Fit results 2 params\np0 = {p0} +- {Dp0}\np1 = {p1} +- {Dp1}')
    
    plt.plot(np.arange(max(x)),fit_func2(np.arange(max(x)),p0,p1))

    plt.show()

    trasf_func = lambda x : fit_func2(x,p0,p1)

    
    lengths = trasf_func(np.arange(len(flt_spectrum)))

    fastplot(lengths,flt_spectrum,title='Corrected and calibrated spectrum of '+sel_obj,labels=['$\lambda$ [$\AA$]','counts'],dim=[17,9])

    plt.show()