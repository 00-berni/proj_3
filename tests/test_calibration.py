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
    
    sel_high = int((np.argmax(sp_lamp,axis=0)).sum()/sp_lamp.shape[1])

    plt.hlines(sel_high,0,len(sp_lamp[0])-1)

    # plt.show()

    spectrum_lamp = sp_lamp[sel_high]
    fastplot(np.arange(len(spectrum_lamp)), spectrum_lamp, title='Lamp spectrum',labels=['x','counts'])

    plt.show()

    ##############################

    from scipy import odr

    cal_file = os.path.join(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], 'data_files', '01_night', 'betaLyr', 'calibration_lines.txt')

    def fit_func3(param,x):
        p0,p1,p2 = param
        return p0 + p1 * x + p2 * x**2 

    lines, x, Dx = np.loadtxt(cal_file, unpack=True)
    initial_values = [3600,2.65,-3.]
    Dy = np.full(lines.shape, 3.63)
    data = odr.RealData(x,lines,sx=Dx,sy=Dy)
    model = odr.Model(fit_func3)
    fit = odr.ODR(data,model,beta0=initial_values)
    out = fit.run()
    pcov = out.cov_beta
    p0,p1,p2 = out.beta
    Dp0,Dp1,Dp2 = np.sqrt(pcov.diagonal())
    chisq = out.sum_square
    free = len(x) - len(out.beta)

    cal_func = lambda x : fit_func3(out.beta,x)

    print(f'\nFit results for {len(out.beta)} params:\n\t y = p0 + p1 * x + p2 * x^2\n\np0 =\t{p0} +- {Dp0}\t-> {Dp0/p0*100:.2f}%\np1 =\t{p1} +- {Dp1}\t-> {Dp1/p1*100:.2f}%\np2 =\t{p2} +- {Dp2}\t-> {Dp2/p2*100:.2f}%\ncor_01 =\t{pcov[0][1]/Dp0/Dp1*100:.2f} %\ncor_02 =\t{pcov[0][2]/Dp0/Dp2*100:.2f} %\ncor_12 =\t{pcov[2][1]/Dp2/Dp1*100:.2f} %')
    print(f'Chi_red =\t{chisq/free:.2f} +- {np.sqrt(2/free):.2f}')

    sigma = np.sqrt(Dy**2 + (p1*Dx)**2 + (p2*x*Dx*2)**2)
    fig = plt.figure('Calibration',figsize=[8,7])
    fig.suptitle('Fit for lamp calibration')
    ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
    # plt.title('Calibration fit')
    ax1.errorbar(x,lines,xerr=Dx,yerr=Dy,fmt='.',color='violet',label='data')
    ax1.plot(x,cal_func(x),label='Best-fit')
    ax1.set_ylabel('$\lambda$ [$\AA$]')
    ax1.legend(numpoints=1)

    # plt.figure(figsize=[10,7])
    # plt.title('Residuals')
    ax2.axhline(0,xmin=0,xmax=1,linestyle='-.',alpha=0.5)
    ax2.errorbar(x,lines-cal_func(x),xerr=Dx,yerr=sigma,fmt='v',linestyle=':',color='violet')
    ax2.set_xlabel('x [px]')
    ax2.set_ylabel('Residuals [$\AA$]')

    plt.show()    
    
    ###

    def fit_func4(param,x):
        p0,p1,p2,p3 = param
        return p0 + p1 * x + p2 * x**2 + p3 * x**3

    
    initial_values = [3600,2.65,-3.,1.]
    Dy = np.full(lines.shape, 3.63)
    data = odr.RealData(x,lines,sx=Dx,sy=Dy)
    model = odr.Model(fit_func4)
    fit = odr.ODR(data,model,beta0=initial_values)
    out = fit.run()
    pcov = out.cov_beta
    p0,p1,p2,p3 = out.beta
    Dp0,Dp1,Dp2,Dp3 = np.sqrt(pcov.diagonal())
    chisq = out.sum_square
    free = len(x) - len(out.beta)

    cal_func = lambda x : fit_func4(out.beta,x)

    print(f'\nFit results for 4 params:\n\t y = p0 + p1 * x + p2 * x^2 + p3 * x^3\n\np0 =\t{p0} +- {Dp0}\t-> {Dp0/p0*100:.2f}%\np1 =\t{p1} +- {Dp1}\t-> {Dp1/p1*100:.2f}%\np2 =\t{p2} +- {Dp2}\t-> {Dp2/p2*100:.2f}%\np3 =\t{p3} +- {Dp3}\t-> {Dp3/p3*100:.2f}%\ncor_01 =\t{pcov[0][1]/Dp0/Dp1*100:.2f} %\ncor_02 =\t{pcov[0][2]/Dp0/Dp2*100:.2f} %\ncor_03 =\t{pcov[0][3]/Dp0/Dp3*100:.2f} %\ncor_12 =\t{pcov[2][1]/Dp2/Dp1*100:.2f} %\ncor_13 =\t{pcov[1][3]/Dp3/Dp1*100:.2f} %\ncor_23 =\t{pcov[2][3]/Dp3/Dp2*100:.2f} %')
    print(f'Chi_red =\t{chisq/free:.2f} +- {np.sqrt(2/free):.2f}')

    sigma = np.sqrt(Dy**2 + (p1*Dx)**2 + (p2*x*Dx*2)**2 + (p3*x**2*3))
    fig = plt.figure('Calibration',figsize=[8,7])
    fig.suptitle('Fit for lamp calibration')
    ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
    # plt.title('Calibration fit')
    ax1.errorbar(x,lines,xerr=Dx,yerr=Dy,fmt='.',color='violet',label='data')
    ax1.plot(x,cal_func(x),label='Best-fit')
    ax1.set_ylabel('$\lambda$ [$\AA$]')
    ax1.legend(numpoints=1)

    # plt.figure(figsize=[10,7])
    # plt.title('Residuals')
    ax2.axhline(0,xmin=0,xmax=1,linestyle='-.',alpha=0.5)
    ax2.errorbar(x,lines-cal_func(x),xerr=Dx,yerr=sigma,fmt='v',linestyle=':',color='violet')
    ax2.set_xlabel('x [px]')
    ax2.set_ylabel('Residuals [$\AA$]')

    plt.show()


    
    # lengths = trasf_func(np.arange(len(flt_spectrum)))

    # fastplot(lengths,flt_spectrum,title='Corrected and calibrated spectrum of '+sel_obj,labels=['$\lambda$ [$\AA$]','counts'],dim=[17,9])

    # plt.show()