import numpy as np
import matplotlib.pyplot as plt
import spectralpy as spc
# from spectralpy import TARGETS

label = lambda i,arr,name : name if i==arr[0] else ''

def plot_line(lines: list[float], name: str, color: str, minpos: float) -> None:
    for line in lines:
        plt.axvline(line,0,1,color=color,label=label(line,lines,name))
        plt.annotate(name,(line,minpos),(line+10,minpos))


b_name = ['H$\\alpha$', 'H$\\beta$', 'H$\\gamma$', 'H$\\delta$', 'H$\\epsilon$', 'H$\\xi$', 'H$\\eta$','H$\\theta$','H$\\iota$','H$\\kappa$']
balmer = [6562.79, 4861.350, 4340.472, 4101.734, 3970.075, 3889.064, 3835.397, 3797.909, 3770.633, 3750.151]
bal_err = [0.03,0.05]+[0.006]*7
feI  = [7610.2676,  7635.8482,  5896.7357, 5274.9807, 4300.2036, 4384.6718, 4401.4425,  4459.3521, 4351.5437]
feII = [7636.2373, 7611.2601, 6871.6994, 6496.9415, 6497.2764, 6497.4985,  5175.3973, 5274.5277, 4384.31313, 4459.67779, 4351.76199, 4336.30962]
tiI  = [6497.683, 4300.4848, 4300.5538, 4301.0787, 4322.1141, 4321.7709 ]
tiII = [4300.0424,  4301.29545, 4350.83776  ]
neI  = [ 5274.0406, 4402.380, 4460.1758, 4336.2251 ]
neII = [4384.1063, 4384.2194, 4322.372]
oI  = [5274.967,5275.123]
oII = [4384.446, 4351.260, 4336.859, 4322.4477]
mgI  = [4351.9057]
mgII = [4384.637]
arI  = []
arII = [4401.75478] 

def display_lines(minpos: float, edges: tuple[float, float]) -> None:
    for (b,err,name) in zip(balmer,bal_err,b_name):
        # b += 100
        if edges[0] <= b <= edges[1]:
            plt.axvline(b,0,1, color='blue',label=label(b,balmer,'H I'))
            plt.annotate(name,(b,minpos),(b+10,minpos))
            plt.axvspan(b-err,b+err,0,1,color='blue',alpha=0.3)
    # plot_line(feI, 'Fe I','orange',minpos)
    # plot_line(feII,'Fe II','yellow',minpos)
    # plot_line(tiI, 'Ti I','violet',minpos)
    # plot_line(tiII,'Ti II','plum',minpos)
    # plot_line(neI, 'Ne I','green',minpos)
    # plot_line(neII,'Ne II','lime',minpos)
    # plot_line(oI, 'O I','deeppink',minpos)
    # plot_line(oII,'O II','hotpink',minpos)
    # plot_line(mgI, 'Mg I','red',minpos)
    # plot_line(mgII,'Mg II','tomato',minpos)
    # plot_line(arI, 'Ar I','aqua',minpos)
    # plot_line(arII,'Ar II','cyan',minpos)
    plt.legend()



if __name__ == '__main__':

    ## DATA
    obs_night = '18-04-22'
    target_name = 'giove'    
    selection = 0
    fit_args = {    'mode': 'curve_fit',
                    'absolute_sigma': True }
    fit_args = {    'mode': 'odr' }
    lim_width = [[0,1391],[[0,112],[108,221]]]
    lag = 20
    jupiter, lamp = spc.get_target_data(obs_night,target_name,selection,angle=None,lim_width=lim_width,lag=lag,gauss_corr=False,lamp_incl=False, fit_args=fit_args, diagn_plots=False)
    heights = np.array([710+i*5 for i in range(4)])
    lamp.spec, lamp.std = spc.mean_n_std(lamp.data[heights],axis=0)
    plt.errorbar(np.arange(*lamp.spec.shape),lamp.spec,lamp.std,fmt='.-')
    plt.show()

    ## WAVELENGTH CALIBRATION
    lines, px, Dpx = spc.get_cal_lines(obs_night,target_name)
    Dlines = lines/20000 / 2

    m0 = np.mean(np.diff(lines)/np.diff(px))
    fit = spc.FuncFit(xdata=px,xerr=Dpx,ydata=lines,yerr=Dlines)
    fit.linear_fit([m0,0])
    fit.plot(mode='subplots',points_num=3)
    plt.show()

    ## BALMER CALIBRATION
    data = jupiter.data.copy()
    row_mean = np.mean(data,axis=1)
    norm_data = np.array([data[i,:] / row_mean[i] for i in range(len(data))])
    jupiter.spec, jupiter.std = spc.mean_n_std(norm_data,axis=0)
    print('LEN',len(jupiter.spec),np.diff(jupiter.lims[2:]))
    jupiter.func = [fit.method, fit.res['errfunc']]
    jupiter.compute_lines()

    cut = 0.867

    plt.figure()
    plt.errorbar(*jupiter.spectral_data(plot_format=True),fmt='.-')
    display_lines(jupiter.spec.min(),(jupiter.lines.min(), jupiter.lines.max()))
    plt.grid()
    plt.axhline(cut,0,1,color='k',linestyle='dashed',alpha=0.5)
    plt.show()

    slice_pos = np.where(jupiter.spec <= cut)[0]
    slice_data = [ elem[slice_pos] for elem in jupiter.spectral_data(True)]


    shift = slice_data[1].max()
    x = slice_data[0] 
    y = slice_data[1] = shift - slice_data[1]

    hm = (y.max()-y.min())/2
    hm_pos = np.argmin(np.abs(hm-y))
    peak_pos = y.argmax()
    hwhm = abs(x[peak_pos]-x[hm_pos])/2
    fit = spc.FuncFit(*slice_data)
    fit.voigt_fit([hwhm,hwhm,y.max(),x.mean()])
    # fit.plot(sel='data')

    slice_data[1] = shift - slice_data[1] 
    plt.figure()
    xx = np.linspace(x.min(),x.max(),200)
    plt.errorbar(*slice_data,fmt='.--')
    plt.plot(xx,shift - fit.method(xx))
    display_lines(jupiter.spec.min(),(jupiter.lines.min(), jupiter.lines.max()))
    plt.grid()
    plt.show()

    med, Dmed = fit.fit_par[-1], fit.fit_err[-1]

    shift = med - balmer[0]
    Dshift = np.sqrt(Dmed**2 + bal_err[0])

    jupiter.lines -= shift
    jupiter.errs = np.sqrt(jupiter.errs**2 + Dshift**2)
    plt.figure()
    plt.errorbar(jupiter.lines,norm_data[10], xerr=jupiter.errs,fmt='.-', color='violet')
    plt.errorbar(jupiter.lines,norm_data[-10], xerr=jupiter.errs,fmt='.-', color='red')
    # plt.errorbar(*jupiter.spectral_data(plot_format=True),fmt='.-')
    display_lines(jupiter.spec.min(),(jupiter.lines.min(), jupiter.lines.max()))
    plt.grid()
    plt.show()

    # cuts = [0,5,10,20]

    # data = jupiter.data.copy()

    # img, imgx = plt.subplots(1,1)
    # imgx.imshow(data,cmap='gray')
    # fig = plt.figure()
    # for i in range(len(cuts)):
    #     sel = cuts[i]
    #     ax = fig.add_subplot(1,len(cuts),i+1)
    #     ax.plot(data[sel,:],label=f'{sel}')
    #     imgx.axhline(sel,0,1)
    #     if sel == 0: sel += 1
    #     sel = data.shape[0] - sel
    #     ax.plot(data[sel,:],label=f'{sel}')
    #     ax.grid()
    #     ax.legend()
    #     ax.set_xlim(800,900)
    #     imgx.axhline(sel,0,1)
    # plt.figure()
    # middle = int(data.shape[0]//2)
    # plt.plot(data[middle,:])
    # plt.plot(data[cuts[2],:])
    # plt.plot(data[-cuts[2],:])
    # plt.grid()
    # plt.xlim(800,900)
    # plt.figure()
    # line_mean = np.mean(data,axis=1)
    # norm_data = np.array([data[i,:] / line_mean[i] for i in range(len(data))])
    # mean_data, std_data = spc.mean_n_std(norm_data,axis=0)
    # plt.errorbar(np.arange(len(mean_data)),mean_data,std_data, np.full(len(mean_data),1/np.sqrt(12)),fmt='.-')
    # plt.grid()
    # plt.show()
