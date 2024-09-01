import numpy as np
import matplotlib.pyplot as plt
import spectralpy.data as dt
import spectralpy.calcorr as clcr
import spectralpy.display as dsp

if __name__ == '__main__':
    from spectralpy.stuff import FuncFit
    TARGETS = dt.open_targets_list()
    night, target_name, selection = TARGETS[:,0]

    target, lamp = clcr.calibration(night, target_name, selection, ord=2)


    # night, target_name, selection = TARGETS[:,1]
    # target2, lamp2 = clcr.calibration(night, target_name, selection, other_lamp=lamp)

    ## Vega
    night = '17-03-27'
    target_name = 'Vega'
    selection = 'mean'
    bin_width = 50
    alt = np.array([20,35,43,58],dtype=float)
    Dalt = np.full(alt.shape,0.5)
    x  = 1/np.sin(alt*np.pi/180)
    Dx = Dalt * np.cos(alt*np.pi/180) * x**2 * np.pi/180 
    vega : list[dt.Spectrum] = []
    line = []
    for i in range(len(alt)):
        tmp, _ = clcr.calibration(night,target_name+f'0{i+1}',selection, other_lamp=lamp)
        vega += [tmp]
        line += [[tmp.lines[0],tmp.lines[-1]]]
    min_line = np.max(line,axis=0)[0]
    max_line = np.min(line,axis=0)[1]
    l_data = []
    Dl_data = []
    y_data = []
    Dy_data = []
    a_bin  = []
    for obs in vega:
        start = np.where(obs.lines == min_line)[0][0]
        end = np.where(obs.lines == max_line)[0][0] + 1
        obs.lines = obs.lines[start:end]
        obs.spec = obs.spec[start:end]
        l, y, bins = obs.binning(bin=bin_width)         
        l_data +=  [l[0]]
        y_data +=  [y[0]]
        Dl_data +=  [l[1]]
        Dy_data +=  [y[1]]
        a_bin  +=  [bins]
    print(l_data)
    l_data = np.array(l_data)
    y_data = np.array(y_data)
    Dl_data = np.array(Dl_data)
    Dy_data = np.array(Dy_data)
    a_bin  = np.array(a_bin)
    a_I0   = np.empty((0,2))
    a_tau  = np.empty((0,2))
    plt.figure()
    for i in range(l_data.shape[0]):
        plt.errorbar(l_data[i],y_data[i],Dy_data[i],Dl_data[i])
    plt.show()
    fig1, ax1 = plt.subplots(1,1)
    fig2, ax2 = plt.subplots(1,1)
    for i in range(l_data.shape[1]):
        y = y_data[:,i]
        Dy = Dy_data[:,i]
        print(y)
        initial_values = [-0.5, np.log(y).max()]
        fit = FuncFit(x, np.log(y), xerr=Dx, yerr=Dy/y)
        fit.linear_fit(initial_values, names=('tau','ln(I0)'))
        pop, Dpop = fit.results()
        I0  = np.exp(pop[1])
        DI0 = Dpop[1] * I0
        a_I0  = np.append(a_I0,  [ [I0, DI0] ], axis=0)
        a_tau = np.append(a_tau, [ [-pop[0], Dpop[0]] ], axis=0)

        func = fit.res['func']
        xx = np.linspace(x.min(),x.max(),200)
        color = (0.5,i/l_data.shape[1],1-i/l_data.shape[1])
        ax1.errorbar(x, np.log(y), xerr=Dx, yerr=Dy/y, fmt='.', color=color)
        ax1.plot(xx, func(xx,*pop), color=color)
        ax2.errorbar(x, np.log(y) - func(x,*pop), xerr=Dx, yerr=Dy/y, fmt='.', color=color)
    ax2.axhline(0, 0, 1, color='black')
    plt.figure()
    plt.errorbar(np.arange(len(a_I0)), a_I0[:,0], a_I0[:,1])
    plt.show()





    # sc_frame, lamp = clcr.get_target_data(night,target_name,selection,angle=None)

    # spectr = np.mean(sc_frame.data, axis=0)

    # dsp.quickplot(np.arange(len(spectr)),spectr)
    # plt.show()

    # height = int(len(lamp.data)/2)
    # spec_lamp = lamp.data[height]
    # fig,ax = dsp.show_fits(lamp)
    # ax.axhline(height,0,1,color='blue')
    # plt.show()
    # l, px, err = np.loadtxt(dt.os.path.join(dt.DATA_DIR,night,target_name,'calibration_lines.txt'),unpack=True)
    # dsp.quickplot(np.arange(len(spec_lamp)),spec_lamp)
    # for p in px:
    #     plt.axvline(p,0,1,color='red',linestyle='dashed')
    # plt.show()


    # height = int(input('Choose the height for the spectrum of the lamp\n> '))


    # tmp = dt.data_extraction(dt.os.path.join(dt.DATA_DIR,night,target_name,'line_curv.json'))
    # print(tmp)
    # centre = []
    # fig, ax = plt.subplots(1,1)
    # ax.imshow(lamp.data,cmap='gray_r')
    # for t in tmp:
    #     ptin = np.array(tmp[t]['in'])
    #     ptout = np.array(tmp[t]['out'])
    #     pt = (ptin+ptout)/2
    #     def fitfunc(x,*args):
    #         return x**2*args[0] + x*args[1] + args[2]
    #     y = pt[:,0]
    #     x = pt[:,1]
    #     Dy = (ptout-ptin)[:,0]/2
    #     Dx = [0.5]*len(y)
    #     initial_values = [0.2,1,1]
    #     fit = FuncFit(xdata=x,ydata=y,yerr=Dy,xerr=Dx)
    #     fit.pipeline(fitfunc,initial_values)
    #     pop, Dpop = fit.results()
    #     a,b,c = pop
    #     c = [-(b**2-4*a*c)/(4*a), -b/(2*a)]
    #     centre += [c]
    #     ffig, axx = plt.subplots(1,1)
    #     xx = np.linspace(x.min(),x.max(),100)
    #     axx.plot(fitfunc(xx,*pop),xx,'--',color='violet')
    #     axx.errorbar(y,x,Dx,Dy,fmt='.',color='blue')
    #     plt.show()
    # def linfit(x,*args):
    #     m,q = args
    #     return m*x + q
    # centre = np.array(centre)
    # x = centre[:,1]
    # y = centre[:,0]
    # fit = FuncFit(x,y)
    # fit.pipeline(linfit,[-1,0],['m','q'])
    # pop, Dpop = fit.results()
    # plt.figure()
    # plt.plot(y,x,'.')
    # plt.plot(linfit(x,*pop),x,'--')
    # plt.show()
    # m = pop[0]
    # angle = np.arctan(m)*180/np.pi   # degrees
    # _, lamp = dt.angle_correction(lamp.data,angle=angle)
    # dt.showfits(lamp)
    # plt.show()
    #     ax.plot(*c,'x',color='red')
    #     ax.plot(fitfunc(xx,*pop),xx,'--',color='violet')
    #     ax.errorbar(y,x,Dx,Dy,fmt='.',color='blue')
    # plt.show()

    # ### DATA
    # display_plots = False
    # night = 0
    # obj_name1 = 'betaLyr'

    # ### ANALYSIS
    # spectrum, lengths, cal_res =  calibrated_spectrum(night,obj_name1,display_plots=display_plots, ret_values='calibration')

    # fastplot(lengths, spectrum,title='Corrected and calibrated spectrum of ' + obj_name1,labels=['$\lambda$ [$\AA$]','counts'],dim=[17,9],grid=True)
    # plt.axvline(6563,0,1,color='r',alpha=0.5,label='H$_\\alpha$')
    # plt.axvline(4861,0,1,color='r',alpha=0.5,label='H$_\\beta$')
    # plt.axvline(4340,0,1,color='r',alpha=0.5,label='H$_\\gamma$')
    # plt.axvline(4102,0,1,color='r',alpha=0.5,label='H$_\\delta$')
    # plt.axvline(3970,0,1,color='r',alpha=0.5,label='H$_\\varepsilon$')
    # plt.axvline(3889,0,1,color='r',alpha=0.5,label='H$_\\xi$')
    # plt.axvline(3646,0,1,color='r',linestyle='--',alpha=0.5,label='Balmer Jump')
    # plt.legend(numpoints=1)
    # plt.show()



    # angle1 = cal_res['angle']
    # flat_value = cal_res['flat']
    # cal_func = cal_res['func']
    # err_func = cal_res['err']

    # x = np.arange(len(spectrum))
    # Dx = np.full(x.shape,1)
    # print(err_func(x,Dx))
    # print(err_func(x,Dx,True))

    # obj_name2 = 'vega'
    # spectrum, lengths, angle2 = calibrated_spectrum(night,obj_name2,flat=flat_value,cal_func=cal_func,err_func=err_func,display_plots=display_plots,ret_values='calibration')

    # fastplot(lengths, spectrum,title='Corrected and calibrated spectrum of ' + obj_name2,labels=['$\lambda$ [$\AA$]','counts'],dim=[17,9],grid=True)
    # plt.axvline(6563,0,1,color='r',alpha=0.5,label='H$_\\alpha$')
    # plt.axvline(4861,0,1,color='r',alpha=0.5,label='H$_\\beta$')
    # plt.axvline(4340,0,1,color='r',alpha=0.5,label='H$_\\gamma$')
    # plt.axvline(4102,0,1,color='r',alpha=0.5,label='H$_\\delta$')
    # plt.axvline(3970,0,1,color='r',alpha=0.5,label='H$_\\varepsilon$')
    # plt.axvline(3889,0,1,color='r',alpha=0.5,label='H$_\\xi$')
    # plt.axvline(3646,0,1,color='r',linestyle='--',alpha=0.5,label='Balmer Jump')
    # plt.legend(numpoints=1)
    # plt.show()


    # lag = lamp_corr(night,(obj_name1,obj_name2),(angle1,angle2),display_plots=True)

    # obj_name2 = 'gammaCygni'
    # spectrum, lengths, angle2 = calibrated_spectrum(night,obj_name2,flat=flat_value,cal_func=cal_func,err_func=err_func,display_plots=display_plots,ret_values='calibration')

    # fastplot(lengths, spectrum,title='Corrected and calibrated spectrum of ' + obj_name2,labels=['$\lambda$ [$\AA$]','counts'],dim=[17,9],grid=True)
    # plt.axvline(6563,0,1,color='r',alpha=0.5,label='H$_\\alpha$')
    # plt.axvline(4861,0,1,color='r',alpha=0.5,label='H$_\\beta$')
    # plt.axvline(4340,0,1,color='r',alpha=0.5,label='H$_\\gamma$')
    # plt.axvline(4102,0,1,color='r',alpha=0.5,label='H$_\\delta$')
    # plt.axvline(3970,0,1,color='r',alpha=0.5,label='H$_\\varepsilon$')
    # plt.axvline(3889,0,1,color='r',alpha=0.5,label='H$_\\xi$')
    # plt.axvline(3646,0,1,color='r',linestyle='--',alpha=0.5,label='Balmer Jump')
    # plt.legend(numpoints=1)
    # plt.show()
 
    # lag = lamp_corr(night,(obj_name1,obj_name2),(angle1,angle2),display_plots=True)