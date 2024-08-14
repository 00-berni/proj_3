import numpy as np
import matplotlib.pyplot as plt
# from .spectralpy import data
from spectralpy.calcorr import calibrated_spectrum, lamp_corr, compute_master_flat
# from spectralpy.display import fastplot, showfits            
import spectralpy.data as spdata
import spectralpy.calcorr as clcr
import spectralpy.display as dsp
if __name__ == '__main__':
    from spectralpy.data import NIGHTS
    from spectralpy.stuff import FuncFit
    TARGETS = spdata.open_targets_list()
    night, target_name, selection = TARGETS[:,0]

    sc_frame, lamp = clcr.get_target_data(night,target_name,selection,angle=None)

    spectr = np.mean(sc_frame.data, axis=1)

    dsp.quickplot(np.arange(len(spectr)),spectr)
    plt.show()

    spec_lamp = lamp.data[300]
    dsp.quickplot(np.arange(len(spec_lamp)),spec_lamp)
    plt.show()


    # height = int(input('Choose the height for the spectrum of the lamp\n> '))


    # tmp = spdata.data_extraction(spdata.os.path.join(spdata.DATA_DIR,night,target_name,'line_curv.json'))
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
    # _, lamp = spdata.angle_correction(lamp.data,angle=angle)
    # spdata.showfits(lamp)
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