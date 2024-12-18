import numpy as np
import matplotlib.pyplot as plt
import spectralpy as spc
from spectralpy import TARGETS

label = lambda i,arr,name : name if i==arr[0] else ''

def plot_line(lines: list[float], name: str, color: str, minpos: float,**figargs) -> None:
    if 'linestyle' not in figargs.keys():
        figargs['linestyle'] = '--'
    for line in lines:
        plt.axvline(line,0,1,color=color,label=label(line,lines,name),**figargs)
        # plt.annotate(name,(line,minpos),(line+10,minpos))


b_name = ['H$\\alpha$', 'H$\\beta$', 'H$\\gamma$', 'H$\\delta$', 'H$\\epsilon$', 'H$\\xi$', 'H$\\eta$','H$\\theta$','H$\\iota$','H$\\kappa$']
balmer = [6562.79, 4861.350, 4340.472, 4101.734, 3970.075, 3889.064, 3835.397, 3797.909, 3770.633, 3750.151]
bal_err = [0.03,0.05]+[0.006]*7
feI  = [3021.0725, 3581.1928, 3820.4249, 4307.9020, 4383.5447,  4668.1341,  4957.5965, 5168.8978, 5269.5370 ] #[7610.2676,  7635.8482,  5896.7357, 5274.9807, 4300.2036, 4384.6718, 4401.4425,  4459.3521, 4351.5437]
feII = [7636.2373, 7611.2601, 6871.6994, 6496.9415, 6497.2764, 6497.4985,  5175.3973, 5274.5277, 4384.31313, 4459.67779, 4351.76199, 4336.30962]
tiI  = [6497.683, 4300.4848, 4300.5538, 4301.0787, 4322.1141, 4321.7709 ]
tiII = [4300.0424,  4301.29545, 4350.83776  ]
neI  = [ 5274.0406, 4402.380, 4460.1758, 4336.2251 ]
neII = [4384.1063, 4384.2194, 4322.372]
oI  = [5274.967,5275.123]
oII = [4384.446, 4351.260, 4336.859, 4322.4477]
mgI  = [4351.9057,5167.3216,5172.6843,5183.6042]
mgII = [4384.637]
arI  = []
arII = [4401.75478] 
caI  = [4226.73,4302.53 , 4307.74 ]
caII = [3968.47,3933.66]
naI = [5889.95095,5895.92424]
o2 = [6276.61, 6867.19,7593.70,8226.96  ]
tio = [6174.86585693, 6320.74407898]
#tio = [5160, 5450, 5850, 6160]

def display_lines(minpos: float, edges: tuple[float, float], sel: str | list[str] = 'HI') -> None:
    if 'HI' in sel:
        for (b,err,name) in zip(balmer,bal_err,b_name):
            # b += 100
            if edges[0] <= b <= edges[1]:
                plt.axvline(b,0,1, color='blue',label=label(b,balmer,'H I'))
                plt.annotate(name,(b,minpos),(b+10,minpos))
                plt.axvspan(b-err,b+err,0,1,color='blue',alpha=0.3)
    if 'BHa' in sel:
        plot_line(balmer[0:1], b_name[0],'orange',minpos)
    if 'BHb' in sel:
        plot_line(balmer[1:2], b_name[1],'orange',minpos)
    if 'BHc' in sel:
        plot_line(balmer[2:3], b_name[2],'orange',minpos)
    if 'BHd' in sel:
        plot_line(balmer[3:4], b_name[4],'orange',minpos)
    if 'Fe' in sel or 'FeI' in sel:
        plot_line(feI, 'Fe I','orange',minpos)
    if 'Fe' in sel or 'FeII' in sel:
        plot_line(feII,'Fe II','yellow',minpos)
    if 'Ti' in sel or 'TiI' in sel:
        plot_line(tiI, 'Ti I','violet',minpos)
    if 'Ti' in sel or 'TiII' in sel:
        plot_line(tiII,'Ti II','plum',minpos)
    if 'Ne' in sel or 'NeI' in sel:
        plot_line(neI, 'Ne I','green',minpos)
    if 'Ne' in sel or 'NeII' in sel:
        plot_line(neII,'Ne II','lime',minpos)
    if 'O' in sel or 'OI' in sel:
        plot_line(oI, 'O I','deeppink',minpos)
    if 'O' in sel or 'OII' in sel:
        plot_line(oII,'O II','hotpink',minpos)
    if 'Mg' in sel or 'MgI' in sel:
        plot_line(mgI, 'Mg I','red',minpos)
    if 'Mg' in sel or 'MgII' in sel:
        plot_line(mgII,'Mg II','tomato',minpos)
    if 'Ar' in sel or 'ArI' in sel:
        plot_line(arI, 'Ar I','aqua',minpos)
    if 'Ar' in sel or 'ArII' in sel:
        plot_line(arII,'Ar II','cyan',minpos)
    if 'Ca' in sel or 'CaII' in sel:
        plot_line(caII,'Ca II','cyan',minpos)
    if 'Na' in sel or 'NaI' in sel:
        plot_line(naI,'Na I','green',minpos)
    plt.legend()

def A_class(minpos: float):
    plot_line(balmer,'Balmer','b',minpos)
    # plot_line(o2[0:1],'O$_2$','purple',minpos)
    plt.legend()

def K_class(minpos: float):
    plot_line(balmer[0:1],'H$\\alpha$','b',minpos)
    plot_line(balmer[1:2],'H$\\beta$','b',minpos)
    plot_line(balmer[2:3],'H$\\gamma$','b',minpos)
    plot_line(balmer[3:4],'H$\\delta$','b',minpos)
    plot_line(caII[1:2],'CaII K','pink',minpos)
    plot_line(caII[0:1],'CaII H','pink',minpos)
    # plot_line(naI[0:2],'Na I D','green',minpos)
    plot_line([np.mean(naI[0:2])],'NaI D mean','green',minpos)
    plot_line([np.mean(mgI[1:3])],'MgI mean','orange',minpos)
    plot_line(mgI[3:4],'MgI','orange',minpos)
    # plot_line(o2[0:],'O$_2$','purple',minpos)
    plt.legend()

def G_class(minpos: float):
    plot_line(balmer[0:1],'H$\\alpha$','b',minpos)
    plot_line(balmer[1:2],'H$\\beta$','b',minpos)
    plot_line(balmer[2:3],'H$\\gamma$','b',minpos)
    plot_line(balmer[3:4],'H$\\delta$','b',minpos)    
    plot_line(caII[1:2],'CaII K','pink',minpos)
    plot_line(caII[0:1],'CaII H','pink',minpos)
    # plot_line(naI[0:2],'Na I D','green',minpos)
    plot_line([np.mean(naI[0:2])],'NaI D mean','green',minpos)
    plot_line([np.mean(mgI[1:3])],'MgI mean','orange',minpos)
    # plot_line(mgI[3:4],'MgI','orange',minpos)
    plot_line(feI[1:2],'FeI','violet',minpos)
    # plot_line(feI[6:7],'FeI','violet',minpos)
    # plot_line(o2[0:],'O$_2$','purple',minpos)
    plot_line(caI[1:2],'CaI','lime',minpos)
    plt.legend()

def M_class(minpos: float):
    plot_line(balmer[0:1],'H$\\alpha$','b',minpos)
    plot_line([np.mean(naI[0:2])],'NaI D mean','green',minpos)
    # plot_line(o2[0:],'O$_2$','purple',minpos)
    # plot_line(caI[:],'CaI','lime',minpos)
    plot_line(tio[:],'TiO','pink',minpos)
    plt.legend()

if __name__ == '__main__':
    ### 17-03-27 night
    """
    0  - alf Lyr > Vega01     : A0Va
    1  - alf Boo > Arturo     : K1.5III
    2  - mu. Cep > Muceps     : M2-Ia
    3  - bet Dra > Rastaban   : G2Ib-IIa
    """
    SEL_OBJ = ['alflyr','alfboo','betdra']
    display_plots = False

    # atmfig, atmax = plt.subplots(1,3)

    if 'alflyr' in SEL_OBJ or SEL_OBJ == 'all':
        ## Vega    
        night, target_file, selection, target_name = TARGETS[0]

        ord1 = 2
        ord2 = 3
        target, lamp = spc.calibration(night, target_file, selection, target_name=target_name, angle_fitargs={'mode':'curve_fit'},lamp_fitargs={'mode': 'curve_fit'}, ord_lamp=ord1,balmer_cal=False, ord_balm=ord2, display_plots=display_plots,diagn_plots=False)

        ## Lines
        data = target.spectral_data(plot_format=True)
        minpos = data[1].min()
        l = data[0]
        spc.quickplot([*data],fmt='-',title= 'calibrated spectrum of ' + target.name, grid=True,labels=('$\\lambda$ [$\\AA$]','I [a.u.]'),capsize=3)
        spc.quickplot([*data],fmt='-',title= 'A type Star', grid=True,labels=('$\\lambda$ [$\\AA$]','I [a.u.]'),capsize=3,color='black',alpha=0.6)
        plt.suptitle(target_name,fontsize=20)
        plt.xlim(4500,7700)
        A_class(minpos)
        plt.show()
        # bal_lines = [
        #     6562.790,
        #     4861.350,
        #     4340.472,
        #     4101.734,
        #     3970.075,
        #     3889.064,
        #     3835.397,
        #     3797.909
        # ]
        # bal_errs = [
        #     0.030,
        #     0.050,
        #     0.006,
        #     0.006,
        #     0.006,
        #     0.006,
        #     0.006,
        #     0.006    
        # ]

        # cen = [
        #     837.5,
        #     347,
        #     200,
        #     132,
        #     94,
        #     71,
        #     55.5,
        #     45  
        # ]

        # width = [
        #     7.5,
        #     7,
        #     10,
        #     12,
        #     8,
        #     6,
        #     4.5,
        #     3
        # ]
        # data = target.spec.copy()
        # data = data.max()-data
        # plt.figure()
        # plt.plot(data,'.--')
        # for c,w in zip(cen,width):
        #     plt.axvline(c,0,1,color='red')
        #     plt.axvspan(c-w,c+w,facecolor='orange')
        # plt.show()

        # for i in range(len(cen)):
        #     c = cen[i]
        #     w = width[i]
        #     lf, rg = int(c-w),int(c+w)
        #     xtmp = np.arange(lf,rg+1)
        #     valtmp = data[lf:rg+1].copy()
        #     cen[i], width[i] = spc.mean_n_std(xtmp,weights=valtmp)
        #     maxpos = np.argmax(valtmp)
        #     hm = valtmp[maxpos]/2
        #     pl_hm = np.argmin(abs(hm-valtmp[:maxpos]))
        #     pr_hm = np.argmin(abs(hm-valtmp[maxpos+1:])) + maxpos
        #     pos = [pl_hm,pr_hm]
        #     pos = pos[np.argmin([abs(valtmp[pl_hm]-hm),abs(valtmp[pr_hm]-hm)])]
        #     hwhm = abs(cen[i]-lf-pos)
        #     width[i] = hwhm

        # width = [1]*len(cen)

        # plt.figure()
        # plt.plot(target.spec,'.--')
        # for c,w in zip(cen,width):
        #     plt.axvline(c,0,1,color='red')
        #     plt.axvspan(c-w,c+w,facecolor='orange')
        # plt.show()
        # # print(len(cen),len(width),len(bal_lines),len(bal_errs))
        # fit = spc.FuncFit(xdata=np.array(cen)+target.lims[2],ydata=bal_lines,xerr=width,yerr=bal_errs)
        # fit.pol_fit(2,mode='curve_fit')
        # fit.plot(mode='subplots')
        # plt.figure()
        # plt.hist(fit.residuals(),5)
        # plt.figure()
        # # x = np.arange(len(target.spec))
        # # plt.errorbar(fit.method(x+target.lims[2]),target.spec,target.std,fit.errvar,'.--k')
        # # A_class(minpos)
        # # plt.show()


    # exit()
    # - - #
    if 'alfboo' in SEL_OBJ or SEL_OBJ == 'all':
        ## Arturo
        night, target_file, selection, target_name = TARGETS[1]
        if SEL_OBJ == 'alfboo': lamp = spc.Spectrum.empty()
        target, lamp = spc.calibration(night, target_file, selection, target_name=target_name, other_lamp=lamp, display_plots=display_plots,diagn_plots=False)
        exit()
        ## Lines
        data = target.spectral_data(plot_format=True)
        minpos = data[1].min()
        l = data[0]
        plt.figure()
        plt.title(target.name+': K type star')
        plt.errorbar(*data,'.-', color='black',alpha=0.5)
        plt.xlim(4500,7800)
        K_class(minpos)
        # plt.figure()
        # plt.plot(fit.method(np.arange(len(data[1]))+target.lims[2]),data[1],'.-')
        # K_class(minpos)
        # plt.xlim(3600,7800)
        plt.show()
        # px = [22,32,285,579,775,867.5]
        # Dpx = [1,0.5,1.5,2,2,2]
        # lines = [*caII,np.mean(naI[0:2]),balmer[1],balmer[0],o2[0]]

        # fit = spc.FuncFit(px,lines,xerr=Dpx)
        # fit.pol_fit(2,[1,5000],mode='curve_fit')
        # fit.plot()

        # plt.figure()
        # plt.plot(fit.method(np.arange(len(data[1]))),data[1],'.-',color='k')
        # K_class(minpos)
        # plt.xlim(3600,7800)
        # plt.show()

    # - - #

    if 'mucep' in SEL_OBJ or SEL_OBJ == 'all':
        ## Muceps
        night, target_file, selection, target_name = TARGETS[2]

        target, lamp = spc.calibration(night, target_file, selection, target_name=target_name, other_lamp=lamp, display_plots=display_plots,diagn_plots=True)

        ## Lines
        data = target.spectral_data(plot_format=True)
        minpos = data[1].min()
        l = data[0]
        plt.figure()
        plt.title(target.name+': M type star')
        plt.errorbar(*data,'.-', color='black',alpha=0.5)
        plt.xlim(4500,7800)
        M_class(minpos)
        # plt.figure()
        # plt.plot(fit.method(np.arange(len(data[1]))+target.lims[2]),data[1],'.-')
        # M_class(minpos)
        # plt.xlim(3600,7800)
        plt.show()

        px = []
        lines = []

    # - - #
    if 'betdra' in SEL_OBJ or SEL_OBJ == 'all':
        ## Rastaban
        night, target_file, selection, target_name = TARGETS[3]

        target, lamp = spc.calibration(night, target_file, selection, target_name, other_lamp=lamp, display_plots=display_plots,diagn_plots=True)

        ## Lines
        data = target.spectral_data(plot_format=True)
        minpos = data[1].min()
        l = data[0]
        plt.figure()
        plt.title(target.name+': G type star')
        plt.errorbar(*data,'.-', color='black',alpha=0.5)
        plt.xlim(4500,7800)
        G_class(minpos)
        # plt.figure()
        # plt.plot(fit.method(np.arange(len(data[1]))+target.lims[2]),data[1],'.-')
        # G_class(minpos)
        # plt.xlim(3600,7800)
        # plt.show()

        px = []
        lines = []

#    ## 22-07-26_ohp night
    """
    4  - vega
    5  - giove
    6  - luna0
    7  - luna1
    8  - pCygni
    9  - m57
    """    
    # night, target_name, selection = TARGETS[4]

    # ord_lamp = 2
    # ord_balm = 2
    # display_plots = True
    # target, lamp = spc.calibration(night, target_name, selection,lag=7,row_num=4, ord_lamp=ord_lamp, ord_balm=ord_balm, display_plots=display_plots,diagn_plots=True)
    
    
    # night, target_name, selection = TARGETS[5]

    # target, lamp = spc.calibration(night, target_name, selection, angle=target.angle, other_lamp=lamp, display_plots=True,diagn_plots=True)

    

#    ## 23-03-28 night
    # night, target_name, selection = TARGETS[-2]

    # ord = 2
    # display_plots = True
    # target, lamp = spc.calibration(night, target_name, selection, ord=ord, display_plots=True)


    # night, target_name, selection = TARGETS[2]

    # target, lamp = spc.calibration(night, target_name, selection, other_lamp=lamp, display_plots=display_plots)

    # data = target.spectral_data(plot_format=True)
    # minpos = data[1].min()
    # l = data[0]
    # plt.figure()
    # plt.errorbar(*data,'.-', color='black',alpha=0.5)
    # plt.xlim(3600,7800)
    # display_lines(minpos,(l.min(),l.max()))
    # plt.show()

    # tg, lp = dt.open_results(['polluce_mean','lamp-polluce'],night,target_name)
    # minpos = tg[2].min()
    # dsp.quickplot((tg[0],tg[2],tg[3],tg[1]),fmt='.--',color='black')
    # plt.legend()
    # plt.yscale('log')
    # plt.show()

    # night, target_name, selection = TARGETS[:,1]
    # target2, lamp2 = spc.calibration(night, target_name, selection, other_lamp=lamp)

#    ## Vega

# - - #

    # file_name = dt.os.path.join(dt.CAL_DIR,'standards','Vega','vega_std.fit')
    # # lims = [592,618,251,-1,600,612,243,-1]
    # lims = [592,618,251,-1,605,610,243,-1]
    # std = dt.get_data_fit(file_name,lims,obj_name='Standard Vega')
    # std, angle = std.angle_correction(diagn_plots=False)
    # std.cut_image()
    # dsp.show_fits(std,show=True)
    # meandata = std.data.mean(axis=1)
    # data = np.array([ std.data[i,:] / meandata[i] for i in range(len(meandata)) ])
    # std.spec, std.std = spc.mean_n_std(data, axis=0)

    # from numpy.typing import ArrayLike
    # print(' - - STARDARD - - ')
    # file_path = dt.os.path.join(dt.CAL_DIR,'standards','Vega','H_calibration.txt')
    # balm, balmerr, lines, errs = np.loadtxt(file_path, unpack=True)
    # lines += std.lims[2]
    # ord = 3
    # fit = spc.FuncFit(xdata=lines, xerr=errs, ydata=balm, yerr=balmerr)
    # initial_values = [0] + [1]*(ord-1) + [np.mean(lines)]
    # fit.pol_fit(ord, initial_values=initial_values)
    # pop = fit.fit_par.copy()
    # cov = fit.res['cov']

    # def balm_calfunc(x: ArrayLike) -> ArrayLike:
    #     return fit.res['func'](x, *pop)

    # def balm_errfunc(x: ArrayLike, Dx: ArrayLike) -> ArrayLike:
    #     par_err = [ x**(2*ord-(i+j)) * cov[i,j] for i in range(ord+1) for j in range(ord+1)]
    #     err = np.sum([ pop[i]*(ord-i)*x**(ord-i-1) for i in range(ord) ],axis=0)
    #     return np.sqrt((err * Dx)**2 + np.sum(par_err, axis=0))

    # target.spec = target.spec / target.get_exposure()
    # target.std = target.std / target.get_exposure()
    # std.spec = std.spec / std.get_exposure()
    # std.std = std.std / std.get_exposure()

    # std.func = [balm_calfunc, balm_errfunc]
    # std.compute_lines()
    # data = std.spectral_data(True)

    # l = data[0]
    # minpos = data[1].min()
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.errorbar(*data,fmt='.-')
    # display_lines(minpos,(l.min(),l.max()))
    # plt.subplot(2,1,2)
    # data = target.spectral_data(True)
    # l = data[0]
    # minpos = data[1].min()    
    # plt.errorbar(*data,fmt='.-')
    # display_lines(minpos,(l.min(),l.max()))


    # plt.show()


#   ## Filters
    # from speclite import filters as flt
    # filter_b = flt.load_filter('bessell-B')(l_data)
    # filter_v = flt.load_filter('bessell-V')(l_data)
    # filter_r = flt.load_filter('bessell-R')(l_data)

    # filter_b /= filter_b.mean()
    # filter_v /= filter_v.mean()
    # filter_r /= filter_r.mean()

    # from scipy.signal import convolve
    # data_b = convolve(I0, filter_b, mode='same')
    # data_v = convolve(I0, filter_v, mode='same')
    # data_r = convolve(I0, filter_r, mode='same')
    # data_b /= data_b.mean()    
    # data_v /= data_v.mean()
    # data_r /= data_r.mean()
    
    # plt.figure()
    # plt.plot(l_data,I0,'.-',color='black')
    # plt.plot(l_data,filter_b,'.-',color='blue')
    # plt.plot(l_data,filter_v,'.-',color='green')
    # plt.plot(l_data,filter_r,'.-',color='red')
    # plt.figure()
    # plt.plot(l_data,data_b,'.-',color='blue')
    # plt.plot(l_data,data_v,'.-',color='green')
    # plt.plot(l_data,data_r,'.-',color='red')
    # plt.figure()
    # plt.plot(l_data,data_b-data_v,'.-',color='violet')
    # plt.plot(l_data,data_v-data_r,'.-',color='orange')
    # plt.show()






    # sc_frame, lamp = spc.get_target_data(night,target_name,selection,angle=None)

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
    # diagn_plots = False
    # night = 0
    # obj_name1 = 'betaLyr'

    # ### ANALYSIS
    # spectrum, lengths, cal_res =  calibrated_spectrum(night,obj_name1,diagn_plots=diagn_plots, ret_values='calibration')

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
    # spectrum, lengths, angle2 = calibrated_spectrum(night,obj_name2,flat=flat_value,cal_func=cal_func,err_func=err_func,diagn_plots=diagn_plots,ret_values='calibration')

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


    # lag = lamp_corr(night,(obj_name1,obj_name2),(angle1,angle2),diagn_plots=True)

    # obj_name2 = 'gammaCygni'
    # spectrum, lengths, angle2 = calibrated_spectrum(night,obj_name2,flat=flat_value,cal_func=cal_func,err_func=err_func,diagn_plots=diagn_plots,ret_values='calibration')

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
 
    # lag = lamp_corr(night,(obj_name1,obj_name2),(angle1,angle2),diagn_plots=True)