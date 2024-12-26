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

fontsize = 18

def A_class(minpos: float):
    plot_line(balmer,'Balmer','b',minpos)
    # plot_line(o2[0:1],'O$_2$','purple',minpos)
    plt.legend(fontsize=fontsize)

def K_class(minpos: float):
    plot_line(balmer[0:1],'H$\\alpha$','b',minpos)
    plot_line(balmer[1:2],'H$\\beta$','b',minpos)
    # plot_line(balmer[2:3],'H$\\gamma$','b',minpos)
    # plot_line(balmer[3:4],'H$\\delta$','b',minpos)
    # plot_line(caII[1:2],'CaII K','pink',minpos)
    # plot_line(caII[0:1],'CaII H','pink',minpos)
    # plot_line(naI[0:2],'Na I D','green',minpos)
    plot_line([np.mean(naI[0:2])],'NaI D','green',minpos)
    plot_line([np.mean(mgI[1:3])],None,'orange',minpos)
    plot_line(mgI[3:4],'MgI','orange',minpos)
    # plot_line(o2[0:],'O$_2$','purple',minpos)
    plt.legend(fontsize=fontsize)

def G_class(minpos: float):
    plot_line(balmer[0:1],'H$\\alpha$','b',minpos)
    plot_line(balmer[1:2],'H$\\beta$','b',minpos)
    # plot_line(balmer[2:3],'H$\\gamma$','b',minpos)
    # plot_line(balmer[3:4],'H$\\delta$','b',minpos)    
    # plot_line(caII[1:2],'CaII K','pink',minpos)
    # plot_line(caII[0:1],'CaII H','pink',minpos)
    # plot_line(naI[0:2],'Na I D','green',minpos)
    plot_line([np.mean(naI[0:2])],'NaI D','green',minpos)
    plot_line([np.mean(mgI[1:3])],'MgI','orange',minpos)
    # plot_line(mgI[3:4],'MgI','orange',minpos)
    # plot_line(feI[1:2],'FeI','violet',minpos)
    # plot_line(feI[6:7],'FeI','violet',minpos)
    # plot_line(o2[0:],'O$_2$','purple',minpos)
    # plot_line(caI[1:2],'CaI','lime',minpos)
    plt.legend(fontsize=fontsize)

def M_class(minpos: float):
    plot_line(balmer[0:1],'H$\\alpha$','b',minpos)
    plot_line(balmer[1:2],'H$\\beta$','b',minpos)
    plot_line([np.mean(naI[0:2])],'NaI D','green',minpos)
    # plot_line(o2[0:],'O$_2$','purple',minpos)
    # plot_line(caI[:],'CaI','lime',minpos)
    plot_line(tio[:],'TiO','pink',minpos)
    plt.legend(fontsize=fontsize)

if __name__ == '__main__':
    ### 17-03-27 night
    """
    0  - alf Lyr > Vega01     : A0Va
    1  - alf Boo > Arturo     : K1.5III
    2  - mu. Cep > Muceps     : M2-Ia
    3  - bet Dra > Rastaban   : G2Ib-IIa
    """
    SEL_OBJ = 'all'
    SEL_OBJ = ['alflyr','alfboo','betdra']
    # SEL_OBJ = ['alfboo']
    display_plots = True
    # display_plots = False

    atm_data = []
    atm_names = []
    figsize = (14,14)

    # atmfig, atmax = plt.subplots(1,3)

    if 'alflyr' in SEL_OBJ or SEL_OBJ == 'all':
        ## Vega    
        night, target_file, selection, target_name = TARGETS[0]

        ord1 = 2
        ord2 = 3
        target, lamp = spc.calibration(night, target_file, selection, target_name=target_name, angle_fitargs={'mode':'curve_fit'},lamp_fitargs={'mode': 'curve_fit'}, ord_lamp=ord1,balmer_cal=False, ord_balm=ord2, display_plots=display_plots,diagn_plots=False)

        vega_lamp = lamp.copy()
        ## Lines
        data = target.spectral_data(plot_format=True)
        data = data[:,data[0]>=4500]
        minpos = data[1].min()
        l = data[0]
        # spc.quickplot([*data],fmt='-',title= 'calibrated spectrum of ' + target.name, grid=True,labels=('$\\lambda$ [$\\AA$]','I [a.u.]'),capsize=3)
        spc.quickplot([*data],fmt='-',title= 'Calibrated spectrum of ' + target.name,dim=figsize, grid=True,labels=('$\\lambda$ [$\\AA$]','I [a.u.]'),capsize=3,color='black',alpha=0.6)
        # plt.suptitle(target_name,fontsize=20)
        plt.xlim(4500,7700)
        A_class(minpos)
        # plt.show()
        atm_data += [data[:,data[0]>=balmer[0]]]
        atm_names += [target.name]

    # exit()
    # - - #
    if 'alfboo' in SEL_OBJ or SEL_OBJ == 'all':
        if len(SEL_OBJ) == 1: lamp = spc.Spectrum.empty()
        ## Arturo
        night, target_file, selection, target_name = TARGETS[1]
        if SEL_OBJ == 'alfboo': lamp = spc.Spectrum.empty()
        target, lamp = spc.calibration(night, target_file, selection, target_name=target_name, other_lamp=vega_lamp, display_plots=display_plots,diagn_plots=False)
        ## Lines
        data = target.spectral_data(plot_format=True)
        minpos = data[1].min()
        l = data[0]
        data = data[:,data[0]>=4500]
        spc.quickplot([*data],fmt='-',title= 'Calibrated spectrum of ' + target.name,dim=figsize, grid=True,labels=('$\\lambda$ [$\\AA$]','I [a.u.]'),capsize=3,color='black',alpha=0.6)
        # plt.figure()
        # plt.title(target.name+': K type star')
        # plt.errorbar(*data,'.-', color='black',alpha=0.5)
        plt.xlim(4500,7800)
        K_class(minpos)
        # plt.figure()
        # plt.plot(fit.method(np.arange(lerrsen(data[1]))+target.lims[2]),data[1],'.-')
        # K_class(minpos)
        # plt.xlim(3600,7800)
        # plt.show()
        atm_data += [data[:,data[0]>=balmer[0]]]
        atm_names += [target.name]
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

        target, lamp = spc.calibration(night, target_file, selection, target_name=target_name, other_lamp=vega_lamp, display_plots=display_plots,diagn_plots=True)

        ## Lines
        data = target.spectral_data(plot_format=True)
        minpos = data[1].min()
        l = data[0]
        data = data[:,data[0]>=4500]
        spc.quickplot([*data],fmt='-',title= 'Calibrated spectrum of ' + target.name,dim=figsize, grid=True,labels=('$\\lambda$ [$\\AA$]','I [a.u.]'),capsize=3,color='black',alpha=0.6)
        # plt.figure()
        # plt.title(target.name+': M type star')
        # plt.errorbar(*data,'.-', color='black',alpha=0.5)
        plt.xlim(4500,7800)
        M_class(minpos)
        # plt.figure()
        # plt.plot(fit.method(np.arange(len(data[1]))+target.lims[2]),data[1],'.-')
        # M_class(minpos)
        # plt.xlim(3600,7800)
        # plt.show()
        atm_data += [data[:,data[0]>=balmer[0]]]
        atm_names += [target.name]


    # - - #
    if 'betdra' in SEL_OBJ or SEL_OBJ == 'all':
        ## Rastaban
        night, target_file, selection, target_name = TARGETS[3]

        target, lamp = spc.calibration(night, target_file, selection, target_name, other_lamp=vega_lamp, display_plots=display_plots,diagn_plots=False)

        ## Lines
        data = target.spectral_data(plot_format=True)
        minpos = data[1].min()
        l = data[0]
        data = data[:,data[0]>=4500]
        spc.quickplot([*data],fmt='-',title= 'Calibrated spectrum of ' + target.name,dim=figsize, grid=True,labels=('$\\lambda$ [$\\AA$]','I [a.u.]'),capsize=3,color='black',alpha=0.6)
        # plt.figure()
        # plt.title(target.name+': G type star')
        # plt.errorbar(*data,'.-', color='black',alpha=0.5)
        plt.xlim(4500,7800)
        G_class(minpos)
        # plt.show()
        # plt.figure()
        # plt.plot(fit.method(np.arange(len(data[1]))+target.lims[2]),data[1],'.-')
        # G_class(minpos)
        # plt.xlim(3600,7800)
        atm_data += [data[:,data[0]>=balmer[0]]]
        atm_names += [target.name]

    fig, ax = plt.subplots(len(atm_data),1,sharex=True)
    for i in range(len(atm_data)):
        ax[i].errorbar(*atm_data[i],color='black',fmt='-',capsize=0,alpha=0.6)
        ax[i].grid()
        lab_o2 = 'O$_2$' if i == 0 else ''  
        lab_h2o = 'H$_2$O' if i == 0 else '' 
        ax[i].axvspan(6860,6885,facecolor='violet',alpha=0.3,label=lab_o2)
        ax[i].axvspan(7590,7670,facecolor='violet',alpha=0.3)
        ax[i].axvspan(7160,7300,facecolor='cyan',alpha=0.3,label=lab_h2o)
        ax[i].set_ylabel('I [a.u.]',fontsize=fontsize)
        ax[i].set_title(atm_names[i],fontsize=fontsize+2,y=.5,loc='right',ha='left',va='center',rotation=270)
    ax[0].annotate('B',(6872.5,0.075),(6872.5,0.13),fontsize=fontsize)
    ax[0].annotate('A',(7629,0.0044),(7629,0.070),fontsize=fontsize)
    ax[0].annotate('a',(7230,0.06),(7230,0.100),fontsize=fontsize)
    ax[0].legend(fontsize=fontsize)
    ax[0].set_title('Atmospheric bands: telluric contamination',fontsize=fontsize+2,y=1)
    ax[-1].set_xlabel('$\\lambda$ [$\\AA$]',fontsize=fontsize)
    plt.xlim(6800,7690)
    plt.show()