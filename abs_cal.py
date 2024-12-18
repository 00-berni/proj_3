import numpy as np
import matplotlib.pyplot as plt
import spectralpy as spc
from scipy.interpolate import CubicSpline

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
caII = [3968.47,3933.66]

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
    plt.legend()


def remove_balmer(lines: np.ndarray, spectrum: np.ndarray, wlen_width: float = 80, display_plots: bool = False) -> np.ndarray:
    wlen = lines.copy()
    spec = spectrum.copy()
    bins = []
    wlen_ends = []
    for bal in balmer:
        pos = np.where((wlen >= bal - wlen_width) & (wlen <= bal + wlen_width))[0]
        if len(pos) != 0:
            bins += [pos]
            wlen_ends += [[bal - wlen_width, bal + wlen_width]]
            wlen = np.delete(wlen,pos)
            spec  = np.delete(spec ,pos)
    if display_plots:
        plt.figure()
        plt.plot(wlen,spec,'.-')
    interpol = CubicSpline(wlen,spec)
    spectrum = spectrum.copy()
    for p in bins:
        spectrum[p] = interpol(lines[p])
    if display_plots:
        plt.figure()
        plt.plot(lines,spectrum,'.-')
        plt.show()
    return spectrum

if __name__ == '__main__':
    ## Data
    print('-- DATA --')
    night = '17-03-27'
    target_name = 'Vega'
    selection = 'mean'
    alt  = np.array([21.57,34.68,43.90,59.06])
    Dalt = np.full(alt.shape,0.03)

    ## Wavelength Calibration
    print('-- WAVELENGTH CALIBRATION --')
    ord1 = 2
    ord2 = 3
    display_plots = False
    target, lamp = spc.calibration(night, target_name+'01', selection, balmer_cal=False, ord_lamp=ord1, ord_balm=ord2, display_plots=display_plots,diagn_plots=False)
    tmp = target.copy()
    tmp.spec = remove_balmer(tmp.lines,tmp.spec)
    
    ## Prepare Data
    print('-- PREPARE DATA --')
    bin_width = 50
    targets = [target]
    vega = [tmp]
    wlen_ends = [[target.lines[0],target.lines[-1]]]
    for i in range(1,len(alt)):
        tmp, _ = spc.calibration(night,target_name+f'0{i+1}',selection, other_lamp=lamp, display_plots=False)
        targets += [tmp]
        tmp.spec = remove_balmer(tmp.lines, tmp.spec)
        vega += [tmp]
        wlen_ends += [[tmp.lines[0],tmp.lines[-1]]]
        # plt.figure()
        # plt.errorbar(*tmp.spectral_data(True),fmt='.-')
        # display_lines(tmp.spec.min(),(tmp.lines.min(),tmp.lines.max()))
        # plt.show()
    
    ## Response Function
    print('-- RESPONSE FUNCTION --')
    wlen_ends = (4500, 6700)

    wlen, rfunc, tau = spc.ccd_response((alt, Dalt), vega, wlen_ends, bin_width=bin_width,display_plots=True)

    # alt_reg = 58 + 9*60 + 33.6*3600
    # airmass = 1/np.sin(alt_reg*np.pi/180)
    
    # (reg_wlen,_), (reg_spec,_), _  = target.binning(bin=wlen[-1])

    # reg_spec *= np.exp(tau[0]*airmass) / target.get_exposure() * rfunc[0]
    print(tau)
    ratio = np.empty((0,len(rfunc[0])))
    x = 1/np.sin(alt*np.pi/180)
    fig,ax = plt.subplots(1,1)
    for sel in range(len(alt)):
        target = targets[sel]
        airmass = x[sel]
        (veg_wlen,Dveg_wlen), (veg_spec,Dveg_spec), _  = target.binning(bin=wlen[-1])
        veg_spec *= np.exp(tau[0]*airmass) / target.get_exposure() / rfunc[0]

        (std_wlen, std_Dwlen), (std_spec, std_Dspec) = spc.vega_std(wlen[-1],balmer_rem=True,diagn_plots=display_plots)

        ratio = np.append(ratio,[veg_spec/std_spec],axis=0)

        # plt.figure()
        # plt.errorbar(std_wlen,std_spec,std_Dspec,std_Dwlen,'.-')
        # display_lines(std_spec.min(),(std_wlen.min(),std_wlen.max()))
        # plt.show()

        # (std_wlen,_), (std_spec,_), _ = std.binning(bin=wlen[-1])

        plt.figure()
        plt.subplot(2,1,1)
        plt.title('Comparison')
        (tar_wlen,_), (tar_spec,_), _ = target.binning(bin=wlen[-1])
        plt.plot(tar_wlen,tar_spec,'.-')
        plt.subplot(2,1,2)
        plt.errorbar(veg_wlen,veg_spec,Dveg_spec,Dveg_wlen,'.-',label=f'data - {alt[sel]}')
        # plt.plot(reg_wlen,reg_spec,'.-')
        plt.plot(std_wlen,std_spec,'.-',label='std')
    plt.show()

    plt.figure()
    plt.plot(np.mean(ratio,axis=1),'.-')

    plt.figure()
    plt.title('Ratio $\\tau$')
    targets = np.array(targets)
    ratio_tau = np.array([np.log(targets[i+1].binning(bin=wlen[-1])[1][0]/targets[i].binning(bin=wlen[-1])[1][0])/(x[i]-x[i+1]) * targets[i].get_exposure()/targets[i+1].get_exposure() for i in range(len(targets)-1)])
    for i in range(ratio_tau.shape[0]):
        val = ratio_tau[i]
        plt.plot(val,label=f'$\\Delta$alt = {alt[i]:.3f}-{alt[i+1]:.3f}')
    plt.plot(tau[0],label='$\\tau$')
    plt.legend()

    plt.figure()
    for i in range(ratio_tau.shape[0]):
        val = ratio_tau[i]
        plt.plot(val-tau[0],label=f'$\\Delta$alt = {alt[i]:.3f}-{alt[i+1]:.3f}')
    plt.legend()

    plt.show()

    ## Regolo
    target, _ = spc.calibration(night,'Regolo',selection,other_lamp=lamp,display_plots=False)
    (bin_wlen,_), (bin_reg,_), _ = target.binning(bin=wlen[-1])
    alt_reg = 58 + 9/60 + 33.6/3600
    airmass = 1/np.sin(alt_reg*np.pi/180)
    print('AIRMASS',airmass)
    print('AIRMASS',x)
    texp = target.get_exposure()
    pos = np.argsort(x)
    factor = np.array([ CubicSpline(x[pos], ratio[pos,i])(airmass) for i in range(ratio.shape[1])])
    # for i in range(ratio.shape[1]):
    #     tmp = CubicSpline(x, ratio[:,i])
    #     factor = (tmp(airmass))
    pos = np.argmin(abs(x-airmass))
    abs_reg = bin_reg / texp / rfunc[0]  * np.exp(tau[0]*airmass)
    plt.figure()
    plt.title('Correction Factor')
    for r, xi in zip(ratio,x):
        plt.plot(bin_wlen,r,'.-',label=f'Veg {xi}')
    plt.plot(bin_wlen,factor,'.-',label=f'Reg {airmass}')
    plt.legend()

    probe = vega[pos].copy()
    _, (b_probe,_), _ = probe.binning(bin=wlen[-1])
    probe = b_probe / probe.get_exposure() / rfunc[0] * np.exp(tau[0]*x[pos])

    (std_wlen,_), (std_spec,_) = spc.vega_std(wlen[-1],balmer_rem=False)

    print(len(bin_wlen),len(abs_reg))
    plt.figure()
    plt.subplot(2,1,1)
    plt.title('Regolo Comparison')
    plt.plot(bin_wlen, bin_reg, '.-')
    plt.plot(bin_wlen, b_probe, '.-',color='green')
    plt.subplot(2,1,2)
    plt.plot(bin_wlen, abs_reg, '.-',label='Regolo')
    plt.plot(bin_wlen, probe, '.-',label='Regolo')
    plt.plot(std_wlen,std_spec,'.-',color='green',label='std of Vega')
    plt.xlim(wlen[-1].min(),wlen[-1].max())
    plt.legend()
    plt.show()
    # exit()
    # fig, (ax1,ax2,ax3) = plt.subplots(3,1)
    
    # ax1.plot(bin_wlen, factor,'.-',color='red')
    # ax2.plot(bin_wlen, rfunc[0],'.-',color='blue')
    # ax3.plot(bin_wlen, tau[0],'.-',color='violet')

    # factor = CubicSpline(bin_wlen, factor)(target.lines)
    # respon = CubicSpline(bin_wlen, rfunc[0])(target.lines)
    # tau_in = CubicSpline(bin_wlen, tau[0])(target.lines)
    # ax1.plot(target.lines,factor,'+--',color='red')
    # ax1.set_xlim(*wlen_ends)
    # ax2.plot(target.lines,respon,'+--',color='blue')
    # ax2.set_xlim(*wlen_ends)
    # ax3.plot(target.lines,tau_in,'+--',color='violet')
    # ax3.set_xlim(*wlen_ends)
    
    # abs_reg = target.spec / texp / respon / factor * np.exp(tau_in*airmass)

    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.title('Regolo Comparison')
    # plt.plot(target.lines, target.spec, '.-')
    # plt.xlim(*wlen_ends)
    # plt.subplot(2,1,2)
    # plt.plot(target.lines, abs_reg, '.-')
    # plt.xlim(*wlen_ends)
    # plt.show()
        # plt.figure()
        # plt.plot(wlen,std_spec-veg_spec)

        # rfunc = CubicSpline(wlen[0],rfunc[0])
        # tau = CubicSpline(wlen[0],tau[0])

        # veg_spec = target.spec * np.exp(tau(target.lines)*airmass) / target.get_exposure() / rfunc(target.lines)
        # # reg_spec = target.spec * np.exp(tau(target.lines)*airmass) / target.get_exposure() / rfunc(target.lines)


        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.title('Comparison Interpolating')
        # plt.plot(target.lines,target.spec,'.-')
        # plt.subplot(2,1,2)
        # plt.plot(target.lines,veg_spec,'.-')
        # # plt.plot(target.lines,reg_spec,'.-')
        # plt.plot(std_wlen,std_spec,'.-')
        # plt.show()
