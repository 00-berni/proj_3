import numpy as np
import matplotlib.pyplot as plt
import spectralpy as spc

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


if __name__ == '__main__':
    ## Data
    night = '17-03-27'
    target_name = 'Vega'
    selection = 'mean'
    alt  = np.array([20,35,43,58],dtype=float)
    Dalt = np.full(alt.shape,0.5)

    ## Wavelength Calibration
    ord1 = 2
    ord2 = 3
    display_plots = True
    target, lamp = spc.calibration(night, target_name+'01', selection, ord_lamp=ord1, ord_balm=ord2, display_plots=display_plots,diagn_plots=False)

    ## Prepare Data
    bin_width = 50
    vega = [target]
    wlen_ends = [[target.lines[0],target.lines[-1]]]
    for i in range(1,len(alt)):
        tmp, _ = spc.calibration(night,target_name+f'0{i+1}',selection, other_lamp=lamp, display_plots=False)
        vega += [tmp]
        wlen_ends += [[tmp.lines[0],tmp.lines[-1]]]
        plt.figure()
        plt.errorbar(*tmp.spectral_data(True),fmt='.-')
        display_lines(tmp.spec.min(),(tmp.lines.min(),tmp.lines.max()))
        plt.show()

    ## Response Function
    wlen, rfunc, tau = spc.ccd_response((alt, Dalt), vega, wlen_ends, bin_width=bin_width,display_plots=True)

    # alt_reg = 58 + 9*60 + 33.6*3600
    # airmass = 1/np.sin(alt_reg*np.pi/180)
    
    # (reg_wlen,_), (reg_spec,_), _  = target.binning(bin=wlen[-1])

    # reg_spec *= np.exp(tau[0]*airmass) / target.get_exposure() * rfunc[0]

    sel = 0
    target = vega[sel]
    airmass = 1/np.sin(alt[sel]*np.pi/180)
    (veg_wlen,Dveg_wlen), (veg_spec,Dveg_spec), _  = target.binning(bin=wlen[-1])
    veg_spec *= np.exp(tau[0]*airmass) / target.get_exposure() / rfunc[0]

    std = spc.vega_std()

    plt.figure()
    plt.errorbar(*std.spectral_data(True))
    display_lines(std.spec.min(),(std.lines.min(),std.lines.max()))
    plt.show()

    (std_wlen,_), (std_spec,_), _ = std.binning(bin=wlen[-1])

    plt.figure()
    plt.subplot(2,1,1)
    plt.title('Comparison')
    (tar_wlen,_), (tar_spec,_), _ = target.binning(bin=wlen[-1])
    plt.plot(tar_wlen,tar_spec,'.-')
    plt.subplot(2,1,2)
    plt.errorbar(veg_wlen,veg_spec,Dveg_spec,Dveg_wlen,'.-')
    # plt.plot(reg_wlen,reg_spec,'.-')
    plt.plot(std_wlen,std_spec,'.-')

    # plt.figure()
    # plt.plot(wlen,std_spec-veg_spec)

    from scipy.interpolate import CubicSpline
    rfunc = CubicSpline(wlen[0],rfunc[0])
    tau = CubicSpline(wlen[0],tau[0])

    veg_spec = target.spec * np.exp(tau(target.lines)*airmass) / target.get_exposure() / rfunc(target.lines)
    # reg_spec = target.spec * np.exp(tau(target.lines)*airmass) / target.get_exposure() / rfunc(target.lines)


    plt.figure()
    plt.subplot(2,1,1)
    plt.title('Comparison Interpolating')
    plt.plot(target.lines,target.spec,'.-')
    plt.subplot(2,1,2)
    plt.plot(target.lines,veg_spec,'.-')
    # plt.plot(target.lines,reg_spec,'.-')
    plt.plot(std.lines,std.spec,'.-')
    plt.show()
