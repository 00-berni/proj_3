import numpy as np
import matplotlib.pyplot as plt
import spectralpy as spc
from spectralpy.calcorr import remove_balmer
from scipy.interpolate import CubicSpline
import astropy.units as u
from numpy.typing import ArrayLike



def compute_pos(lat: str, lon: str):
    res = []
    for pos in [lat,lon]:
        pos = np.array(pos.split(' ')).astype(float)
        res += [(pos[0]+pos[1]/60+pos[2]/3600) * u.deg]
    return res

if __name__ == '__main__':
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time
    ## Data
    print('-- DATA --')
    night = '17-03-27'
    target_name = 'Vega'
    selection = 'mean'
    wlen_ends = (4500, 6700)
    obs_num = 3

    ## Wavelength Calibration
    print('-- WAVELENGTH CALIBRATION --')
    ord1 = 2
    ord2 = 3
    display_plots = False
    target, lamp = spc.calibration(night, target_name+'01', selection, norm=False, balmer_cal=False, ord_lamp=ord1, ord_balm=ord2, display_plots=display_plots)
    tmp = target.copy()
    spans = [tmp.span]
    targets = [target]
    for i in range(1,obs_num):
        tmp, _ = spc.calibration(night,target_name+f'0{i+1}',selection, norm=False, other_lamp=lamp, display_plots=False)
        spans += [tmp.span]
        targets += [tmp]

    # spectrum extraction
    alt = []
    Dalt = []
    vega = []
    wlen_gap = [
        [[4814,4920],[6520,6700]],
        [[4790,5000],[6238,6650]],
        [[4790,5000],[6238,6650]],
        [[4790,5000],[6238,6650]],
        [[4790,5000],[6238,6650]]
    ]
    min_span = np.min(spans)
    for i in range(len(targets)):
        tag = targets[i].copy()
        span = slice(tag.cen-min_span,tag.cen+min_span+1)
        data = tag.data[span].copy()
        tag.spec = np.sum(data,axis=0)
        cen_val = data[min_span]
        tag.std = np.mean([abs(cen_val - data[0]),abs(cen_val - data[-1])],axis=0)
        tag.spec = remove_balmer(tag.lines, tag.spec,wlen_gap=wlen_gap[i],display_plots=True,xlim=wlen_ends)
        vega += [tag]
        lat, lon = tag.header[0]['SITELAT'], tag.header[0]['SITELONG']
        lat, lon = compute_pos(lat,lon)
        print(tag.name,'\tPOS:',lat,lon)
        obs = EarthLocation(lat=lat,lon=lon)
        obj = SkyCoord.from_name('alf Lyr')
        estalt = []
        for h in tag.header:
            time = Time(h['DATE-OBS'])
            coord = obj.transform_to(AltAz(obstime=time,location=obs))
            print('ALT',coord.alt)
            estalt += [coord.alt.value]
        Destalt = (np.max(estalt)-np.min(estalt))/ 2
        estalt = np.mean(estalt)   
        print(tag.name,'\tALT:',estalt, Destalt)
        alt +=  [estalt]
        Dalt += [Destalt]
    fig1, ax1 = plt.subplots(1,1)
    fig2, ax2 = plt.subplots(1,1)
    for tg, veg, a in zip(targets,vega,alt):
        tmp_tgdata = tg.spectral_data(True) 
        tmp_data = veg.spectral_data(True)
        ax1.errorbar(*tmp_tgdata,fmt='.',linestyle='dashed',label=f'{a:.2f} deg')
        ax2.errorbar(*tmp_data,fmt='.',linestyle='dashed',label=f'{a:.2f} deg')
    ax1.legend()
    ax2.set_xlim(*wlen_ends)
    ax2.legend()
    plt.show()
    alt = np.array(alt)
    Dalt = np.array(Dalt)
    exit()

    print('-- RESPONSE FUNCTION --')
    # alt = alt[[0,1,3]]
    # Dalt = Dalt[[0,1,3]]
    # vega = [*vega[:2]]+[vega[3]]
    # alt = alt[:-1]
    # Dalt = Dalt[:-1]
    # vega = vega[:-1]

    bin_width = 50
    wlen, rfunc, tau = spc.ccd_response((alt, Dalt), vega, wlen_ends, bin_width=bin_width,display_plots=True,diagn_plots=True)

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
    target, _ = spc.calibration(night,'Regolo',selection,norm=False,other_lamp=lamp,display_plots=False)
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
