import numpy as np
import matplotlib.pyplot as plt
import spectralpy as spc
import astropy.units as u
from spectralpy.calcorr import remove_balmer
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

def compute_pos(lat: str, lon: str):
    res = []
    for pos in [lat,lon]:
        pos = np.array(pos.split(' ')).astype(float)
        res += [(pos[0]+pos[1]/60+pos[2]/3600) * u.deg]
    return res

if __name__ == '__main__':

    ### CONSTANTS
    FONTSIZE = 18
    NIGHT = '17-03-27'          #: selected observation
    TARGET_NAME = 'Vega'        #: selected target
    SELECTION = 0               #: selected acquisition
    WLEN_ENDS = (4500,6490)     #: selected wavelength range
    # define a method to select values in set wavelength range
    wl_lim = lambda wlen : (wlen >= WLEN_ENDS[0]) & (wlen <= WLEN_ENDS[-1])

    ### WAVELENGTH CALIBRATION
    print('-- WAVELENGTH CALIBRATION --')
    ord1 = 2
    ord2 = 3
    display_plots = False
    diagn_plots = False
    # compute the calibration
    cal_target, cal_lamp = spc.calibration(NIGHT, TARGET_NAME+'01','mean', norm=False, balmer_cal=False, ord_lamp=ord1, ord_balm=ord2, display_plots=display_plots,diagn_plots=diagn_plots)

    ### DATA
    # initialize variables
    targets: list[spc.Spectrum] = []    #: list of data
    vega:    list[spc.Spectrum] = []    #: list of spectra after balmer's removal
    spans = []                          #: list of each extraction region width
    # collect data
    for i in [2,3]:
        tmp, _ = spc.calibration(NIGHT,TARGET_NAME+f'0{i}',SELECTION, norm=False, other_lamp=cal_lamp, display_plots=False,diagn_plots=diagn_plots)
        # store data
        spans   += [tmp.span]
        targets += [tmp.copy()]

    ## Spectrum Extraction
    wlen_gap = [                #: Balmer's lines range for each observation
        [[4805,4924],[6520,6690]],
        [[4790,4910],[6520,6690]]
    ]
    alt  = np.array([])         #: array of altitudes
    Dalt = np.array([])         #: array of the uncertainty in them 
    min_span = np.min(spans)    #: minimum width
    # remove Balmer's lines, compute airmass and store data
    for i in range(len(targets)):
        tag = targets[i].copy()
        # take data inside a section wide `min_span`
        span = slice(tag.cen-min_span,tag.cen+min_span+1)
        data = tag.data[span].copy()
        tag.spec = np.sum(data,axis=0)
        # set the uncertainty to semi-dispersion
        cen_val = data[min_span]
        tag.std = np.mean([abs(cen_val - data[0]),abs(cen_val - data[-1])],axis=0)
        # update data
        targets[i] = tag.copy()
        # remove Balmer's lines
        tag.spec = remove_balmer(tag.lines, tag.spec,wlen_gap=wlen_gap[i],display_plots=display_plots,xlim=WLEN_ENDS)
        vega += [tag.copy()]
        # extract observation site information
        lat = tag.header['SITELAT'] 
        lon = tag.header['SITELONG']
        lat, lon = compute_pos(lat,lon)
        print(tag.name,'\tPOS:',lat,lon)
        obs = EarthLocation(lat=lat,lon=lon)
        # compute sky coordinates of Vega
        obj = SkyCoord.from_name('alf Lyr')
        # compute the altitude on the observation date
        time = Time(tag.header['DATE-OBS'])
        coord = obj.transform_to(AltAz(obstime=time,location=obs))
        print('ALT',coord.alt)
        estalt = coord.alt.value
        # set an uncertainty a priori
        Destalt = 0.01
        print(tag.name,'\tALT:',estalt, Destalt)
        spc.print_measure(estalt,Destalt,'ALT','deg')
        # store the results
        alt  = np.append(alt,[estalt])
        Dalt = np.append(Dalt,[Destalt])
    # plot
    fig1, ax1 = plt.subplots(1,1)
    fig2, ax2 = plt.subplots(1,1)
    ax1.set_title('Selected data',fontsize=FONTSIZE+2)
    ax2.set_title("Balmer's lines removal",fontsize=FONTSIZE+2)
    for tg, veg, a in zip(targets,vega,alt):
        tmp_tgdata = tg.spectral_data(True) 
        tmp_data = veg.spectral_data(True)
        ax1.errorbar(*tmp_tgdata,fmt='.',linestyle='dashed',label=f'{a:.2f} deg')
        ax2.errorbar(*tmp_data,fmt='.',linestyle='dashed',label=f'{a:.2f} deg')
    ax1.grid(linestyle='--',alpha=0.2,color='grey')
    ax1.set_xlabel('$\\lambda$ [$\\AA$]',fontsize=FONTSIZE)
    ax1.set_ylabel('counts',fontsize=FONTSIZE)
    ax1.legend(fontsize=FONTSIZE)
    ax2.grid(linestyle='--',alpha=0.2,color='grey')
    ax2.set_xlabel('$\\lambda$ [$\\AA$]',fontsize=FONTSIZE)
    ax2.set_ylabel('counts',fontsize=FONTSIZE)
    ax2.legend(fontsize=FONTSIZE)
    ax2.set_xlim(*WLEN_ENDS)
    plt.show()


    ### RESPONSE FUNCTION
    print('-- RESPONSE FUNCTION --')
    ## Bin data
    BIN_WIDTH = 50      #: width of each bin
    # compute the airmass on the observation date
    x  = 1/np.sin(alt*np.pi/180)
    Dx = Dalt * np.cos(alt*np.pi/180) * x**2 * np.pi/180 
    for airmass, Dairmass in zip(x,Dx):
        spc.print_measure(airmass,Dairmass,'X')
    # bin data
    unbin_wlen  = []        #: list of data wavelength range
    unbin  = []             #: list of spectra normalized by exposure time
    ydata  = []             #: list of binned spectral data normalized by exposure time
    Dydata = []             #: list of the uncertainty in them
    for obs in vega:
        obs = obs.copy()        
        exp_time = obs.get_exposure()
        obs.spec = obs.spec / exp_time
        # store unbinned data
        unbin_wlen += [obs.lines[wl_lim(obs.lines)]]
        unbin += [obs.spec[wl_lim(obs.lines)]]
        # bin data
        (wlen, Dwlen), (bin_spec, Dbin_spec), bins = obs.binning(bin=BIN_WIDTH,edges=WLEN_ENDS)
        # store bin data
        ydata  += [bin_spec]
        Dydata += [Dbin_spec]

    plt.figure()
    plt.title('Binned spectra normilized by the exposure time',fontsize=FONTSIZE+2)
    # plt.errorbar(wlen,ydata[0],Dydata[0],fmt='.--',label=f'$X = {x[0]:.3f}$')    
    # plt.errorbar(wlen,ydata[1],Dydata[1],fmt='.--',label=f'$X = {x[1]:.3f}$')
    plt.errorbar(wlen,ydata[0],fmt='.--',label=f'$X = {x[0]:.3f}$')    
    plt.errorbar(wlen,ydata[1],fmt='.--',label=f'$X = {x[1]:.3f}$')
    plt.grid(True,which='both',axis='x',linestyle='dotted',alpha=0.6)
    plt.xticks(bins,bins.astype(int),rotation=45)
    plt.legend(fontsize=FONTSIZE)
    plt.ylabel('$N_k\\; / \\;t_{exp}$ [counts s$^{-1}$]', fontsize=FONTSIZE)
    plt.xlabel('$\\lambda_k$ [$\\AA$]', fontsize=FONTSIZE)
    plt.show()    

    ## Compute tau
    # Ii = exp(-tau xi) S R
    # I1/I2 = exp(-tau (x1-x2))
    # ln(I1/I2) = -tau * (x1-x2)
    # tau = ln(I1/I2)/(x2-x1)
    # Dtau = [DI1/I1 + DI2/I2 + ln(I1/I2)/(x2-x1)*(Dx1+Dx2)]/(x2-x1) 
    tau  = np.log(ydata[1]/ydata[0]) / (x[0]-x[1])
    Dtau = (Dydata[1]/ydata[1] + Dydata[0]/ydata[0] + tau*(Dx[1]+Dx[0]))/(x[0]-x[1])
    plt.figure()
    plt.errorbar(wlen,tau,Dtau,fmt='.--')
    plt.show()

    # fit an exponential to `tau` data
    def fit_func(x, *params):
        b,c,d = params
        return np.exp(b*(x-c)) + d
    initial_values = [-1e-4,wlen[1],tau.min()]
    # initial_values = [-1e-4,6270,tau.min()]
    # fit = spc.FuncFit(xdata=wlen,ydata=tau,yerr=Dtau)
    fit = spc.FuncFit(xdata=wlen,ydata=tau)
    fit.pipeline(fit_func,initial_values)
    fit.plot(mode='subplots',plot1={'label':'data'},plot2={'label':'fit'},plotargs={'title':'Estimated Optical Depth','fontsize':FONTSIZE,'ylabel':'$\\tau_k$'},fontsize=FONTSIZE,xlabel='$\\lambda_k$ [$\\AA$]')
    plt.figure()
    plt.hist(fit.residuals(),15)
    plt.show()    


    ## 0-Airmass
    # compute binned 0-airmass spectrum
    # ln(I) = -tau x + ln(SR)
    # ln(SR) = ln(Ii) + tau xi
    Sigma = [ydata[i]*np.exp(tau*x[i]) for i in range(2) ]
    DSigma = abs(Sigma[1]-Sigma[0])/2
    Sigma = np.mean(Sigma,axis=0)
    plt.figure()
    plt.title('')
    plt.errorbar(wlen,Sigma,DSigma,fmt='.--')
    plt.figure()
    plt.errorbar(wlen,Sigma,DSigma,fmt='.--')
    plt.plot(wlen,ydata[0],'.--') 
    plt.plot(wlen,ydata[1],'.--')
    plt.show() 

    # compute unbinned 0-airmass spectrum
    ends = (max(unbin_wlen[0][0],unbin_wlen[1][0]),min(unbin_wlen[0][-1],unbin_wlen[1][-1]))
    for i in range(2):
        wlen_i = unbin_wlen[i] 
        sel_pos = (wlen_i >= ends[0]) & (wlen_i <= ends[1])
        unbin_wlen[i] = wlen_i[sel_pos]
        unbin[i] = unbin[i][sel_pos]
    unbin_wlen = unbin_wlen[0]
    unbin_S = np.mean([unbin[i] * np.exp(fit_func(unbin_wlen,*fit.fit_par)*x[i]) for i in range(2)],axis=0)
    plt.figure()
    plt.title('Estimated 0-airmass spectrum',fontsize=FONTSIZE+2)
    plt.plot(unbin_wlen,unbin_S,'.--')
    plt.xlabel('$\\lambda$ [$\\AA$]',fontsize=FONTSIZE)
    plt.ylabel('$\\Sigma_\\lambda$ [counts s$^{-1}$]',fontsize=FONTSIZE)
    plt.grid(linestyle='--',alpha=0.2,color='grey')
    # plt.errorbar(wlen,Sigma,DSigma,fmt='.--')
    
    # bin the 0-airmass spectrum
    (wlen,Dwlen),(Sigma, DSigma), bins = spc.binning(unbin_S,unbin_wlen,bins)
    # plt.errorbar(wlen,Sigma,DSigma,fmt='.--',color='green')
    plt.show()

    ## Filters
    # from speclite import filters

    # blue = filters.load_filter('bessell-B')
    # visible = filters.load_filter('bessell-V')
    # red = filters.load_filter('bessell-R')
    # filters.plot_filters(filters.load_filters('bessell-B','bessell-V','bessell-R'))
    # plt.plot(wlen,Sigma)
    # plt.show()
    # index_col = []
    # for filt in [blue,visible,red]:
    #     tmp_wlen = wlen.copy()
    #     tmp_data = Sigma.copy()
    #     filt_wlen = filt.wavelength.copy()
    #     prv_pos = np.where(tmp_wlen <= filt_wlen[0])[0]
    #     if len(prv_pos) == 0:
    #         bin_diff = (wlen[0]-filt_wlen[0])
    #         bin_diff = bin_diff//BIN_WIDTH if bin_diff%BIN_WIDTH==0 else bin_diff//BIN_WIDTH+1
    #         bin_diff = bin_diff.astype(int)
    #         tmp_wlen = np.append(np.linspace(wlen[0]-bin_diff*BIN_WIDTH,wlen[0]-BIN_WIDTH,bin_diff),tmp_wlen)
    #         tmp_data = np.append([0]*bin_diff,tmp_data)
    #         print('Prev',wlen.shape,tmp_wlen.shape,tmp_data.shape)
    #     else:
    #         tmp_wlen = tmp_wlen[prv_pos[-1]:]
    #         tmp_data = tmp_data[prv_pos[-1]:]
    #     fll_pos = np.where(tmp_wlen >= filt_wlen[-1])[0]
    #     if len(fll_pos) == 0:
    #         bin_diff = (filt_wlen[-1]-wlen[-1])
    #         bin_diff = bin_diff//BIN_WIDTH if bin_diff%BIN_WIDTH==0 else bin_diff//BIN_WIDTH+1
    #         bin_diff = bin_diff.astype(int)
    #         tmp_wlen = np.append(tmp_wlen,np.linspace(wlen[-1]+BIN_WIDTH,wlen[-1]+BIN_WIDTH*bin_diff,bin_diff))
    #         tmp_data = np.append(tmp_data,[0]*bin_diff)
    #         print('Foll',wlen[-1],tmp_wlen[-1],filt_wlen[-1])
    #     else:
    #         tmp_wlen = tmp_wlen[:fll_pos[0]+1]
    #         tmp_data = tmp_data[:fll_pos[0]+1]
        
    #     index_col += [filt.convolve_with_array(tmp_wlen,tmp_data)]
    # print(index_col)
    # blue, visible, red = index_col 
    # print(np.diff(index_col))
    # print(blue-red)

    ## Compute the Response
    # bin standard spectrum and remove Balmer's lines
    (_,_),(std_spec,_) = spc.vega_std(bins)
    response = Sigma/std_spec       #: binned response function
    plt.figure()
    plt.title('Binned Standard Spectrum',fontsize=FONTSIZE+2)
    plt.plot(wlen,std_spec,'.--')
    plt.xlabel('$\\lambda_k$ [$\\AA$]',fontsize=FONTSIZE)
    plt.ylabel('$S_{k}^{std}$ [erg cm$^{-2}$ s$^{-1}$ $\\AA^{-1}$]',fontsize=FONTSIZE)
    plt.grid(True,which='both',axis='x',linestyle='dotted',alpha=0.6)
    plt.xticks(bins,bins.astype(int),rotation=45)
    plt.figure()
    plt.title('Binned Response Function',fontsize=FONTSIZE+2)
    plt.plot(wlen,response,'.--')
    plt.xlabel('$\\lambda_k$ [$\\AA$]',fontsize=FONTSIZE)
    plt.ylabel('$R_k$ [counts cm$^{2}$ $\\AA$ erg$^{-1}$]',fontsize=FONTSIZE)
    plt.grid(True,which='both',axis='x',linestyle='dotted',alpha=0.6)
    plt.xticks(bins,bins.astype(int),rotation=45)

    from scipy.interpolate import CubicSpline
    # keep Balmer's lines
    _, (std_spec,_) = spc.vega_std(bins,balmer_rem=False)
    int_response = CubicSpline(wlen,response)       #: interpolator for the response function

    ### FROM STD TO COUNTS
    for obs,airmass in zip(targets,x):
        est_tau = fit.method(wlen)
        exp_time = obs.get_exposure()
        rec_spec = np.exp(-est_tau*airmass)*std_spec*exp_time*response
        _, (obs_sp,_),_ = obs.binning(bins)
        plt.figure(figsize=(10,13))
        plt.subplot(2,1,1)
        plt.plot(wlen,obs_sp,'.--')
        plt.plot(wlen,rec_spec,'x--')
        plt.subplot(2,1,2)
        plt.plot(wlen,(obs_sp-rec_spec)/obs_sp*100,'.--')

    plt.show()  

    ### FROM COUNTS TO STD
    # compute the averaged absolute spectrum
    avg_abssp = 0
    for obs, airmass in zip(targets,x):
        wave = obs.lines[wl_lim(obs.lines)]
        sp_data = obs.spec[wl_lim(obs.lines)]
        # sel_pos = (obs.lines >= targets[1].lines[0]) & (obs.lines <= targets[1].lines[-1])
        # wave = obs.lines[sel_pos]
        # sp_data = obs.spec[sel_pos]
        ext_tau = fit.method(wave)
        exp_time = obs.get_exposure()
        rec_sp  = sp_data * np.exp(ext_tau*airmass) / int_response(wave) / exp_time
        avg_abssp += rec_sp
    avg_abssp /= 2

    # get the standard
    std_wl, std_sp = spc.get_standard()
    std_sp = std_sp[wl_lim(std_wl)]
    std_wl = std_wl[wl_lim(std_wl)]
    plt.figure(figsize=(10,13))
    plt.title('Absolute calibration',fontsize=FONTSIZE+2)
    plt.plot(std_wl,std_sp,'.--',label='standard')
    plt.plot(wave,avg_abssp,'.--',label='calibrated')
    plt.grid(linestyle='--',alpha=0.2,color='grey')
    plt.legend(fontsize=FONTSIZE)
    plt.xlabel('$\\lambda$ [$\\AA$]',fontsize=FONTSIZE)
    plt.ylabel('$S_\\lambda$ [erg cm$^{-2}$ s$^{-1}$ $\\AA^{-1}$]',fontsize=FONTSIZE)
    plt.show()

    # bin data to compare them
    (std_wl,_),(std_sp,_), _ = spc.binning(std_sp,std_wl,bins)
    (bin_wl,_),(bin_sp,_), _ = spc.binning(avg_abssp,wave,bins)
    # (std_wl,_),(std_sp,_), _ = spc.binning(std_sp,std_wl,edges=(targets[1].lines[0],targets[1].lines[-1]))
    # (bin_wl,_),(bin_sp,_), _ = spc.binning(avg_abssp,wave,edges=(targets[1].lines[0],targets[1].lines[-1]))
    norm_diff = (bin_sp-std_sp)/std_sp*100
    avg_diff = np.mean(norm_diff[bin_wl>5000][:-1])
    print(norm_diff.max())
    print(abs(avg_diff))
    print(norm_diff[bin_wl>5000][:-1].max(),norm_diff[bin_wl>5000][:-1].min())
    plt.figure(figsize=(10,13))
    plt.subplot(2,1,1)
    plt.title('Absolute Calibration: comparison',fontsize=FONTSIZE+2)
    plt.plot(std_wl,std_sp,'.--',label='standard')
    plt.plot(bin_wl,bin_sp,'.--',label='calibrated')
    plt.grid(True,which='both',axis='x',linestyle='dotted',alpha=0.6)
    plt.legend(fontsize=FONTSIZE)
    plt.ylabel('$S_{k}$ [erg cm$^{-2}$ s$^{-1}$ $\\AA^{-1}$]',fontsize=FONTSIZE)
    plt.xticks(bins,['']*len(bins),rotation=45)
    plt.subplot(2,1,2)
    plt.ylabel('$(S_{k}-S_{k}^{std})/S_{k}^{std}$ [%]',fontsize=FONTSIZE)
    plt.xlabel('$\\lambda_k$ [$\\AA$]',fontsize=FONTSIZE)
    plt.grid(True,which='both',axis='x',linestyle='dotted',alpha=0.6)
    plt.xticks(bins,bins.astype(int),rotation=45)
    plt.plot(bin_wl,norm_diff,'.--')
    # plt.axhline(norm_diff[bin_wl>5000][:-1].max(),0,1)
    # plt.axhline(norm_diff[bin_wl>5000][:-1].min(),0,1)
    plt.show()
