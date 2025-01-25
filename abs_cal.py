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
    NIGHT = '17-03-27'
    TARGET_NAME = 'Vega'
    SELECTION = 0
    WLEN_ENDS = (4500,6490)
    wl_lim = lambda wlen : (wlen >= WLEN_ENDS[0]) & (wlen <= WLEN_ENDS[-1])

    ### WAVELENGTH CALIBRATION
    print('-- WAVELENGTH CALIBRATION --')
    ord1 = 2
    ord2 = 3
    display_plots = False
    diagn_plots = False
    cal_target, cal_lamp = spc.calibration(NIGHT, TARGET_NAME+'01','mean', norm=False, balmer_cal=False, ord_lamp=ord1, ord_balm=ord2, display_plots=display_plots,diagn_plots=diagn_plots)

    ### DATA
    # initialize variables
    targets: list[spc.Spectrum] = []    #: list of data
    vega:    list[spc.Spectrum] = []    #: list of spectra after balmer's removal
    spans = []  #: list of each extraction region width
    alt  = np.array([])  #: array of altitudes
    Dalt = np.array([])  #: array of the uncertainty in them 
    # collect data
    for i in [2,3]:
        tmp, _ = spc.calibration(NIGHT,TARGET_NAME+f'0{i}',SELECTION, norm=False, other_lamp=cal_lamp, display_plots=False,diagn_plots=diagn_plots)
        # store data
        spans   += [tmp.span]
        targets += [tmp.copy()]

    ## Spectrum Extraction
    wlen_gap = [
        [[4805,4924],[6520,6690]],
        [[4790,4910],[6520,6690]]
    ]
    min_span = np.min(spans)    #: minimum width
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
        if isinstance(tag.header,list):
            lat = tag.header[0]['SITELAT'] 
            lon = tag.header[0]['SITELONG']
        else:
            lat = tag.header['SITELAT'] 
            lon = tag.header['SITELONG']
        lat, lon = compute_pos(lat,lon)
        print(tag.name,'\tPOS:',lat,lon)
        obs = EarthLocation(lat=lat,lon=lon)
        # compute sky coordinates of Vega
        obj = SkyCoord.from_name('alf Lyr')
        # compute the altitude and the airmass on the observation date
        if isinstance(tag.header,list):
            estalt = []
            for h in tag.header:
                time = Time(h['DATE-OBS'])
                coord = obj.transform_to(AltAz(obstime=time,location=obs))
                print('ALT',coord.alt)
                estalt += [coord.alt.value]
            Destalt = (np.max(estalt)-np.min(estalt))/ 2
            estalt = np.mean(estalt)   
        else:
            time = Time(tag.header['DATE-OBS'])
            coord = obj.transform_to(AltAz(obstime=time,location=obs))
            print('ALT',coord.alt)
            estalt = coord.alt.value
            # set an uncertainty a priori
            Destalt = 0.02
        print(tag.name,'\tALT:',estalt, Destalt)
        # store the results
        alt  = np.append(alt,[estalt])
        Dalt = np.append(Dalt,[Destalt])

    fig1, ax1 = plt.subplots(1,1)
    fig2, ax2 = plt.subplots(1,1)
    for tg, veg, a in zip(targets,vega,alt):
        tmp_tgdata = tg.spectral_data(True) 
        tmp_data = veg.spectral_data(True)
        ax1.errorbar(*tmp_tgdata,fmt='.',linestyle='dashed',label=f'{a:.2f} deg')
        ax2.errorbar(*tmp_data,fmt='.',linestyle='dashed',label=f'{a:.2f} deg')
    ax1.legend()
    ax2.set_xlim(*WLEN_ENDS)
    ax2.legend()
    plt.show()


    ### RESPONSE FUNCTION
    print('-- RESPONSE FUNCTION --')

    BIN_WIDTH = 50
    # wlen, rfunc, tau = spc.ccd_response((alt, Dalt), vega, WLEN_ENDS, bin_width=bin_width,display_plots=True,diagn_plots=True)
    x  = 1/np.sin(alt*np.pi/180)
    Dx = Dalt * np.cos(alt*np.pi/180) * x**2 * np.pi/180 
    unbin_wlen  = []
    unbin  = []
    ydata  = []
    Dydata = []
    for obs in vega:
        obs = obs.copy()        
        exp_time = obs.get_exposure()
        obs.spec = obs.spec / exp_time
        unbin_wlen += [obs.lines[wl_lim(obs.lines)]]
        unbin += [obs.spec[wl_lim(obs.lines)]]
        obs.std  = obs.std  / exp_time if obs.std is not None else None
        (wlen, Dwlen), (bin_spec, Dbin_spec), bins = obs.binning(bin=BIN_WIDTH,edges=WLEN_ENDS)
        ydata  += [bin_spec]
        Dydata += [Dbin_spec]

    plt.figure()
    plt.errorbar(wlen,ydata[0],Dydata[0],fmt='.--',label=f'$X = {x[0]:.3f}$')    
    plt.errorbar(wlen,ydata[1],Dydata[1],fmt='.--',label=f'$X = {x[1]:.3f}$')
    plt.grid(True,which='both',axis='x')
    plt.legend(fontsize=FONTSIZE)
    plt.show()    


    # Ii = exp(-tau xi) S R
    # I1/I2 = exp(-tau (x1-x2))
    # ln(I1/I2) = -tau * (x1-x2)
    # tau = ln(I1/I2)/(x2-x1)
    # Dtau = [DI1/I1 + DI2/I2 + ln(I1/I2)/(x2-x1)*(Dx1+Dx2)]/(x2-x1) 
    tau  = np.log(ydata[1]/ydata[0]) / (x[0]-x[1])
    Dtau = (Dydata[1]/ydata[1] + Dydata[0]/ydata[0] + tau*(Dx[1]+Dx[0]))/(x[0]-x[1])
    plt.figure()
    plt.errorbar(wlen,tau,Dtau,fmt='.--')
    # plt.show()

    def fit_func(x, *params):
        b,c,d = params
        return np.exp(b*(x-c)) + d
    initial_values = [-1e-4,wlen[1],tau.min()]
    fit = spc.FuncFit(xdata=wlen,ydata=tau,yerr=Dtau)
    fit.pipeline(fit_func,initial_values)
    fit.plot(mode='subplots')
    plt.figure()
    plt.hist(fit.residuals(),15)
    plt.show()    


    # ln(I) = -tau x + ln(SR)
    # ln(SR) = ln(Ii) + tau xi

    Sigma = [ydata[i]*np.exp(tau*x[i]) for i in range(2) ]
    DSigma = abs(Sigma[1]-Sigma[0])/2
    Sigma = np.mean(Sigma,axis=0)
    plt.errorbar(wlen,Sigma,DSigma,fmt='.--')
    plt.figure()
    plt.errorbar(wlen,Sigma,DSigma,fmt='.--')
    plt.plot(wlen,ydata[0],'.--') 
    plt.plot(wlen,ydata[1],'.--')
    plt.show() 


    ends = (max(unbin_wlen[0][0],unbin_wlen[1][0]),min(unbin_wlen[0][-1],unbin_wlen[1][-1]))
    for i in range(2):
        wlen_i = unbin_wlen[i] 
        sel_pos = (wlen_i >= ends[0]) & (wlen_i <= ends[1])
        unbin_wlen[i] = wlen_i[sel_pos]
        unbin[i] = unbin[i][sel_pos]
    unbin_wlen = unbin_wlen[0]
    unbin_S = np.mean([unbin[i] * np.exp(fit_func(unbin_wlen,*fit.fit_par)*x[i]) for i in range(2)],axis=0)
    plt.figure()
    plt.plot(unbin_wlen,unbin_S,'.--')
    plt.errorbar(wlen,Sigma,DSigma,fmt='.--')

    (wlen,Dwlen),(Sigma, DSigma), bins = spc.binning(unbin_S,unbin_wlen,bins)
    plt.errorbar(wlen,Sigma,DSigma,fmt='.--',color='green')
    plt.show()


    from speclite import filters

    blue = filters.load_filter('bessell-B')
    visible = filters.load_filter('bessell-V')
    red = filters.load_filter('bessell-R')
    filters.plot_filters(filters.load_filters('bessell-B','bessell-V','bessell-R'))
    plt.plot(wlen,Sigma)
    plt.show()
    index_col = []
    for filt in [blue,visible,red]:
        tmp_wlen = wlen.copy()
        tmp_data = Sigma.copy()
        filt_wlen = filt.wavelength.copy()
        prv_pos = np.where(tmp_wlen <= filt_wlen[0])[0]
        if len(prv_pos) == 0:
            bin_diff = (wlen[0]-filt_wlen[0])
            bin_diff = bin_diff//BIN_WIDTH if bin_diff%BIN_WIDTH==0 else bin_diff//BIN_WIDTH+1
            bin_diff = bin_diff.astype(int)
            tmp_wlen = np.append(np.linspace(wlen[0]-bin_diff*BIN_WIDTH,wlen[0]-BIN_WIDTH,bin_diff),tmp_wlen)
            tmp_data = np.append([0]*bin_diff,tmp_data)
            print('Prev',wlen.shape,tmp_wlen.shape,tmp_data.shape)
        else:
            tmp_wlen = tmp_wlen[prv_pos[-1]:]
            tmp_data = tmp_data[prv_pos[-1]:]
        fll_pos = np.where(tmp_wlen >= filt_wlen[-1])[0]
        if len(fll_pos) == 0:
            bin_diff = (filt_wlen[-1]-wlen[-1])
            bin_diff = bin_diff//BIN_WIDTH if bin_diff%BIN_WIDTH==0 else bin_diff//BIN_WIDTH+1
            bin_diff = bin_diff.astype(int)
            tmp_wlen = np.append(tmp_wlen,np.linspace(wlen[-1]+BIN_WIDTH,wlen[-1]+BIN_WIDTH*bin_diff,bin_diff))
            tmp_data = np.append(tmp_data,[0]*bin_diff)
            print('Foll',wlen[-1],tmp_wlen[-1],filt_wlen[-1])
        else:
            tmp_wlen = tmp_wlen[:fll_pos[0]+1]
            tmp_data = tmp_data[:fll_pos[0]+1]
        
        index_col += [filt.convolve_with_array(tmp_wlen,tmp_data)]
    print(index_col)
    blue, visible, red = index_col 
    print(np.diff(index_col))
    print(blue-red)

    (_,_),(std_spec,_) = spc.vega_std(bins)
    plt.plot(wlen,std_spec,'.--')
    response = Sigma/std_spec
    plt.figure()
    plt.plot(wlen,response,'.--')

    from scipy.interpolate import CubicSpline

    _, (std_spec,_) = spc.vega_std(bins,balmer_rem=False)

    int_response = CubicSpline(wlen,response)

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