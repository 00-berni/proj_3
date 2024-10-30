import numpy as np
import matplotlib.pyplot as plt
import spectralpy as spc
from astropy import units as u
from scipy.signal import correlate
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
    lag = 15
    jupiter, lamp = spc.get_target_data(obs_night,target_name,selection,angle=None,lim_width=lim_width,lag=lag,gauss_corr=False,lamp_incl=False, fit_args=fit_args, diagn_plots=False,norm='log')
    
    ## VELOCITY ESTIMATION
    data = jupiter.data.copy()[5:] 

    plt.figure()
    for i in range(data.shape[1])[::5]:
        plt.plot(data[:,i])
    plt.show()


    v  = np.array([])
    Dv = np.array([])
    s  = np.array([])
    Ds = np.array([])
    fig, ax = plt.subplots(2,1)
    xdata = np.arange(data.shape[0])
    N = data.shape[1]
    step = 10
    xx = np.arange(N)[::step]
    for i in xx:
        ydata = data[:,i]
        fit = spc.FuncFit(xdata=xdata,ydata=ydata,xerr=1)
        fit.pol_fit(2,[-0.2,1,1],mode='curve_fit')
        color1 = (i/N,0,0.5)
        color2 = (i/N,1-i/N,0.5)
        fit.data_plot(ax[0],pltarg1={'color':color1},pltarg2={'color':color2})
        fit.residuals_plot(ax[1],color=color1)
        a,b,c = fit.fit_par
        cov = fit.res['cov']
        delta = b**2 - 4*a*c
        der = [ 4*c/(4*a) + delta/(4*a**2),
                -2*b/(4*a),
                1 ]
        err = np.sqrt(np.sum([der[j]*der[k]*cov[j,k] for k in range(len(der)) for j in range(len(der))]))
        s  = np.append(s,-delta/4/a)
        Ds = np.append(Ds,err)
        der = [ b/(2*a**2),
                -1/(2*a) ]
        err = np.sqrt(np.sum([der[j]*der[k]*cov[j,k] for k in range(len(der)) for j in range(len(der))]))
        v  = np.append(v,-b/2/a)
        Dv = np.append(Dv,err)
    # plt.show()
    plt.figure()
    plt.imshow(data,origin='lower',norm='log',cmap='gray_r')
    plt.errorbar(xx,v,Dv,fmt='.-')
    plt.show()


    fit = spc.FuncFit(xdata=xx,ydata=v,yerr=Dv)
    fit.linear_fit([0,v.mean()],mode='curve_fit')
    fit.plot(mode='subplots')
    plt.show()
    m = fit.fit_par[0]
    angle = np.arctan(m) * 180/np.pi
    print(angle)

    jupiter.data = jupiter.hdul[0].data.copy()
    jupiter = jupiter.rotate_target(jupiter.angle + angle)


    lim0 = jupiter.lims[0]
    angle *= np.pi/180
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle),np.cos(angle)] ])
    _,p1  = np.array([rot_mat.dot([xx[i],v[i]]) for i in range(len(xx))]).mean(axis=0)
    print('mid',p1)

    uncut_data = jupiter.data.copy()
    jupiter.lims = [626,730,42,1391]
    jupiter.cut_image()
    data = jupiter.data.copy()

    v  = []
    Dv = []
    r  = []
    Dr = []
    tmp = []
    fig, ax = plt.subplots(2,1)
    fig0, ax0 = plt.subplots(1,1)
    xdata = np.arange(data.shape[0])
    N = data.shape[1]
    step = 10
    xx = np.arange(N)[::step]
    ext_data = uncut_data[jupiter.lims[0]-20 : jupiter.lims[1]+20, jupiter.lims[2]:].copy()
    for i in xx:
        ydata = data[:,i]
        fit = spc.FuncFit(xdata=xdata,ydata=ydata,xerr=1)
        fit.gaussian_fit([ydata.max(),xdata.mean(),2],mode='curve_fit')
        color1 = (i/N,0,0.5)
        color2 = (i/N,1-i/N,0.5)
        k,mu,sigma = fit.fit_par
        Dk,Dmu,Dsigma = fit.fit_err

        cov = fit.res['cov']
        v  += [mu]
        Dv += [Dmu]
        k10 = k / 10
        pos = np.argmin(abs(ext_data[:,i]-k10))
        r  += [abs(mu-pos)]
        Dr += [np.sqrt(Dmu**2 + 0.5**2)]
        tmp += [sigma]
        if i % 3 == 0:
            fit.data_plot(ax[0],pltarg1={'color':color1},pltarg2={'color':color2})
            fit.residuals_plot(ax[1],color=color1)
            ax[0].axhline(k,0,1)
            ax0.plot(ext_data[:,i],color=color1)
            ax0.axvline(pos,0,1,color=color2)

    plt.figure()
    plt.imshow(data,origin='lower',norm='log',cmap='gray_r')
    plt.errorbar(xx,v,Dv,fmt='.-')
    plt.errorbar(xx,np.array(v)+np.array(r),Dr,fmt='.-',color='orange')
    plt.errorbar(xx,np.array(v)-np.array(r),Dr,fmt='.-',color='orange')
    plt.show()

    cen, Dcen = spc.mean_n_std(v)
    r, Dr = spc.mean_n_std(r) 

    print('Centre',cen,Dcen,Dcen/p1*100)
    print('Radius',r,Dr,Dr/r*100)
    print('Sigma',np.mean(tmp))
     
    lim0 = jupiter.lims[0]
    cen += lim0
    up  = cen + r
    low = cen - r
    jupiter.data = uncut_data.copy()
    jupiter.lims = [np.floor(low).astype(int),np.floor(up).astype(int),42,1391]
    jupiter.cut_image()
    data = jupiter.data.copy()

    plt.figure()
    for i in range(data.shape[1])[::50]:
        plt.plot(data[:,i])
    plt.axvline(cen-jupiter.lims[0],0,1,linestyle='--')
    plt.axvline(up -jupiter.lims[0],0,1,linestyle='--')
    plt.axvline(low-jupiter.lims[0],0,1,linestyle='--')
    plt.show()


    plt.figure()
    plt.imshow(jupiter.data,origin='lower',norm='log',cmap='gray_r')
    plt.figure()
    plt.imshow(uncut_data,origin='lower',norm='log',cmap='gray_r')
    plt.axhline(cen,0,1,color='red')
    plt.axhline(up ,0,1,color='red')
    plt.axhline(low,0,1,color='red')
    plt.show()


    mid = cen - jupiter.lims[0]
    Dmid = np.sqrt((2*Dcen)**2 + Dr**2)
    sel = input('Select the height: ')
    bottom = 24 if sel == '' else int(sel)
    top = np.floor(2*mid - bottom).astype(int)
    h = mid - bottom
    Dh = Dmid

    print(bottom)
    if bottom > len(data): print('Oh no'); exit()

    j1 = data[top]
    j2 = data[bottom]
    j1, Dj1 = spc.mean_n_std(data[top-1:top+2],axis=0)
    j2, Dj2 = spc.mean_n_std(data[bottom-1:bottom+2],axis=0)


    plt.figure()
    plt.imshow(data,cmap='gray_r',norm='log',origin='lower')
    plt.axhline(top,0,1,linestyle='dashed',color='black',alpha=0.7)
    plt.axhline(mid,0,1,linestyle='dashed',color='red',alpha=0.7)
    plt.axhline(bottom,0,1,linestyle='dashed',color='black',alpha=0.7)
    plt.colorbar()
    plt.figure()
    plt.plot(j1,'-',label=f'h = {top}')
    plt.plot(j2,'-',label=f'h = {bottom}')
    plt.legend()
    plt.show()
    # exit()

    sel = input('Select ends separated by comma or type "pass" / press enter:\n> ')
    if sel == 'pass' or sel == '':
        start, stop = 0, None
    else:
        start, stop = np.fromstring(sel,dtype=int,sep=',')

    if stop is None:
        px_val = np.arange(start,data.shape[1])
    elif stop > 0:
        px_val = np.arange(start,stop)
    else:
        px_val = np.arange(start,data.shape[1]+stop)

    j1_p = j1[slice(start,stop)].copy()
    j2_p = j2[slice(start,stop)].copy()
    j1 = j1_p - data.mean(axis=0)[slice(start,stop)]
    j2 = j2_p - data.mean(axis=0)[slice(start,stop)]
    plt.figure()
    plt.imshow(data[:,slice(start,stop)],cmap='gray_r',norm='log',origin='lower')
    plt.axhline(top,0,1,linestyle='dashed',color='black',alpha=0.7)
    plt.axhline(mid,0,1,linestyle='dashed',color='red',alpha=0.7)
    plt.axhline(bottom,0,1,linestyle='dashed',color='black',alpha=0.7)
    plt.figure()
    plt.plot(px_val,j1,'-',label=f'h = {top}')
    plt.plot(px_val,j2,'-',label=f'h = {bottom}')
    plt.grid()
    plt.legend()
    plt.show()
    corr = np.correlate(j1,j2,mode='full')
    from scipy.signal import find_peaks
    pks, _ = find_peaks(corr,threshold=0.0001e8)
    plt.figure()
    plt.plot(corr)
    plt.plot(pks,corr[pks],'.',color='red')
    lags = np.arange(len(corr)) - (len(j1)-1)
    p_corr = np.where(lags >= 0, corr, 0)
    n_corr = np.where(lags <= 0, corr, 0)
    shift1 = lags[n_corr.argmax()]
    shift2 = lags[p_corr.argmax()]
    # shift = lags[corr.argmax()]
    print('CORR',shift1)
    print('CORR',shift2)
    print(lags[[0,-1]])
    plt.figure()
    plt.plot(lags,corr)
    plt.axvline(shift1,0,1,color='orange',linestyle='--')
    plt.axvline(shift2,0,1,color='orange',linestyle='--')
    plt.grid()
    plt.show()

    j1 = j1_p.copy()
    j2 = j2_p.copy()

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(px_val,j1)
    plt.plot(px_val,j2)
    plt.grid()
    plt.subplot(3,1,2)
    plt.title(f'shift: {shift1}')
    plt.plot(px_val,j1,label=f'h = {top}')
    plt.plot(px_val[:shift1],j2[-shift1:],label=f'h = {bottom}')
    plt.grid()
    plt.legend()
    plt.subplot(3,1,3)
    plt.title(f'shift: {shift2}')
    plt.plot(px_val,j1,label=f'h = {top}')
    plt.plot(px_val[shift2:],j2[:-shift2],label=f'h = {bottom}')
    plt.grid()
    plt.legend()
    plt.show()

    ## WAVELENGTH CALIBRATION
    heights = np.array([710+i*5 for i in range(4)])
    lamp.spec, lamp.std = spc.mean_n_std(lamp.data[heights],axis=0)
    plt.figure()
    plt.errorbar(np.arange(*lamp.spec.shape),lamp.spec,lamp.std,fmt='.-')
    plt.show()

    lines, px, Dpx = spc.get_cal_lines(obs_night,target_name)
    Dlines = lines/20000 / 2

    m0 = np.mean(np.diff(lines)/np.diff(px))
    fit = spc.FuncFit(xdata=px,xerr=Dpx,ydata=lines,yerr=Dlines)
    fit.linear_fit([m0,0])
    fit.plot(mode='subplots',points_num=3)
    plt.show()

    ang_diam = 44.29871
    ang_rad  = ang_diam / 2

    print('Ang_pix', ang_rad/r)

    ## PERIOD ESTIMATION
    # T = 8pi Rg/c l/Dl x/R
    from spectralpy.stuff import unc_format
    from astropy.constants import c
    m  = fit.fit_par[0]
    Dm = fit.fit_err[0]
    R = 71492 * u.km
    DR = 4 * u.km 
    c = c.to(u.m/u.s)

    P = []
    DP = []

    print('shift1',shift1)
    delta_l1 = m*abs(shift1)
    Dl1 = Dm*abs(shift1)
    fmt = unc_format(delta_l1,Dl1)
    res_str = 'Delta L1 = {delta_l:' + fmt[0][1:] + '} +/- {Dl:' + fmt[1][1:] + '} AA'
    print(res_str.format(delta_l=delta_l1,Dl=Dl1))
    v  = c.value*delta_l1/balmer[0] / 4
    Dv = v * np.sqrt((Dl1/delta_l1)**2 + (bal_err[0]/balmer[0])**2)
    fmt1 = unc_format(v,Dv)
    res_str = 'v1 = {v:' + fmt1[0][1:] + '} +/- {Dv:' + fmt1[1][1:] + '} m/s'
    print(res_str.format(v=v,Dv=Dv))
    T = R/(v*u.m/u.s)*2*np.pi
    DT = (Dv/v + DR/R) * T
    print('T1',T.to(u.h), DT.to(u.h))
    T = 8*np.pi * (R/c) * (balmer[0]/delta_l1) * (h/r)
    DT = T * np.sqrt( (DR/R)**2 + (bal_err[0]/balmer[0])**2 + (Dl1/delta_l1)**2 + (Dr/r)**2 + (Dh/h)**2)
    print('T1',T.to(u.h),DT.to(u.h))

    P  += [T.to(u.h).value]
    DP += [DT.to(u.h).value]

    print('shift2',shift2)
    delta_l2 = m*abs(shift2)
    Dl2 = Dm*abs(shift2)
    fmt = unc_format(delta_l2,Dl2)
    res_str = 'Delta L2 = {delta_l:' + fmt[0][1:] + '} +/- {Dl:' + fmt[1][1:] + '} AA'
    print(res_str.format(delta_l=delta_l2,Dl=Dl2))
    v  = c.value*delta_l2/balmer[0] / 4
    Dv = v * np.sqrt((Dl2/delta_l2)**2 + (bal_err[0]/balmer[0])**2)
    fmt1 = unc_format(v,Dv)
    res_str = 'v2 = {v:' + fmt1[0][1:] + '} +/- {Dv:' + fmt1[1][1:] + '} m/s'
    print(res_str.format(v=v,Dv=Dv))
    T = R/(v*u.m/u.s)*2*np.pi
    DT = (Dv/v + DR/R) * T
    print('T2',T.to(u.h), DT.to(u.h))
    T = 8*np.pi * (R/c) * (balmer[0]/delta_l2) * (h/r)
    DT = T * np.sqrt( (DR/R)**2 + (bal_err[0]/balmer[0])**2 + (Dl2/delta_l2)**2 + (Dr/r)**2 + (Dh/h)**2)
    print('T2',T.to(u.h),DT.to(u.h))

    P  += [T.to(u.h).value]
    DP += [DT.to(u.h).value]

    P = np.mean(P)
    DP = np.sqrt(DP[0]**2 + DP[1]**2)
    print(f'period = {P:.2f} +/- {DP:.2f} h --> {DP/P:.2%}')
    period = 9 + 55/60 + 29.711/3600
    print(f'period = {period} h -->', 'OK' if P-DP <= period <= P+DP else 'NO') 