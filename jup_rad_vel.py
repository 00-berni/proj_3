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

    # N = 100
    # shift = 4
    # xx = np.arange(N)
    # s1 = np.random.normal(0,0.2,N)
    # s1[5] += -2
    # s1[13] += -1
    # s2 = np.random.normal(0,0.2,N)
    # s2[5+shift] += -2
    # s2[13+shift] += -1

    # plt.figure()
    # plt.plot(xx,s1,label='s1')
    # plt.plot(xx,s2,label='s2')
    # plt.legend()

    # corr = correlate(s1,s2,mode='full')
    # print(len(corr))
    # lags = np.arange(len(corr)) - (N - 1)
    # plt.figure()
    # plt.plot(lags,corr)
    # plt.axvline(-shift,0,1,color='k')

    # sh_pos = corr.argmax()
    # shift1 = lags[sh_pos]
    # print(shift1)
    # if shift1 < 0:
    #     plt.figure()
    #     plt.plot(xx,s1,label='s1')
    #     plt.plot(xx[:shift1],s2[-shift1:],label='s2')
    #     plt.legend()
    # elif shift1 >0:
    #     plt.figure()
    #     plt.plot(xx[:-shift1],s1[shift1:],label='s1')
    #     plt.plot(xx,s2,label='s2')
    #     plt.legend()
    # plt.show()


    # exit()
    ## DATA
    obs_night = '18-04-22'
    target_name = 'giove'    
    selection = 0
    fit_args = {    'mode': 'curve_fit',
                    'absolute_sigma': True }
    fit_args = {    'mode': 'odr' }
    lim_width = [[0,1391],[[0,112],[108,221]]]
    lag = 15
    # jupiter, lamp = spc.get_target_data(obs_night,target_name,selection,angle=None,lim_width=lim_width,lag=lag,gauss_corr=True,lamp_incl=False, fit_args=fit_args, diagn_plots=True,norm='log')
    jupiter, lamp = spc.get_target_data(obs_night,target_name,selection,angle=None,lim_width=lim_width,lag=lag,gauss_corr=False,lamp_incl=False, fit_args=fit_args, diagn_plots=False,norm='log')
    

    # plt.figure()
    # plt.imshow(jupiter.data,origin='lower',norm='log',cmap='gray_r')
    # plt.show()


    ## VELOCITY ESTIMATION
    data = jupiter.data.copy()[5:] 

    plt.figure()
    for i in range(data.shape[1])[::5]:
        plt.plot(data[:,i])
    plt.show()


    v  = np.array([])
    Dv = np.array([])
    t  = np.array([])
    Dt = np.array([])
    s  = np.array([])
    Ds = np.array([])
    xxx = np.array([])
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
        if delta >=0:
            t = np.append(t,-b/2/a - np.sqrt(delta/4/a**2))
            xxx = np.append(xxx,i)
    # plt.show()
    plt.figure()
    plt.imshow(data,origin='lower',norm='log',cmap='gray_r')
    plt.errorbar(xx,v,Dv,fmt='.-')
    plt.plot(xxx,t,'.')
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
    # _,p1 = rot_mat.dot(np.array([0,fit.fit_par[1]]))
    _,p1  = np.array([rot_mat.dot([xx[i],v[i]]) for i in range(len(xx))]).mean(axis=0)
    _,t  = np.array([rot_mat.dot([xxx[i],t[i]]) for i in range(len(xxx))]).mean(axis=0)
    print('mid',p1,'low',t)
    print(p1-t)

    uncut_data = jupiter.data.copy()
    jupiter.lims = [626,730,42,1391]
    jupiter.cut_image()
    data = jupiter.data.copy()

    v  = np.array([])
    Dv = np.array([])
    t  = np.array([])
    Dt = np.array([])
    s  = np.array([])
    Ds = np.array([])
    xxx = np.array([])
    fig, ax = plt.subplots(2,1)
    xdata = np.arange(data.shape[0])
    N = data.shape[1]
    step = 10
    xx = np.arange(N)[::step]
    for i in xx:
        ydata = data[:,i]
        fit = spc.FuncFit(xdata=xdata,ydata=ydata,xerr=1)
        fit.gaussian_fit([ydata.max(),xdata.mean(),2],mode='curve_fit')
        color1 = (i/N,0,0.5)
        color2 = (i/N,1-i/N,0.5)
        fit.data_plot(ax[0],pltarg1={'color':color1},pltarg2={'color':color2})
        fit.residuals_plot(ax[1],color=color1)
        k,mu,sigma = fit.fit_par
        Dk,Dmu,Dsigma = fit.fit_err

        cov = fit.res['cov']
        v  = np.append(v,mu)
        Dv = np.append(Dv,Dmu)

        # ydata = data[:,i]
        # fit = spc.FuncFit(xdata=xdata,ydata=ydata,xerr=1)
        # fit.pol_fit(2,[-0.2,1,1],mode='curve_fit')
        # color1 = (i/N,0,0.5)
        # color2 = (i/N,1-i/N,0.5)
        # fit.data_plot(ax[0],pltarg1={'color':color1},pltarg2={'color':color2})
        # fit.residuals_plot(ax[1],color=color1)
        # a,b,c = fit.fit_par
        # cov = fit.res['cov']
        # delta = b**2 - 4*a*c
        # der = [ 4*c/(4*a) + delta/(4*a**2),
        #         -2*b/(4*a),
        #         1 ]
        # err = np.sqrt(np.sum([der[j]*der[k]*cov[j,k] for k in range(len(der)) for j in range(len(der))]))
        # s  = np.append(s,-delta/4/a)
        # Ds = np.append(Ds,err)
        # der = [ b/(2*a**2),
        #         -1/(2*a) ]
        # err = np.sqrt(np.sum([der[j]*der[k]*cov[j,k] for k in range(len(der)) for j in range(len(der))]))
        # v  = np.append(v,-b/2/a)

        # Dv = np.append(Dv,err)
        # if delta >=0:
        #     t = np.append(t,-b/2/a - np.sqrt(delta/4/a**2))
        #     xxx = np.append(xxx,i)
    # plt.show()
    plt.figure()
    plt.imshow(data,origin='lower',norm='log',cmap='gray_r')
    plt.errorbar(xx,v,Dv,fmt='.-')
    plt.plot(xxx,t,'.')
    plt.show()

    plt.figure()
    for i in xx:
        plt.plot(data[:,i])
    plt.axvline(v.mean(),0,1,linestyle='--')
    plt.axvline(t.mean(),0,1,linestyle='--')
    plt.show()

    p1 = v.mean()
    t = -10#t.mean()
    lim0 = jupiter.lims[0]
    jupiter.data = uncut_data.copy()
    jupiter.lims = [np.floor(t+lim0).astype(int),np.floor(2*p1-t+lim0).astype(int),42,1391]
    jupiter.cut_image()
    data = jupiter.data.copy()

    plt.figure()
    for i in range(data.shape[1])[::50]:
        plt.plot(data[:,i])
    plt.axvline(p1+lim0-jupiter.lims[0],0,1,linestyle='--')
    plt.show()


    plt.figure()
    plt.imshow(jupiter.data,origin='lower',norm='log',cmap='gray_r')
    plt.figure()
    plt.imshow(uncut_data,origin='lower',norm='log',cmap='gray_r')
    plt.axhline(p1+lim0,0,1,color='red')
    plt.axhline( t+lim0,0,1,color='red')
    plt.axhline(2*p1-t+lim0+1,0,1,color='red')


    mid = p1+lim0-jupiter.lims[0]
    r = mid
    plt.show()
    sel = input('Select the height: ')
    bottom = 20 if sel == '' else int(sel)
    top = np.floor(2*mid - bottom).astype(int)
    h = mid - bottom

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

    j1 = j1[slice(start,stop)] - data.mean(axis=0)[slice(start,stop)]
    j2 = j2[slice(start,stop)] - data.mean(axis=0)[slice(start,stop)]
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
    lags = np.arange(len(corr)) - (len(j1)-1)
    pos_corr = np.where(lags >= 0, corr, 0)
    shift = lags[pos_corr.argmax()]
    shift = lags[corr.argmax()]
    print('CORR',shift)
    print(lags[[0,-1]])
    plt.figure()
    plt.plot(lags,corr)
    plt.grid()
    plt.show()

    j1 += data.mean(axis=0)[slice(start,stop)]
    j2 += data.mean(axis=0)[slice(start,stop)]

    if shift < 0: 
        plt.figure()
        plt.subplot(2,1,2)
        plt.plot(px_val,j1,label=f'h = {top}')
        plt.plot(px_val[:shift],j2[-shift:],label=f'h = {bottom}')
        plt.grid()
        plt.subplot(2,1,1)
        plt.plot(px_val,j1)
        plt.plot(px_val,j2)
        plt.grid()
        plt.show()
    elif shift > 0: 
        plt.figure()
        plt.subplot(2,1,2)
        plt.plot(px_val,j1,label=f'h = {top}')
        plt.plot(px_val[shift:],j2[:-shift],label=f'h = {bottom}')
        plt.grid()
        plt.subplot(2,1,1)
        plt.plot(px_val,j1)
        plt.plot(px_val,j2)
        plt.grid()
        plt.show()
    # else:
    #     plt.figure()
    #     plt.suptitle('MIO')
    #     plt.subplot(2,1,2)
    #     plt.plot(px_val,j1,label=f'h = {height}')
    #     plt.plot(px_val[3:],j2[:-3],label=f'h = {data.shape[0]-height}')
    #     plt.grid()
    #     plt.subplot(2,1,1)
    #     plt.plot(px_val,j1)
    #     plt.plot(px_val,j2)
    #     plt.grid()
    #     plt.show()

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

    ## PERIOD ESTIMATION
    # T = 8pi Rg/c l/Dl x/R

    m  = fit.fit_par[0]
    Dm = fit.fit_err[0]
    delta_l = m*abs(shift)
    Dl = Dm*abs(shift)
    from spectralpy.stuff import unc_format
    fmt = unc_format(delta_l,Dl)
    res_str = 'Delta L = {delta_l:' + fmt[0][1:] + '} +/- {Dl:' + fmt[1][1:] + '} AA'
    print(res_str.format(delta_l=delta_l,Dl=Dl))
    from astropy.constants import c
    c = c.to(u.m/u.s)
    v  = c.value*delta_l/balmer[0] / 4
    Dv = c.value*Dl/balmer[0] / 4
    fmt1 = unc_format(v,Dv)
    res_str = 'v = {v:' + fmt1[0][1:] + '} +/- {Dv:' + fmt1[1][1:] + '} m/s'
    print(res_str.format(v=v,Dv=Dv))
    R = 69911 * u.km
    DR = 6 * u.km 
    T = R/(v*u.m/u.s)*2*np.pi
    DT = (Dv/v + DR/R) * T
    print(T.to(u.h), DT.to(u.h))
    T = 8*np.pi * (R/c) * (balmer[0]/delta_l) * (h/r)
    print(T.to(u.h))