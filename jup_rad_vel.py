import numpy as np
import matplotlib.pyplot as plt
import spectralpy as spc
from astropy import units as u
import astropy.constants as const

## Constants 
BALMER = 6562.79                        #: H_alpha data from NIST
ERRBAL = 0.03                           #: its uncertainty
R = 71492 * u.km                        #: equat. radius of Jupiter from https://ssd.jpl.nasa.gov/horizons/app.html#/
DR = 4 * u.km                           #: its uncertainty
C = const.c.to(u.m/u.s)                 #: light velocity
PERIOD = (9+55/60+29.711/3600) * u.h    #: Sid. rot. period (III) of Jupiter from https://ssd.jpl.nasa.gov/horizons/app.html#/



OBS_NIGHT = '18-04-22'
TARGET_NAME = 'giove'    
SELECTION = 0
fit_args = { 'mode': 'odr' }                #: parameters for the fit
lim_width = [[0,1391],[[0,112],[108,221]]]  #: coordinates of the ends of the spectrum
lag = 15                                    #: step length to collect xdata
# open and extract the fit file 
jupiter, lamp = spc.get_target_data(OBS_NIGHT,TARGET_NAME,SELECTION,angle=None,lim_width=lim_width,lag=lag,gauss_corr=False,lamp_incl=False, fit_args=fit_args, diagn_plots=False,norm='log',aspect='equal')


plt.figure(figsize=(10,7))
plt.imshow(jupiter.data,cmap='gray_r',norm='log')
# from scipy.signal import medfilt2d
# data = medfilt2d(jupiter.data.copy())
data = jupiter.data.copy()
plt.figure(figsize=(10,7))
plt.imshow(data,cmap='gray_r',norm='log')
plt.show()

plt.figure()
for i in range(data.shape[1])[::5]:
    plt.plot(data[:,i])
plt.show()

### Inclination Correction

## Parabola Fits
step = 10
N = data.shape[1]                   #: image horizontal size
col = np.arange(N)[::step]          #: selected columns
v  = []                             #: vertex coordinates of the parabolas for each column
Dv = []                             #: their uncertainties
xdata = np.arange(data.shape[0])
fig, ax = plt.subplots(2,1)
for i in col:
    ydata = data[:,i]
    # fit a parabola
    fit = spc.FuncFit(xdata=xdata,ydata=ydata,xerr=1)
    fit.pol_fit(2,[-0.2,1,1],mode='curve_fit')
    # extract the parameters and the covariances
    a,b,c = fit.fit_par
    cov = fit.res['cov']
    delta = b**2 - 4*a*c
    # compute the uncertainty related with the estimated vertex value
    der = [ b/(2*a**2),
            -1/(2*a) ]
    err = np.sqrt(np.sum([der[j]*der[k]*cov[j,k] for k in range(len(der)) for j in range(len(der))]))
    # store the results
    v  += [-b/2/a]
    Dv += [err]
    # plot
    color1 = (i/N,0,0.5)
    color2 = (i/N,1-i/N,0.5)
    fit.data_plot(ax[0],pltarg1={'color':color1},pltarg2={'color':color2})
    fit.residuals_plot(ax[1],color=color1)
plt.figure()
plt.imshow(data,origin='lower',norm='log',cmap='gray_r')
plt.errorbar(col,v,Dv,fmt='.-')
plt.show()

## Linear Fit
fit = spc.FuncFit(xdata=col,ydata=v,yerr=Dv)
fit.linear_fit([0,np.mean(v)],mode='curve_fit')
fit.plot(mode='subplots')
plt.show()
m  = fit.fit_par[0]
Dm = fit.fit_err[0]

# compute the inclination angle
angle  = np.arctan(m) * 180/np.pi
Dangle = Dm/(1+m**2)  * 180/np.pi
# print the value with the correct digits
spc.print_measure(angle,Dangle,'Angle2','deg')

# compute the total inclination angle
angle  = jupiter.angle[0] + angle
Dangle = np.sqrt(jupiter.angle[1]**2 + Dangle**2)
spc.print_measure(angle,Dangle,'Angle_tot','deg')
# rotate the image
jupiter.data = jupiter.hdul[0].data.copy()
jupiter = jupiter.rotate_target(angle)



### Looking for the center
# store data before slicing
uncut_data = jupiter.data.copy()
cp_jup = jupiter.copy()
# ends of the slice
cp_jup.lims = [626,730,42,1391]
cp_jup.cut_image()
data = cp_jup.data.copy()

## Gaussian Fit 
step = 10
N = data.shape[1]                   #: image horizontal size
col = np.arange(N)[::step]          #: selected columns
c  = []                             #: coordinates of the centre for each column
Dc = []                             #: their uncertainties
xdata = np.arange(data.shape[0])
# extend the ends of the sliced data 
ext_data = uncut_data[cp_jup.lims[0]-40 : cp_jup.lims[1]+40, cp_jup.lims[2]:].copy()
fig, ax = plt.subplots(2,1)
fig0, ax0 = plt.subplots(1,1)
for i in col:
    ydata = data[:,i]
    # fit a Gaussian
    fit = spc.FuncFit(xdata=xdata,ydata=ydata,xerr=1)
    fit.gaussian_fit([ydata.max(),xdata.mean(),2],mode='curve_fit')
    # extract the estimated paramters
    k,mu = fit.fit_par[:2]
    Dk,Dmu = fit.fit_err[:2]
    # store the center value
    c  += [mu]
    Dc += [Dmu]
    # plot
    if i % 3 == 0:
        color1 = (i/N,0,0.5)
        color2 = (i/N,1-i/N,0.5)
        fit.data_plot(ax[0],pltarg1={'color':color1},pltarg2={'color':color2})
        fit.residuals_plot(ax[1],color=color1)
        ax[0].axhline(k,0,1)
        ax0.plot(ext_data[:,i],color=color1)
plt.figure()
plt.imshow(data,origin='lower',norm='log',cmap='gray_r')
plt.errorbar(col,c,Dc,fmt='.-')
plt.show()

# compute the mean and the STD
cen, Dcen = spc.mean_n_std(c)
spc.print_measure(cen,Dcen,'Centre','pxs')

# compute the radius
PX_WIDTH = jupiter.header['YPIXSZ']*1e-6 
FOCAL = jupiter.header['FOCALLEN']*1e-3 
ANG_DIAM = 44.29871 / 3600 * np.pi /180
rad = np.round(ANG_DIAM*FOCAL / PX_WIDTH /2).astype(int)


## Slicing
# store the value of the 0 point
lim0 = cp_jup.lims[0] 
# compute the values in the coordinates of the full image
cen += lim0
top = cen + rad     #: top end of the planet in px 
low = cen - rad     #: bottom end of the planet in px 

plt.figure()
for i in range(data.shape[1])[::50]:
    plt.plot(data[:,i])
plt.axvline(cen-cp_jup.lims[0],0,1,linestyle='--')
plt.axvline(top-cp_jup.lims[0],0,1,linestyle='--')
plt.axvline(low-cp_jup.lims[0],0,1,linestyle='--')
plt.show()

plt.figure()
plt.imshow(cp_jup.data,origin='lower',norm='log',cmap='gray_r')
plt.figure()
plt.imshow(uncut_data,origin='lower',norm='log',cmap='gray_r')
plt.axhline(cen,0,1,color='red')
plt.axhline(top,0,1,color='red')
plt.axhline(low,0,1,color='red')
plt.show()


### Wavelength Calibration
# average over 4 rows
heights = np.array([710+i*1 for i in range(4)])
lamp.spec, lamp.std = spc.mean_n_std(lamp.data[heights],axis=0)
plt.figure()
plt.errorbar(np.arange(*lamp.spec.shape),lamp.spec,lamp.std,fmt='.-')
plt.show()
# get data
lines, px, Dpx = spc.get_cal_lines(OBS_NIGHT,TARGET_NAME)
Dlines = lines/20000 / 2
# fit a line
m0 = np.mean(np.diff(lines)/np.diff(px))
fit = spc.FuncFit(xdata=px,xerr=Dpx,ydata=lines,yerr=Dlines)
fit.linear_fit([m0,0])
fit.plot(mode='subplots',points_num=3)
plt.show()

### Period Estimation

## Heights
# compute the position of the centre
mid = cen - cp_jup.lims[0]
px_u = np.round(mid+rad).astype(int)
px_d = np.round(mid-rad).astype(int)
def find_minimum(values, avg: float, dir: str):
    pos = values.argmax()
    step = -1 if dir == 'left' else +1
    while values[pos] > avg:
        pos += step
        if pos == 0 or pos == len(values):
            break
    return pos

## Lines Selection
cut_list = [(598,620),(676,705),(886,910),(1184,1205)]
# sel_cut = (885,910)
plt.figure(figsize=(15,10))
plt.imshow(data,origin='lower',cmap='gray')
for sel_cut in cut_list: 
    plt.axvline(sel_cut[0],0,1,color='w',linestyle='dashdot')
    plt.axvline(sel_cut[1],0,1,color='w',linestyle='dashdot')

(mf, qf), (Dmf, Dqf) = fit.results()
est_ps = []
for sel_cut in cut_list: 
    print('\n- - -\nValue for',sel_cut)
    sh = 0
    p1 = data[px_u+sh,slice(*sel_cut)]
    p2 = data[px_d-sh,slice(*sel_cut)]
    p1 = p1.max()-p1    
    p2 = p2.max()-p2    
    x1 = np.arange(len(p1))    
    x2 = np.arange(len(p2))    
    mp1 = p1.mean()
    mp2 = p2.mean()

    plt.figure(figsize=(15,10))
    plt.imshow(data[:,slice(*sel_cut)],origin='lower')
    plt.axhline(px_u+sh,0,1)
    plt.axhline(px_d-sh,0,1)

    plt.figure(figsize=(15,10))
    plt.plot(x1,p1,'.--',color='b')
    plt.plot(x2,p2,'.--',color='orange')
    plt.axhline(mp1,0,1,linestyle='dashed',color='b',alpha=0.6)
    plt.axhline(mp2,0,1,linestyle='dashed',color='orange',alpha=0.6)
    plt.xticks(x1,x1)

    if p1.argmax() > p2.argmax():
        right = find_minimum(p1,mp1,'right')
        left  = find_minimum(p2,mp2,'left')
    elif p1.argmax() < p2.argmax():
        right = find_minimum(p2,mp2,'right')
        left  = find_minimum(p1,mp1,'left')
    else:
        break

    print(left,right)
    print(right-left)
    shift3 = abs(right-left)
    plt.figure(figsize=(15,10))
    plt.plot(x1,p1,'.--',color='b')
    plt.plot(x2,p2,'.--',color='orange')
    plt.axvline(left,0,1)
    plt.axvline(right,0,1)
    plt.axhline(mp1,0,1,linestyle='dashed',color='b',alpha=0.6)
    plt.axhline(mp2,0,1,linestyle='dashed',color='orange',alpha=0.6)
    plt.xticks(x1,x1)
    
    minpos = np.argmin(np.sum(data[:,slice(*sel_cut)],axis=0)) + sel_cut[0] + jupiter.lims[2]
    minval = fit.method(minpos)
    print(minval)
    # 8pi R/C (min+q/m)/sh
    # 8pi R/C [ (Dq/m)² + (q*Dm/m²)² + 2 q/m³ Cov] / sh
    # T [ (Dq/q) + (Dm/m) + 2Cov/qm ]
    T3 = (8*np.pi*R/C * minval/(shift3*mf)).to(u.h)
    DT3 = (T3-(8*np.pi*R/C*minpos/shift3).to(u.h)) * np.sqrt((Dmf/mf)**2+(Dqf/qf)**2+2*fit.res['cov'][0,1]/mf/qf)
    # DT3 = (T3-8*np.pi*R/C*minpos/shift3).to(u.h) * np.sqrt((Dmf/mf)**2+(Dqf/qf)**2)
    print(C*(shift3*mf)/minval/4)
    print(T3,DT3,DT3/T3*100)
    print(PERIOD)
    print((T3-PERIOD))
    print((T3-PERIOD)/DT3)
    est_ps += [T3.value]
est_ps, std_ps = spc.mean_n_std(est_ps)
print('\n\n= = =\n')
print(est_ps,std_ps,std_ps/est_ps*100)
if (est_ps-std_ps) <= PERIOD.value <= (est_ps+std_ps): 
    print('OK')
else:
    print((est_ps-PERIOD.value)/std_ps*100)
plt.show()