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
FONTSIZE = 18



OBS_NIGHT = '18-04-22'
TARGET = 'giove'    
SELECTION = 0
TARGET_NAME = 'Jupiter'
fit_args = { 'mode': 'odr' }                #: parameters for the fit
lim_width = [[0,1391],[[0,112],[108,221]]]  #: coordinates of the ends of the spectrum
lag = 15                                    #: step length to collect xdata
# open and extract the fit file 
jupiter, lamp = spc.get_target_data(OBS_NIGHT,TARGET,SELECTION,obj_name=TARGET_NAME,angle=None,lim_width=lim_width,lag=lag,gauss_corr=False,lamp_incl=False, fit_args=fit_args, diagn_plots=False,norm='log',aspect='equal')


plt.figure(figsize=(10,7))
plt.imshow(jupiter.data[:,::-1],cmap='gray',norm='log')
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
    # fit a gaussian
    fit = spc.FuncFit(xdata=xdata,ydata=ydata)
    hm = ydata.max()/2
    hm_pos = np.argmin(abs(hm-ydata))
    hwhm = abs(ydata.argmax()-hm_pos)
    initial_values = [ydata.max(),ydata.argmax(),hwhm]
    print('initial_values',initial_values)
    fit.gaussian_fit()
    v  += [fit.fit_par[1]]
    Dv += [fit.fit_err[1]]
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
fit.linear_fit(mode='curve_fit')
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

fig,ax = plt.subplots(1,1)
spc.fits_image(fig,ax,jupiter)
plt.show()


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
    fit = spc.FuncFit(xdata=xdata,ydata=ydata)
    fit.gaussian_fit(mode='curve_fit')
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
plt.title('Estimated center positions',fontsize=FONTSIZE+2)
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
print('PX_WIDTH',PX_WIDTH)
print('FOCAL',FOCAL)
print('ANG_DIAM',ANG_DIAM)
print('RAD',rad)

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

fig,ax = plt.subplots(1,1)
fig.suptitle('Light Frame',fontsize=FONTSIZE)
spc.fits_image(fig,ax,cp_jup,fontsize=FONTSIZE,aspect='equal',origin='lower')
plt.show()
plt.figure()
plt.imshow(cp_jup.data,origin='lower',norm='log',cmap='gray_r')
plt.figure()
plt.imshow(uncut_data,origin='lower',norm='log',cmap='gray_r')
plt.axhline(cen,0,1,color='red')
plt.axhline(top,0,1,color='red')
plt.axhline(low,0,1,color='red')
plt.show()

lamp = lamp.rotate_target(angle=angle)

### Wavelength Calibration
# average over 4 rows
heights = np.array([750+i*1 for i in range(10)])
plt.figure()
plt.imshow(lamp.data,aspect='auto',origin='lower')
plt.axhline(700,0,1)
plt.axhline(800,0,1)

incl = 10
incl_ang = np.arctan(incl)*180/np.pi -90
print(incl_ang)
lamp = lamp.rotate_target(angle=incl_ang)
plt.figure()
plt.imshow(lamp.data,aspect='auto',origin='lower')
plt.show()

lamp.spec, lamp.std = spc.mean_n_std(lamp.data[heights],axis=0)
# get data
lines, px, Dpx = spc.get_cal_lines(OBS_NIGHT,TARGET)
Dlines = lines/20000 / 2
plt.figure()
plt.title('Lamp spectrum',fontsize=FONTSIZE)
plt.errorbar(np.arange(*lamp.spec.shape),lamp.spec,lamp.std,fmt='.-')
for pxi,Dpxi in zip(px,Dpx):
    plt.axvline(pxi,0,1,color='red',linestyle='dashdot')
    plt.axvspan(pxi-Dpxi,pxi+Dpxi,facecolor='orange',alpha=0.4)
plt.ylabel('I [a.u.]',fontsize=FONTSIZE)
plt.xlabel('x [px]',fontsize=FONTSIZE)
plt.show()
# fit a line
# m0 = np.mean(np.diff(lines)/np.diff(px))
fit = spc.FuncFit(xdata=px,xerr=Dpx,ydata=lines,yerr=Dlines)
fit.linear_fit(mode='curve_fit')
fit.plot(mode='subplots',points_num=3,plotargs={'title':'Wavelength calibration','ylabel':'$\\lambda$ [$\\AA$]','fontsize': FONTSIZE},xlabel='x [px]',fontsize=FONTSIZE)
plt.show()

plt.figure()
plt.imshow(jupiter.data)
plt.xticks(np.arange(jupiter.data.shape[1])[::30],fit.method(np.arange(jupiter.data.shape[1])[::30]+jupiter.lims[2]))

plt.figure()
plt.plot(fit.method(np.arange(jupiter.data.shape[1])+jupiter.lims[2]),jupiter.data.sum(axis=0))
plt.show()

### Period Estimation

## Heights
# compute the position of the centre
mid = cen - cp_jup.lims[0]
px_u = np.round(mid+rad).astype(int)
px_d = np.round(mid-rad).astype(int)
# def find_minimum(values, avg: float, dir: str):
#     pos = values.argmax()
#     step = -1 if dir == 'left' else +1
#     while values[pos] > avg:
#         pos += step
#         if pos == 0 or pos == len(values):
#             break
#     return pos

## Lines Selection
cut_list = [(232,257),(428,450),(570,594),(598,620),(676,705),(886,910),(1184,1205)]
# sel_cut = (885,910)
plt.figure(figsize=(15,10))
plt.title('Selected Lines',fontsize=FONTSIZE+2)
plt.imshow(data,origin='lower',cmap='gray')
for sel_cut in cut_list: 
    # plt.axvspan(*sel_cut,color='yellow',alpha=0.2)
    plt.axvline(sel_cut[0],0,1,color='w',linestyle='dashdot')
    plt.axvline(sel_cut[1],0,1,color='w',linestyle='dashdot')
plt.show()
(mf, qf), (Dmf, Dqf) = fit.results()
pcov = fit.res['cov'].copy()

sel_cut = (676,705)
p1 = data[px_u,slice(*sel_cut)]
p2 = data[px_d,slice(*sel_cut)]
P1 = p1.copy()
P2 = p2.copy()
av1 = p1.max() - np.mean(data[px_u])
av2 = p2.max() - np.mean(data[px_d])
p1 = p1.max()-p1    
p2 = p2.max()-p2    
x1 = np.arange(len(p1))    
x2 = np.arange(len(p2))    

plt.figure(figsize=(15,10))
plt.title(f'Selected tilted line',fontsize=FONTSIZE+2)
plt.imshow(data[:,slice(*sel_cut)],cmap='gray_r')
plt.axhline(px_u,0,1)
plt.axhline(px_d,0,1)

from scipy.signal import correlate, correlation_lags

new_corr = correlate(p1-av1,p2-av2,mode='full')
new_corr /= new_corr.max()
new_lag = correlation_lags(len(p1),len(p2))
corr_shift = abs(new_lag[new_corr.argmax()])
Dcorr_shift = 1
print('CORR',corr_shift,Dcorr_shift/corr_shift*100)
plt.figure()
plt.plot(p1)
plt.plot(p2)
plt.figure()
plt.plot(p1-av1)
plt.plot(p2-av2)
plt.show()
plt.figure()
plt.plot(new_lag,new_corr)
plt.errorbar(corr_shift,new_corr.max(),xerr=Dcorr_shift,color='red',capsize=3)

plt.figure()
plt.subplot(211)
plt.title('Cross-correlation of the two spectra', fontsize=FONTSIZE+2)
plt.plot(P1,label='top')
plt.plot(P2,label='bottom')
plt.legend(fontsize=FONTSIZE)
plt.grid(which='both',linestyle='--',alpha=0.2,color='grey')
plt.xlabel('x [px]',fontsize=FONTSIZE)
plt.ylabel('counts',fontsize=FONTSIZE)
plt.subplot(212)
plt.plot(new_lag,new_corr,label='cross-correlation')
plt.errorbar(corr_shift,new_corr.max(),xerr=Dcorr_shift,fmt='.',color='red',capsize=3,label='maximum')
plt.legend(fontsize=FONTSIZE)
plt.grid(which='both',linestyle='--',alpha=0.2,color='grey')
plt.xlabel('lag [px]',fontsize=FONTSIZE)
plt.ylabel('Norm. values',fontsize=FONTSIZE)



int_spec = np.sum(data[:,slice(*sel_cut)],axis=0)
minpos = np.argmin(int_spec)
minpx  = minpos + sel_cut[0] + cp_jup.lims[2]
Dminpos = 1
minval = fit.method(minpx)
print('MINVAL',minpx,minval)
plt.figure()
plt.subplot(121)
plt.title(f'Selected tilted line',fontsize=FONTSIZE+2)
plt.imshow(data[:,slice(*sel_cut)],cmap='gray_r')
plt.colorbar()
plt.subplot(122)
plt.title('Cumulative spectrum',fontsize=FONTSIZE+2)
plt.plot(int_spec,'.--')
plt.errorbar(minpos,int_spec[minpos],xerr=Dminpos,fmt='.',capsize=3,color='red',label='$p_0$')
plt.xlabel('x [px]',fontsize=FONTSIZE)
plt.ylabel('counts',fontsize=FONTSIZE)
plt.grid(which='both',linestyle='--',alpha=0.2,color='grey')
plt.show()
print('ERR',np.sqrt((Dqf/mf)**2 + (qf*Dmf/mf**2)**2 + 2*fit.res['cov'][0,1]/mf/qf))
# 8pi R/C (min+q/m)/sh
# 8pi R/C [ (Dq/m)² + (q*Dm/m²)² + 2 q/m³ Cov] / sh
# T [ (Dq/q) + (Dm/m) + 2Cov/qm ]
T3 = (8*np.pi*R/C * minval/(corr_shift*mf)).to(u.h)
DT3 = (8*np.pi*R/C).to(u.h) * ((Dminpos + np.sqrt((Dqf/mf)**2 + (qf*Dmf/mf**2)**2 + 2*fit.res['cov'][0,1]*qf/mf**3))/corr_shift + (minpx + qf/mf)*Dcorr_shift/corr_shift**2) 
# DT3 = (8*np.pi*R/C/corr_shift).to(u.h) * (Dminpos + np.sqrt((Dqf/mf)**2 + (qf*Dmf/mf**2)**2 + 2*fit.res['cov'][0,1]*qf/mf**3)) 
# DT3 = (T3-(8*np.pi*R/C*minpos/shift3).to(u.h)) * np.sqrt((Dmf/mf)**2+(Dqf/qf)**2+2*fit.res['cov'][0,1]/mf/qf)
# DT3 = (T3-8*np.pi*R/C*minpos/shift3).to(u.h) * np.sqrt((Dmf/mf)**2+(Dqf/qf)**2)
print('CONST',(8*np.pi*R/C).to(u.h))
print((C*(corr_shift*mf)/minval/4).to(u.km/u.s))
print(T3,DT3,DT3/T3*100,'%')
print(PERIOD)
print((T3-PERIOD))
print((T3-PERIOD)/DT3)
if (T3-DT3).value <= PERIOD.value <= (T3+DT3).value: 
    print('OK')
else:
    print(((T3-PERIOD)/DT3).value)  


int_data = data.sum(axis=0)
ha_px = int_data.argmin() + cp_jup.lims[2]
Dha_px = 1
est_ha = fit.method(ha_px)
discr = BALMER - est_ha
Ddiscr = mf*Dha_px + np.sqrt((Dmf*ha_px)**2 + Dqf**2 + 2*fit.res['cov'][0,1]*ha_px)
print('Halpha',ha_px,est_ha)
print('DISCR',discr,Ddiscr, abs(discr/Ddiscr))
# A * ((m p + q) + B - m ha - q) / (m Ds)
# A * (m(p-ha) + B) / (m Ds)
# A * (p-ha + B/m) / Ds
# A/Ds * (Dp + Dha + Dm*B/m**2 + (p-ha + B/m)*DDs/Ds) 
T3 = (8*np.pi*R/C * (minpx - ha_px + BALMER/mf)/(corr_shift)).to(u.h)
DT3 = (8*np.pi*R/C/corr_shift).to(u.h) * (Dminpos + Dha_px + BALMER*Dmf/mf**2 + (minpx+ha_px + BALMER/mf)*Dcorr_shift/corr_shift)
print((8*np.pi*R/C/corr_shift).to(u.h) *( Dha_px + BALMER*Dmf/mf**2))
print((8*np.pi*R/C/corr_shift).to(u.h) *(  (minpx+ha_px + BALMER/mf)*Dcorr_shift/corr_shift))
print('New T3',T3,DT3)
print('\n\n========\n')
plt.figure()
plt.imshow(jupiter.rotate_target(angle=incl_ang).data,vmin=data.min(),vmax=data.max(),origin='lower',aspect='auto')
for i in [271,361,608,717,960]:
    plt.axvline(i,0,1,linestyle='dashed')
# plt.plot([px_ul[0],px_ur[0]],[px_ul[1],px_ur[1]])
# plt.plot([px_dl[0],px_dr[0]],[px_dl[1],px_dr[1]])
corr_jup = jupiter.rotate_target(angle=incl_ang)
corr_jup.lims = np.array([561,798,0,1363])
corr_jup.cut_image()
data = corr_jup.data.copy()
plt.figure()
plt.imshow(data,origin='lower',aspect='auto',cmap='gray')
plt.figure()
plt.plot(fit.method(np.arange(data.shape[1])+corr_jup.lims[2]),data.sum(axis=0))
pos_halpha = data.sum(axis=0).argmin()+corr_jup.lims[2]
new_halpha = fit.method(pos_halpha)
Dnew_halpha = abs(mf+np.sqrt((Dmf*pos_halpha)**2+Dqf**2+2*pcov[0,1]*pos_halpha))
v_bulk = C.to(u.km/u.s)/2*(new_halpha-BALMER)/BALMER
Dv_bulk = abs(C.to(u.km/u.s)/2*Dnew_halpha/BALMER)
print('New Halpha',new_halpha,'+-',Dnew_halpha,Dnew_halpha/new_halpha*100,BALMER,new_halpha-BALMER)
print('V bulk',v_bulk,'+-',Dv_bulk)

def new_computation(sel_line: tuple[int,int], sel_range: tuple[tuple[int,int],tuple[int,int]]):
    htp, hbt = sel_line      #: heights
    rtp, rbt = sel_range     #: widths
    jtp = data[htp,slice(*rtp)]
    jbt = data[hbt,slice(*rbt)]
    jsum = data[slice(hbt,htp),slice(rtp[0],rbt[-1])].sum(axis=0)
    px_diff = abs((jtp.argmin()+rtp[0])-(jbt.argmin()+rbt[0]))
    diff = px_diff*mf
    
    midpos  = jsum.argmin() if jsum.argmin() != 1223 else ((jtp.argmin()+rtp[0])+(jbt.argmin()+rbt[0]))/2 - rtp[0]
    # T = CONST * (x + q/m)/(pxdiff)
    # DT = CONST * (1/pxdiff + midline/diff * 1/pxdiff + sqrt((q*Dm/m**2)**2 + (Dq/m)**2 + 2pcov[0,1]*q/m**3)/pxdiff)
    midline = fit.method(midpos+rtp[0]+corr_jup.lims[2])
    period = (np.pi*8*R/C).to(u.h)*midline/diff 
    error  = (np.pi*8*R/C).to(u.h) * (1/px_diff + midline/diff * 1/px_diff + np.sqrt((qf*Dmf/mf**2)**2 + (Dqf/mf)**2 + 2*pcov[0,1]*qf/mf**3)/px_diff)
    print('\nResults:',sel_line,midpos+rtp[0]+corr_jup.lims[2])
    print((jtp.argmin()+rtp[0]),(jbt.argmin()+rbt[0]))
    print('PERIOD',period,'+-',error,'diff',period-PERIOD,(period-PERIOD)/error)
    # T = CONST * (m*(x-xalpha) + Halpha)/(m*pxdiff)
    # DT = CONST * ((Dx+Dxalpha)/pxdiff + (m*(x-xalpha) + Halpha)/(m*pxdiff)*Dpxdiff/pxdiff)
    corr_p = (np.pi*8*R/C).to(u.h)*(midpos+rtp[0]+corr_jup.lims[2]-pos_halpha+BALMER/mf)/px_diff
    corr_e = (np.pi*8*R/C).to(u.h)/px_diff*(2+(midpos+rtp[0]+corr_jup.lims[2]-pos_halpha+BALMER/mf)/px_diff+BALMER*Dmf/mf**2)
    print('CORR PERIOD',corr_p,'+-',corr_e,'diff',corr_p-PERIOD,(corr_p-PERIOD)/corr_e)
    plt.figure()
    plt.imshow(data[slice(hbt-4,htp+4),slice(rtp[0],rbt[-1])],cmap='gray',origin='lower')
    plt.axhline(4,0,1,linestyle='dashed',color='blue',label='bottom')
    plt.axhline((htp-hbt)+4,0,1,linestyle='dashed',color='orange',label='top')
    plt.legend()
    plt.figure()
    plt.plot(jtp,color='orange',label='top')
    plt.plot(jbt,color='blue',label='bottom')
    plt.legend()
    plt.grid()
    plt.figure()
    plt.plot(jsum)
    plt.errorbar(midpos,jsum.min(),xerr=1,color='red',capsize=3)
    plt.grid()
    plt.show()
    return period.to(u.h).value

selected_lines = [
                    [(154,64),((622,634),(635,647))      ],
                    [(186,96),((908,920),(921,935))      ],
                    [(218,123),((1205,1219),(1219,1233)) ],
                    [(218,123),((1219,1233),(1233,1250)) ],
                    [(220,131),((1264,1280),(1278,1292)) ]
                ]

fig,ax = plt.subplots(1,1)
# plt.imshow(data,cmap='gray',origin='lower')
spc.fits_image(fig,ax,corr_jup,v=0,subtitle=None,vmax=7000,origin='lower',aspect='equal')
ax.set_title('Selected Lines',fontsize=FONTSIZE+5)
for sel in selected_lines:
    top, bottom = sel[0]
    lends, rends = sel[1]
    if lends[0] != 1219:
        ax.plot([lends[0],lends[0]],[bottom-4,top+4],linestyle='dashed',color='white')
    if rends[0] != 1219:
        ax.plot([rends[-1],rends[-1]],[bottom-4,top+4],linestyle='dashed',color='white')
plt.show()
periods = [ new_computation(*sel) for sel in selected_lines ]

mean_period, std_period = spc.mean_n_std(periods)
print('\nResult')
spc.print_measure(mean_period,std_period,'P','h')
print(mean_period-PERIOD.value,(mean_period-PERIOD.value)/std_period)

# [186,908:920]
# [96,921:935]

# [218,1205:1219]
# [123,1219:1233]

# [218,1219:1232]
# [123,1233:1250]

# [220,1264:1280]
# [131,1278:1292]
