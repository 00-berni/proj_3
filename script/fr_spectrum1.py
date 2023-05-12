from statistics import mean
from struct import unpack
import numpy as np
import astropy as astr
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from scipy.optimize import curve_fit
import scipy.ndimage as ndimage
import functions as my

folder_name = './data_files/26072022'

'''
In the file targets.txt the names of the fits data for the various targets. We took for each observation a lamp calibration.
	name	matrix with 3 columns: first one name of fits file target, second one calibration lamp and third (only for Jupiter) flat 
This is the index of rows:
	 [0]	Beta Lyr
	 [1]	Gamma Cyg
	 [2]	Jupiter		with flat
	 [3]	Moon White face	no calibration lamp
	 [4]	Moon Black face
	 [5]	Moon Black face only IMAGE
	 [6]	M57
	 [7]	Mars
	 [8]	Pi Cyg
	 [9]	Saturn
	[10]	Saturn only IMAGE
	[11]	Vega
	[12]	WR137
'''
names = pd.read_csv('./'+ folder_name +'/targets.csv',sep='\t')
names = pd.DataFrame(names).to_numpy() 
folders = ['betaLyr','gammaCygni','giove','luna','luna','luna','m57','marte','pyCygni','saturno','saturno','vega','wr137']
fold0 = 'C:/Users/berna/Desktop/FISICA/ASTROFISICA/#OHP_France/26072022/'	


	
obj_ind = 0

target_file = names[obj_ind][0]
calibration_file = names[obj_ind][1]
fold = fold0 + folders[obj_ind]+'/'

path = lambda name : fold + name + '.fit'


# Open target image
hdul = fits.open(path(target_file))

hdul.info()
print(hdul[0].header)

ly,ry = [650,1400]
lx,rx = [760,-1]

hdul, data = my.targetDatafit(path(target_file))

# data[Y,X]
data = hdul[0].data
data = data[ly:ry,lx:rx]
nx = len(data[0,:]); ny = len(data[:,0])

# hot px
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
kernel = Gaussian2DKernel(x_stddev=1)
data = interpolate_replace_nans(data, kernel)


# Spectrum image
plt.figure()
plt.title(folders[obj_ind])
plt.imshow(data, cmap='gray')#, norm=LogNorm())#,vmin=2E3,vmax=3E3)
plt.colorbar()



# rotation correction
mean_val = np.array([])
for i in range(0,nx):
	tmp  = data[:,i]
	tmp2 = np.where(tmp == max(tmp))
	tmp2 = tmp2[0]
	if(len(tmp2)>1):
		tmp2 = tmp2.sum()/len(tmp2)
	mean_val = np.append(mean_val,tmp2)


angle, data_rot = my.rotation(nx,mean_val,data)

plt.figure()
plt.title(folders[obj_ind]+' - rotated')
plt.imshow(data_rot, cmap='gray')
plt.colorbar()


# spectrum graph
data = data_rot
cut = 563
values = data[cut,:]
nx = len(data[0,:]); ny = len(data[:,0])


# remove noise
# values -= data[350,:]

plt.figure()
plt.title(f'Object spectrum at {cut}')
plt.plot(np.linspace(0,nx,nx),values)


values = np.array([data[555:574,j].sum() for j in range(nx) ])

# remove noise
# values -= data[350,:]

plt.figure()
plt.title('Object spectrum')
plt.plot(np.linspace(0,nx,nx),values)

plt.show()



###############################################

# Open calibraion file
hdul = fits.open(path(calibration_file))

hdul.info()

# data[Y,X]
cdata = hdul[0].data
cdata = cdata[ly:ry,lx:rx]
cnx = len(cdata[0,:]); cny = len(cdata[:,0])


cdata_rot = ndimage.rotate(cdata, angle, reshape=False)

# Spectrum image
plt.figure()
plt.title('Calibration Lamp')
plt.imshow(cdata, cmap='gray')
plt.colorbar()

plt.figure()
plt.title('Calibration Lamp rotated')
plt.imshow(cdata_rot, cmap='gray')
plt.colorbar()


# spectrum graph
cdata = cdata_rot
cut = 350
cvalues = cdata[cut,:]
cnx = len(cdata[0,:]); cny = len(cdata[:,0])


plt.figure()
plt.title(f'Lamp spectrum at {cut}')
plt.plot(np.linspace(0,cnx,cnx),cvalues)



# fit
lA, lpx = np.loadtxt(fold+'calibration_lines.txt',unpack=True)

def FitFun(x,p0,p1,p2,p3):
	return p0 + p1*x + p2*x**2 + p3*x**3

initial_values = np.array([1e2,1.,0.,0.])
pop, pcov = curve_fit(FitFun,lpx,lA,initial_values)
p1,p2,p3,p4 = pop
Dp1,Dp2,Dp3,Dp4 = np.sqrt(pcov.diagonal())

for i in range(2):
	initial_values = pop
	pop, pcov = curve_fit(FitFun,lpx,lA,initial_values)
	p1,p2,p3,p4 = pop
	Dp1,Dp2,Dp3,Dp4 = np.sqrt(pcov.diagonal())


plt.figure()
xx = np.linspace(0,max(lpx)+100,1000)
plt.plot(lpx,lA,'.')
plt.plot(xx,FitFun(xx,p1,p2,p3,p4))

plt.figure()
plt.plot(lpx,abs(1-FitFun(lpx,p1,p2,p3,p4)/lA),'.')
plt.plot([min(lpx),max(lpx)],[0,0],'--')




##! True target spectrum
px = np.linspace(0,nx,nx)
l = FitFun(px,p1,p2,p3,p4)

plt.figure()
plt.title('Calibrated Spectrum of ' + folders[obj_ind])
plt.xlabel('$\lambda$ [A]'); plt.ylabel('[a.u.]')
plt.plot(l,values)


plt.savefig(fold+folders[obj_ind]+'-calibrated_spectrum.png')



hdul.close()


hdulFlat, dataFlat = my.targetDatafit(fold0+'giove/'+names[2][2]+'.fit',lims=[ly,ry,lx,rx])
nxf = len(dataFlat[0,:]); nyf = len(dataFlat[:,0])

valuesFlat = np.array([dataFlat[:,j].sum() for j in range(nxf)])

flat = dataFlat.sum()

print(flat)

flat = valuesFlat/flat

plt.figure()
plt.title('Flat')
plt.plot(np.linspace(0,nxf,nxf),flat)

plt.figure()
plt.title('Spectrum with flat')
plt.plot(l,values/flat)

plt.show()

################################################

from scipy.signal import correlate

obj_ind = 11

target_file = names[obj_ind][0]
calibration_file = names[obj_ind][1]
fold = fold0 + folders[obj_ind]+'/'

# function to open fits
path = lambda name : fold + name + '.fit'

hdul1,data1 = my.targetDatafit(path(target_file),lims=[ly,ry,lx,rx])
hdul2,data2 = my.targetDatafit(path(calibration_file),lims=[ly,ry,lx,rx])

angle, data1 = my.RotCor(data1)
data2 =  ndimage.rotate(data2, angle, reshape=False)

my.showfits(data1,title='Rotated')
my.showfits(data2,title='Rotated')


## Correlation
# set the fit lamp values as a standard
stdLamp = cvalues
stdnx   = len(stdLamp)

values2 = data2[cut,:]
nx2 = len(values2)

print(nx2,stdnx)

my.fastplot(range(nx2),values2)
plt.plot(range(stdnx),stdLamp)
my.fastplot(range(nx2),values2-stdLamp)


corr = correlate(stdLamp,values2)/nx2


my.fastplot(range(len(corr)),corr)


plt.show()