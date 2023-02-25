"""
SCRIPT FOR FUNCTIONS IMPLEMENTATION

These are the functions used in the `fr_spectrum1.py` script

The implemented functions are:
 - `fastplot` 	:	Makes a simple plot
 - `hotpxRemove` : 	Removes hotpx [taken by internet]
 - `showfits` 	:	Prints fits image
 - `targetDatafit` :	Extracts data from fits and print a row image
 - `RotCor`	:	Corrects the inclination of the spectrum, making a linear fit
 - `initialization`

"""

import numpy as np
import astropy as astr
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage as ndimage
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy.signal import correlate

##*
def fastplot(x,y,numfig=None,title='',labels=['',''],dim=[10,7]):
	"""`fastplot(x,y,numfig,title,labels,dim)`

	Function to have a simple plot quickly.
	You can choose to make a simple plot or add a title and some labels. 
		
	I wrote it only because of my laziness in writing code.

	## Parameters

	`x` : array_like 
	  data on x axis
	`y` : array_like
	  data on y axis
	`numfig` : int, optional
	  number of the generated figure. Default is `None`.
	`title` : str, optional
	  title of the figure. Default is `''`
	`labels` : list, optional
	  `x` and `y` labels. The format is `[xlabel,ylabel]`.
	  Default is `['','']`
	`dim` : list, optional
	  the dimension of the image to pass to `figsize` parameters of `plt.figure` function.
	  Default is `[10,7]`

	## Return

	`None`

	The plot of data
	"""
	xl,yl = labels
	plt.figure(numfig,figsize=dim)
	plt.title(title)
	plt.plot(x,y)
	plt.xlabel(xl); plt.ylabel(yl)
	return
##*

##* Function to remove hot px from the images.
#! [Take from internet]
#	param	data	image matrix
#
#	return	the image without hotpx
def hotpxRemove(data):
	"""Function to remove hot px from the images"""
	kernel = Gaussian2DKernel(x_stddev=1)
	return interpolate_replace_nans(data, kernel)
##*

##*
def showfits(data,title='',n=None,dim=[10,7]):
	"""`showfits(data,title,n,dim)`

	Function to print the fits image.
	  You can print simply the image or set a figure number and a title.

	## Parameters

	`data` : array_like
	  image matrix of fits file
	`title` : str, optional
	  title of the image. Default is `''`
	`n` : int, optional
	  figure number. Default is `None`
	`dim` : list, optional
	  the dimension of the image to pass to `figsize` parameters of `plt.figure` function.
	  Default is `[10,7]`
	
	## Return

	`None`

	The image in grayscale
	"""
	plt.figure(n,figsize=dim)
	plt.title(title)
	plt.imshow(data, cmap='gray')#, norm=LogNorm())#,vmin=2E3,vmax=3E3)
	plt.colorbar()
	return
##*

##* 
def targetDatafit(path,title='',lims=[],hotpx=False):
	"""`targetDatafit(path,title,lims,hotpx)`
	
	Function to open fits file and extract data.
	  It brings the path and exacts the data, giving a row image.
	  You can set a portion of image and also the correction for hotpx.

	  It calls the functions `hotpxRemove` and `showfits`.

	## Parameters

	`path` : str
	  path of the fits file to open
	`title` : str, optional
	  title of the image in `showfits`. Default is `''`
	`lims` : list, optional
	  list with the limits on `x` and `y` axis to get a portion of the image.
 	  The format is [ly,ry,lx,rx]:
 	     - ly -> left  y limit 
 	     - ry -> right y limit 
 	     - lx -> left  x limit 
 	     - rx -> right x limit
	     
	  Default is `[]`
	`hotpx` : bool, optional
	  if it's `True` the correction for hotpx will be.
	  Default is `False`

	## Returns

	`hdul` : astropy.hdulist_like
	  hdul list of the chosen fits file
	`data` : array_like
	  image matrix 
	"""
	# open the file
	hdul = fits.open(path)
	# print fits info
	hdul.info()

	# Data extraction
	# format -> data[Y,X]
	data = hdul[0].data
	if(lims!=[]):
		ly,ry,lx,rx = lims
		data = data[ly:ry,lx:rx]
	# hot px correction
	if(hotpx==True):
		data = hotpxRemove(data)

	# Spectrum image
	showfits(data,title)

	return hdul,data
##*

##*
def RotCor(data):
	"""`RotCor(data)`
	
	Function to correct the inclination, rotating the image.
	  It takes the maximum of each column and does a fit to find the angle with the horizontal.

	## Parameter

	`data` : array_like
	  image matrix

	## Returns

	`angle` : float
	  the angle of the slope
	`data_rot` : array_like
	  rotated image matrix 
	"""
	nx = len(data[0,:])
	mean_val = np.array([])
	for i in range(nx):
		tmp  = data[:,i]
		tmp2 = np.where(tmp == max(tmp))
		tmp2 = tmp2[0]
		if(len(tmp2)>1):
			tmp2 = tmp2.sum()/len(tmp2)
		mean_val = np.append(mean_val,tmp2)
	def fitlin(x,m,q):
		return x*m+q
	initial_values = [0.9,0.]
	for i in range(3):
		pop, pcov = curve_fit(fitlin,range(nx),mean_val,initial_values)
		m = pop[0]
		initial_values = pop

	angle = np.arctan(m)*180/np.pi   # degrees
	data_rot  = ndimage.rotate(data, angle, reshape=False)

	return angle, data_rot
##*


##* 
def initialization(target_file,calibration_file,jupiter_file,flat_file):
	"""`initialization(target_file,calibration_file,jupiter_file,flat_file)`

	Makes the calibration on Beta Lyr spectrum and calculates the flat correction.

	## Parameters

	`target_file` : str
	  name of Beta Lyr fits file
	`calibration_file` : str
	  name of lamp fits file in Beta Lyr folder
	`jupiter_file` : str
	  name of Jupiter fits file
	`flat_file` : str
	  name of Flat fits file

	## Returns

	`values` : array_like			
	  calibrated values array of Beta Lyr spectrum with flat
 	`stdLamp` : array_like
	  values array for the Lamp of Beta Lyr, used for calibration 
	`dataJup` : array_like		
	  rotated image matrix of Jupiter
	`angleJup` : array_like
	  the angle of slope for Jupiter data
	`Flat` : float			
	  flat correction value
 	`TransferFunc` : function
	  function to pass from px to lambda 	  	
	"""
	ly,ry = [650,1400]
	lx,rx = [760,-1]
	cut   = [ly,ry,lx,rx]

	##* Open target image
	print('Open the Target Fits File\n')
	# data[Y,X]
	hdul, data = targetDatafit(target_file,title='$\\beta$ Lyr',lims=cut,hotpx=True)
	hdul[0].header['OBJECT'] = 'Beta Lyr'
	# rotation
	angle, data = RotCor(data)
	nx = len(data[0,:]); ny = len(data[:,0])
	showfits(data,f'$\\beta$ Lyr - Rotated of {angle} degrees')
	# sum in a range of the spectrum
	values = np.array([data[555:574,j].sum() for j in range(nx) ])
	# plot
	fastplot(np.linspace(0,nx,nx),values,title='Spectrum without calibration and flat correction',labels=['x [px]','counts'])
	hdul.close()

	##* Open calibration lamp
	print('\n\nOpen the Calibration Fits File\n')
	hdul, Cdata = targetDatafit(calibration_file,title='Calibration Lamp',lims=cut)
	# rotation
	Cdata = ndimage.rotate(Cdata,angle,reshape=False)
	showfits(Cdata,f'Calibration Lamp - Rotated of {angle} degrees')
	# Spectrum
	Cvalues = Cdata[350,:]
	if(nx != len(Cdata[0,:])):
		print(f'Data Error -> Different Shape of Images after Rotation\nTarget\tLamp\n{nx}\t{len(Cdata[0,:])}')
		return 
	# plot
	fastplot(range(nx),Cvalues,title='Lamp spectrum',labels=['x [px]','counts'])
	## Fit
	fold = 'C:/Users/berna/Desktop/FISICA/ASTROFISICA/#OHP_France/26072022/'
	lA, lpx = np.loadtxt(fold+'betaLyr/calibration_lines.txt',unpack=True)
	def FitFun(x,p0,p1,p2,p3):
		return p0 + p1*x + p2*x**2 + p3*x**3
	initial_values = np.array([1e2,1.,0.,0.])
	for i in range(3):
		pop, pcov = curve_fit(FitFun,lpx,lA,initial_values)
		p0,p1,p2,p3 = pop
		initial_values = pop
	error = np.array([np.sqrt((data[i,:]).sum()) for i in range(4)])
	def TransferFunc(Lpx):
		return FitFun(Lpx,p0,p1,p2,p3)
	# plot the fit
	plt.figure()
	xx = np.linspace(0,max(lpx)+100,1000)
	plt.subplot(121)
	plt.title('Fit')
	plt.plot(lpx,lA,'.')
	plt.plot(xx,TransferFunc(xx))
	plt.subplot(122)
	plt.plot(lpx,abs(1-TransferFunc(lpx)/lA),'.')
	plt.plot([min(lpx),max(lpx)],[0,0],'--')

	# plot spectrum
	px = np.linspace(0,nx,nx)
	l = TransferFunc(px)
	fastplot(l,values,title='Calibrated Spectrum of $\\beta$ Lyr without flat correction',labels=['$\lambda$ [A]','counts'])

	hdul.close()

	##* Flat
	print('Open Flat Fits File\nFlat')
	hdulFlat, dataFlat = targetDatafit(fold+'giove/'+flat_file+'.fit',lims=cut)
	print('Jupiter')
	hdulJup, dataJup   = targetDatafit(fold+'giove/'+jupiter_file+'.fit',lims=cut)
	# Rot
	angleJup, dataJup = RotCor(dataJup)
	dataFlat = ndimage.rotate(dataFlat,angleJup,reshape=False)
	if(nx != len(dataFlat[0,:])):
		print(f'Data Error -> Different Shape of Images after Rotation\nTarget\Flat\n{nx}\t{len(dataFlat[0,:])}')
		return
	# calculate flat
	Flat = np.array([dataFlat[:,j].sum() for j in range(nx)])/dataFlat.sum()
	
	# Spectrum calibrated with flat
	fastplot(l,values/Flat,title='Calibrated Spectrum of $\\beta$ Lyr with flat correction',labels=['$\lambda$ [A]','counts'])

	hdulFlat.close()
	hdulJup.close()
	stdLamp = Cvalues
	return values, stdLamp, dataJup, angleJup, Flat, TransferFunc
##*

##*
def Correlation(values,stdLamp):
	corr = correlate(stdLamp,values)/np.sqrt((values**2).sum()*(stdLamp**2).sum())
	corr = max(corr)
	pos = lambda x : np.where( x == max(x))
	return
##*

##*
def OpenSpect():
	return
##*

