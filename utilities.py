# -*- coding: utf-8 -*-
"""
Descripcion: Miscellaneous of functions used for calculating the uncertainty of the inference.
@author: Pablo Rodríguez Beltrán
"""

import numpy as np
from scipy.signal import fftconvolve as sgfftconv
from astropy.convolution import convolve_fft 
from astropy.modeling.models import Sersic2D
from astropy.modeling.models import Gaussian2D

##### Creates 2D Sersic model for the mean galaxy
def sersic_image(data, amp, sersicInd, theta, Reff, elip, xCenter, yCenter):
    row,col = np.shape(data)
    x,y = np.meshgrid(np.arange(row), np.arange(col))
    xbias=0.5
    ybias=0.5
    x0 = (xCenter - xbias)
    y0 = (yCenter - ybias)
    
    mod = Sersic2D(amplitude = amp, r_eff = Reff, n=sersicInd, x_0=x0, y_0=y0, ellip=elip, theta=theta)
    img = mod(x, y)
    return img

##### Creates the PSF as a 2D Gaussian
def gaussian_PSF(radius):
    row=radius*2.+1
    col=radius*2.+1
    xmean, ymean = [radius, radius]
    xbias=0.
    ybias=0.
    x,y = np.meshgrid(np.arange(row), np.arange(col))

    sig=radius/3.
    mod = Gaussian2D(amplitude = 1, x_mean= xmean - xbias,
    y_mean=ymean-ybias, x_stddev=sig, y_stddev=sig, theta=0,
    cov_matrix=None)
    img = mod(x,y)
    img = img/np.sum(img)
    return img

##### Resizes the PSF respect to the size of the galaxy image
def PSF2Image(data,PSF):
    row,col=np.shape(data)
    data_out = np.zeros((col,row))
    xmean,ymean = [row/2.,col/2.]
    bias=0.5
    data_out[int(xmean),int(ymean)] = 1.
    PSFnew = convolvenorm_fft_wrap(data_out, PSF)
    return PSFnew

##### Adds Poissionian noise for the readout noise
def AddNoise(data):
    row,col=np.shape(data)
    data_out = np.zeros((col,row))
    for icol in range (0,col):
      for irow in range (0,row):
          mean_in= data[icol,irow]
          data_out[icol,irow] = np.random.poisson(mean_in)
    return data_out

##### Adds Gaussian noise for the fluctuation of the stellar population luminosity
def SBF_noise(data, SBF):
    row,col=np.shape(data)
    data_out = np.zeros((col,row))
    for icol in range (0,col):
      for irow in range (0,row):
          mean_in= data[icol,irow]
          SBF_in=np.sqrt(SBF * mean_in)
          data_out[icol,irow] = np.random.normal(mean_in,SBF_in)
    return data_out

##### Convolution functions
def convolvenorm_fft(data,kernel):
    newima = sgfftconv(data,kernel,mode="same")
    return newima
def convolvenorm_fft_wrap(data,kernel):
    newima = convolve_fft(data, kernel, boundary = "wrap")
    return newima

##### Calculating the power spectrum image
def PowerSpecData(data):
    num_rows,num_cols=np.shape(data)
    center_data=[int(num_rows/2.),int(num_cols/2.)]

    xNormFF = np.sqrt(num_cols*num_rows)

    F_data = np.fft.fft2(data)/xNormFF
    F_data = np.fft.fftshift(F_data)
#     PS_data = F_data * np.matrix.conjugate(F_data)
    PS_data = np.abs( F_data )**2
    return PS_data

##### Azimuthally averaging a power spectrum image (creating a radial profile)
def PowerSpecArray(data):
    num_rows,num_cols=np.shape(data)
    center_data=[int(num_rows/2.),int(num_cols/2.)]
    PS_data = PowerSpecData(data)
    PS_array = radial_profile(PS_data,center_data)
    return PS_array

##### Azimuthally averaging an image
def radial_profile(data, center):
    xbias=0.5
    ybias=0.5
    x0 = (center[0] - xbias)
    y0 = (center[1] - ybias)
    y, x = np.indices((data.shape))
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    r = r.astype(np.int)
    
    tbin = np.bincount(r.ravel(), data.ravel()) 
    nr = np.bincount(r.ravel()) 
    radialprofile = tbin / nr 
    return radialprofile

##### Creates image of an annular mask
def RadialMask(data,kmin,kmax,xCenter,yCenter):
    row,col=np.shape(data)
    mask = np.zeros((col,row))
    xbias=0.5
    ybias=0.5
    x0 = (xCenter - xbias)
    y0 = (yCenter - ybias)
    for icol in range (0,col):
      for irow in range (0,row):
        adum = (icol - y0)**2. + (irow - x0)**2.
        if(adum <= kmax**2. and adum >= kmin**2.):
#          mask[icol,irow] = int(np.sqrt(adum))
            mask[icol,irow] = 1.
    return mask

###########################################################################################################################################
