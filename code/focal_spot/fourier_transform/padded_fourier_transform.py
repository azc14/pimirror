# |**********************************************************************;
# * Project           : MSci Project: PLAS-Smith-3
# *
# * Program name      : padded_fourier_transform.py
# *
# * Author            : Alice Cao
# *
# * Date created      : 4 Feb 2018
# *
# * Purpose           : Perform a 2D spatial Fourier transform of focal spot data.
# *                     The data must be padded before performing the transform in order to produce a high resolution magnitude spectrum.
# *
# * Revision History  : v1.0
# *
# |**********************************************************************;

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import trapz

import sys
sys.path.insert(0, '../')
from profiler import Profile

supergaussian_spot = cv2.imread('../supergaussian_fit/supergaussian_spot.png',0)
blurred_spot = cv2.imread('../d_cam/blurred/blurred.tif',0)

supergaussian_spot_ft = cv2.imread('supergaussian_spot_ft_colour.png',0)
blurred_spot_ft = cv2.imread('actual_ft.png',0)


def pad_transform(img, b=0):
    
    padded = cv2.copyMakeBorder(img, b,b,b,b, cv2.BORDER_CONSTANT, value=0)
    f = np.fft.fft2(padded)
    fshift = np.fft.fftshift(f)
    mag_spec = np.abs(fshift)

    plt.rcParams.update({'font.size': 20})
    
    plt.figure(1)
    plt.subplot(121),plt.imshow(padded)
    plt.title('Spatial profile'), plt.xticks([]), plt.yticks([])

    plt.subplot(122),plt.imshow(mag_spec)
    plt.title('Fourier transform'), plt.xticks([]), plt.yticks([])
    
#    plt.imsave("actual_ft.png", mag_spec)
    
    return padded, mag_spec

pad_transform(blurred_spot, b=5000)
    
print(np.where(blurred_spot_ft==blurred_spot_ft.max()))
    
x=np.arange(200)
plt.figure(1)
plt.rcParams.update({'font.size': 20})
plt.title("1D intensity profile: ideal spot")
ideal_spot_profile_0, ideal_spot_profile_1  = Profile(supergaussian_spot_ft, [101,102]).plot_profile()

area_ideal = trapz(ideal_spot_profile_0, dx=1)
area_ideal_peak = trapz(ideal_spot_profile_0[93:111], dx=1)
print("% energy in peak =", 100*area_ideal_peak/area_ideal)

plt.plot(x[60:140],ideal_spot_profile_0[60:140], dashes=[6, 2], color="blue", label="L/R scan", lw=3)
plt.plot(x[60:140],ideal_spot_profile_1[60:140], label="T/B scan", lw=3)
plt.xlabel("Pixel")
plt.ylabel("Normalised intensity")
plt.legend(loc=1)

plt.figure(2)
plt.title("1D intensity profile: actual spot")
real_spot_profile_0, real_spot_profile_1  = Profile(blurred_spot_ft, [100,98]).plot_profile()

area_actual = trapz(real_spot_profile_0, dx=1)
area_actual_peak = trapz(real_spot_profile_0[91:105], dx=1)
print("% energy in actual peak =", 100*area_actual_peak/area_actual)

plt.plot(x[60:140], real_spot_profile_0[60:140], dashes=[6, 2], color="red", label="L/R scan", lw=3)
plt.plot(x[60:140], real_spot_profile_1[60:140], color="red", label="T/B scan", lw=3)
plt.xlabel("Pixel")
plt.ylabel("Normalised intensity")
plt.legend(loc=1)


