#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:31:54 2018

@author: alicecao
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '../')
from profiler import Profile

# definition of supergaussian function

def supergaussian(x, x_0 = 1050., sd = 675., P = 4., A = 0.75):
    e = np.e
    exponent = ((x-x_0)**2)/(2*(sd**2))
    f = A*e**((-1)*exponent**P)
    return f

# =============================================================================
# # obtain raw data from file
# =============================================================================
blurred_focal_spot = cv2.imread('../d_cam/blurred/blurred.tif',0)
spot_profile = Profile(blurred_focal_spot, [1024, 1024]).plot_profile()

# =============================================================================
# # generate a supergaussian fit
# =============================================================================
k = np.arange(0.,2048.)
k_y = supergaussian(k, x_0 = 1024., sd = 700., P = 10., A = 0.9)

# supergaussian kernel

s_g = k_y.reshape(2048,1)
supergaussian_spot = s_g*s_g.T

# =============================================================================
# # compare the profiles of data and generated supergaussian
# =============================================================================
plt.figure(1)
plt.rcParams.update({'font.size': 20})
plt.title("Ideal Supergaussian intensity profile fit")
plt.xlabel("Pixel")
plt.ylabel("Normalised intensity")
plt.plot(k, k_y, "go-", markersize=2, label="Supergaussian fit" )
plt.plot(k, spot_profile[1], "ro-", markersize=2, label="Actual vertical intensity profile")
plt.plot(k, spot_profile[0], "bo-", markersize=2, label="Actual horizontal intensity profile")
plt.legend()

plt.figure(2)
img = np.zeros((2048, 2048))
cv2.circle(img,(1024,1024), 1024, 255, -1)
rows,cols = img.shape

for i in range(72):
    M = cv2.getRotationMatrix2D((cols/2,rows/2),5,1)
    img = cv2.warpAffine(img*k_y,M,(cols,rows))

plt.imshow(img, cmap="gray")
plt.imsave("supergaussian_spot.png", img)



