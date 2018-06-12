# |**********************************************************************;
# * Project           : MSci Project: PLAS-Smith-3
# *
# * Program name      : profiler.py
# *
# * Author            : Alice Cao
# *
# * Date created      : 4 Feb 2018
# *
# * Purpose           : Takes a horizontal and vertical slice across an input image, with specified centre,
# *                     and plots pixel value against pixel number. Used for plotting intensity profiles of focal spot images.
# *
# * Revision History  : v1.0
# *
# |**********************************************************************;

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

class Profile:
    def __init__(self, image=[], centre=[]):
        self.__image = np.array(image)
        self.__centre = np.array(centre)
        
    def img(self):
        return self.__image
    
    def cntr(self):
        return self.__centre
        
    def plot_profile(self, draw=False):
        img = self.img()
        
        centre_x = int(self.cntr()[0])
        centre_y = int(self.cntr()[1])
        
        pixels_x = np.arange(0,img.shape[0])
        pixels_y = np.arange(0,img.shape[1])
        
        plt.rcParams.update({'font.size': 20})
        
        x = img[centre_x]/img[centre_x].max()
        y = img.T[centre_y]/img.T[centre_y].max()
        
        if draw==True:
        
            plt.plot(pixels_x, x, "bo-", label="scan L/R", markersize=2, lw=1.5)
            plt.plot(pixels_y, y, "ro-", label="scan T/B", markersize=2, lw=1.5)
    
            plt.title("Intensity profile")
            plt.xlabel("pixel number")
            plt.ylabel("Normalised intensity")
            plt.legend()
            
            plt.show()
            return x, y
        
        else:
            return x, y
    
    
    
    


