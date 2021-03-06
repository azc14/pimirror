﻿focal spot analysis v.1.0 - 06/06/2018



The files and folders in this directory are used for data collection from the Adafruit 16-Channel 12-bit PCA9685 PWM/Servo Driver
and Adafruit ADS1115 16-Bit ADC - 4 Channel with Programmable Gain.




Required modules:


All critical libraries are outlined in requirements.txt.
Please refer to https://stackoverflow.com/questions/7225900/how-to-install-packages-using-pip-according-to-the-requirements-txt-file-from-a to see how to install the packages using pip.


The code in this directory also requires openCV to run: https://opencv.org/





————————————————————————————————



Modules:
	profiler.py
		
		* This module takes an input image array with a given centre and plots the pixel intensity value over the pixel number



Data directories:
	brighter
		
		* The images in this directory are of the focal spot of Cerberus in oscillator mode. The images have been enhanced by rescaling the intensity values to range from 0-255.

	

		d_cam
		
		* spot.tiff is an image of the Cerberus spatial intensity profile after the apodizer.
		
		* intensity_profile_comparison.png is an intensity profile of spot.tiff generated using profiler.py.
	
	

		fourier_transform
		
		* padded_fourier_transform.py is a module which pads a focal spot image with 0s, giving a border of specified thickness. This padding is required in order to produce a high resolution fourier transformed spot.
		
		* This module can also return a value for the area under the intensity profile of the fourier transformed image, which is a measure of how much energy is contained within the spot.	
		
		* actual_ft.png is a fourier transform of ../dcam/spot.tiff, produced with a border 5000 pixels thick, and then cropped.
		
		* ideal_supergaussian_spot_ft_colour.png is a fourier transform, using a 5000 pixel border, of the ideal supergaussian spot, found in supergaussian_fit/supergaussian_spot.png

	
	


		raw
		
		* raw images taken at the focal point of the oscillator beam.

	


		sumix
		
		* square-cropped raw images taken at the focal point of the oscillator beam.

	

		supergaussian_fit
 		
		* The supergaussian.py module was used to generate a supergaussian fit of the spatial intensity profile data.
		
		* The resulting ideal supdergaussian spot can be found as supergaussian_spot.png.
		
		* supergaussian_fit.png shows the fit function superimposed on the actual measured intensity profile.
		
		* actual_spot_ft_profile.png and ideal_spot_ft_profile.png show the intensity plots (generated by ../profiler.py) of the fourier transforms of the measured spatial intensity profile and the ideal spatial profiles, respectively.
	



————————————————————————————————



e-mail: alice.cao14@imperial.ac.uk
	kelvin.chan14@imperial.ac.uk