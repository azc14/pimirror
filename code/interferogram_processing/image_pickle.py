# |**********************************************************************;
# * Project           : MSci Project: PLAS-Smith-3
# *
# * Program name      : pickle_test.py
# *
# * Author            : Kelvin Chan
# *
# * Date created      : 15 Dec 2017
# *
# * Purpose           : Read the pickle file generated by image_analyse.py
# *
# * Revision History  : v1.0
# *
# |**********************************************************************;

import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.ticker as ticker
from matplotlib.widgets import Slider
from scipy.optimize import curve_fit
import plots

############### INPUT ###################

#Wavelength of the Laser (in microns)
wavelength = 543.5e-3

#Find the stroke at a specific voltage with extrapolation.
max_stroke_voltage = 150

#The name of the pickle file that was stored and created by the "image_analyse.py" file.
pickle_file = "data1.p"

#########################################

def format_fn(tick_val, tick_pos):
    """
    Used in plotdata function to have y-axis ticks in 'pi' intervals.
    """
    if tick_val == np.pi:
        return "$\pi$"
    elif tick_val == -np.pi:
        return "$-\pi$"
    elif tick_val == 0:
        return "$0$"
    else:
        return "$%i\pi$"%(tick_val/np.pi)

def largest_intensity(intensity_values_fit):
    """
    Finds the largest intensity value of the mean_intensities. Used for the setting graph axes limits.

    Param:
    intensity_values_fit = The 3D array that includes the mean and sd of the intensities.
    """
    max_intensity = 0
    for i in range(len(intensity_values_fit)):
        temp_max = max(intensity_values_fit[i][0])
        if temp_max > max_intensity:
            max_intensity = temp_max
    return max_intensity

def smallest_intensity(intensity_values_fit):
    """
    Finds the smallest intensity value of the mean_intensities. Used for the setting graph axes limits.

    Param:
    intensity_values_fit = The 3D array that includes the mean and sd of the intensities.
    """
    min_intensity = largest_intensity(intensity_values_fit)
    for i in range(len(intensity_values_fit)):
        temp_min = min(intensity_values_fit[i][0])
        if temp_min < min_intensity:
            min_intensity = temp_min
    return min_intensity

def linear_fit(x, y, yerr):
    """
    Fit a linear fit to the data points. Includes the stroke errors in the fit.
    """
    line_func = lambda V, m, c: m*V + c

    m0 = 0.016
    c0 = 0
    parameters, error = curve_fit(line_func, x, y, p0=[m0, c0], sigma=yerr, absolute_sigma=True, maxfev=200000)

    m, c = parameters
    y_fit = line_func(x, *parameters)
    return y_fit, m, c, error

def plot0():
    """
    Plots the intensity/interference pattern.
    """
    fig = plt.figure(figsize=(9, 6))
    ax00 = plt.subplot2grid((10, 1), (0, 0), rowspan=8)

    pattern00 = ax00.errorbar(x, intensity_values[0][0], yerr=intensity_values[0][1], fmt='g-', linewidth=1,
                              label='measurements')
    fitted_pattern00 = ax00.errorbar(x, intensity_values_fit[0][0], yerr=intensity_values_fit[0][1], fmt='b-',
                                     linewidth=1, label='fitted')
    title00 = ax00.set_title("$i$ = %i, $\lambda$ = %i, $stroke$ = %0.4f$\mu m$, $voltage$ = %0.2fV" % (0, fitted_cos_period[0][0], stroke_values[0][0], voltages[0]), fontsize=16)
    ax00.set_xlim(x[0], x[-1])
    ax00.set_ylim(smallest_intensity(intensity_values_fit), largest_intensity(intensity_values_fit))
    ax00.legend()
    ax00.tick_params(labelsize=16)
    ax00.set_ylabel('$Intensity$', fontsize=18)
    ax00.set_xlabel('$Distance$ $(Pixel)$', fontsize=18)
    ax00.legend(fontsize=18)

    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
    slide = Slider(ax_slider, 'i', 0, no_of_images - 1, valinit=0, valfmt='%i')

    plot_list = plots.Plots()

    plot_list.adderrorbar(pattern00)
    plot_list.adderrorbar(fitted_pattern00)

    def update(val):
        """
        Allows the change of interference pattern with the slider.
        """
        val = int(val)
        plot_list.removeerrorbar(n=-1)
        plot_list.removeerrorbar(n=-1)
        pattern00 = ax00.errorbar(x, intensity_values[val][0], yerr=intensity_values[val][1], fmt='g-', linewidth=1,
                                  label='measurements')
        fitted_pattern00 = ax00.errorbar(x, intensity_values_fit[val][0], yerr=intensity_values_fit[val][1], fmt='b-',
                                         linewidth=1, label='fitted')
        plot_list.adderrorbar(pattern00)
        plot_list.adderrorbar(fitted_pattern00)
        title00.set_text(
            "$i$ = %i, $\lambda$ = %0.5f, $stroke$ = %0.4f$\mu m$, $voltage$ = %0.2fV" % (val, fitted_cos_period[0][val], stroke_values[0][val], voltages[val]))
        plt.draw()
        return

    slide.on_changed(update)
    return slide

def plot1():
    """
    Plots the graph of stroke against voltage.
    """
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    phase_down_fit_plot = ax.plot(voltages[num_up - 1:], phase_fit_down, 'k--', label='Decreasing voltage (fit)')
    phase_up_fit_plot = ax.plot(voltages[:num_up], phase_fit_up, 'g--', label='Increasing voltage (fit)')
    phase_down = ax.errorbar(voltages[num_up:], stroke_values[0][num_up:], yerr=stroke_values[1][num_up:], fmt='kx', ms=6,
                             label='Decreasing voltage (measured)')
    phase_up = ax.errorbar(voltages[:num_up], stroke_values[0][:num_up], yerr=stroke_values[1][:num_up], fmt='gx', ms=6,
                           label='Increasing voltage (measured)')

    ax.tick_params(labelsize=16)
    tick_spacing_x = 10
    tick_spacing_y = 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_y))
    ax.set_ylabel('$Stroke$ $(\hspace{0.1} \mu m)$', fontsize=18)
    ax.set_xlabel('$Voltage$ $(V)$', fontsize=18)
    ax.legend(fontsize=13)

############### SCRIPT ####################

data = pickle.load(open(pickle_file, "rb"))

no_of_images, filenumbers, no_of_repeats, vertical_fringes, \
voltages, amp_values, k_values, phase_values, offset_values, original_phases, intensity_values, intensity_values_fit = data[:]

fitted_cos_period = (2*np.pi)/np.array(k_values)

num_up = int(no_of_images / 2) + 1

if vertical_fringes == False:
    x = np.arange(1024)
else:
    x = np.arange(1280)

stroke_values = np.array(phase_values) * wavelength/(2*np.pi)

phase_fit_up, grad, const, error = linear_fit(x=voltages[:num_up], y=stroke_values[0][:num_up], yerr=stroke_values[1][:num_up])
phase_fit_down, temp1, temp2, err = linear_fit(x=voltages[num_up-1:], y=stroke_values[0][num_up-1:], yerr=stroke_values[1][num_up-1:])

m_err = error[0, 0]**0.5
c_err = error[1, 1]**0.5

stroke = lambda V: grad*V + const
stroke_err = lambda V, m_err, c_err: np.sqrt((m_err*V)**2 + c_err**2)
delta_stroke_err = lambda V1, V2, m_err, c_err: np.sqrt(stroke_err(V1, m_err, c_err)**2 + stroke_err(V2, m_err, c_err)**2)

print("INFORMATION OF INCREASING VOLTAGE FIT")
print("Gradient: %f +- %f" %(grad, m_err))
print("Stroke at 0V: %f +- %f" %(const, c_err))
print("Max Stroke = %f +- %f" %(stroke(max_stroke_voltage)-stroke(0), delta_stroke_err(max_stroke_voltage, 0, m_err, c_err)))

slide = plot0()
plot1()
plt.show()