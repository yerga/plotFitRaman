import numpy as np
import scipy.signal
from scipy.optimize import curve_fit
from detect_peaks import detect_peaks
CURVE = "Lorentzian"
SIGMAVALUE = None

def lorentzian(x, amp, ctr, wid):
    return amp*wid**2/((x-ctr)**2+wid**2)

def gaussian(x, height, center, width):
    return height*np.exp(-(x - center)**2/(2*width**2))

def func(x, *params):
    ''' *params of the form [center, amplitude, width ...] '''
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        if CURVE == "Gaussian":
            #print 'fitting to Gauss'
            y = y + gaussian(x, amp, ctr, wid)
        else:
            #print 'fitting to Lorentz'
            y = y + lorentzian(x, amp, ctr, wid)
    return y


def fit_curves(guess, func, x, y):
    popt, pcov = curve_fit(func, x, y, p0=guess, maxfev=14000, sigma=SIGMAVALUE)
    print('popt:', popt)
    fit = func(x, *popt)
    return (popt, fit)


# Returns the highest peaks found by the peak finding algorithms
def get_highest_n_from_list(a, n):
    return sorted(a, key=lambda pair: pair[1])[-n:]


def get_peaks(xs, ys, n):
    ind = detect_peaks(ys, mph=0, mpd=5, show=False)
    xpeaks = []
    ypeaks = []
    for peak in ind:
        xpeaks.append(xs[peak])
        ypeaks.append(ys[peak])

    peak_indexes_xs_ys = np.asarray([list(a) for a in list(zip(xpeaks, ypeaks))])
    return get_highest_n_from_list(peak_indexes_xs_ys, n)


def predict_and_plot_lorentzians(xs, ys, n_peaks_to_find=5, initialwidth=10):
    n_peaks = np.asarray(get_peaks(xs, ys, n_peaks_to_find))
    print'n_peaks: ', (n_peaks)
    guess = []

    for idx, xs_ys in enumerate(n_peaks):
        guess.append(xs_ys[0])  # peak X value
        guess.append(xs_ys[1])  # height
        guess.append(initialwidth)  # width
    print('Fit Guess: ', guess)

    params, fit = fit_curves(guess, func, xs, ys)
    ###params is the array of gaussian stuff, fit is the y's of lorentzians

    return (params, fit, ys, n_peaks)