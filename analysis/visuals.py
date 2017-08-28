from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, pylab

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_b(x, x0, a, b):
    return a*sigmoid(x-x0) + b

def sigprime(x, x0, a):
    return a*sigmoid(x-x0)*(1 - sigmoid(x-x0))

def gaus(x, x0, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp( (pow(x-x0, 2)) / (-2*pow(sigma, 2)) )

def gaus_b(x, A, x0, sigma, b):
    return A*gaus(x, x0, sigma) + b

def LSE(x, y, a, x0, sigma, b, err):
    return sum( [(pow(gaus_b(ix, a, x0, sigma, b) - iy, 2))/err for ix, iy in zip(x, y)])/len(y)

def Norm(X):
    m = X.mean()
    s = X.std()
    return (X-m)/s

def GetBkg(y):
    mu = y.mean()
    std = y.std()
    y = np.array([i for i in y if i < mu + 3*std])
    return y.mean(), y.std()

def GetSeed(x, y):    
    mean = x[np.argmax(y)]
    peak = y.max()
    minFW = 1000
    for arg, i in enumerate(y):
        if abs(i - peak/2) < minFW:
            minFW = abs(i - peak/2)
            argFW = arg
    sigma = abs(x[argFW]-mean) / 2.355    
    area = sigma*peak
    return area, mean, sigma

def GetVars(x, y, inds, bkg):
    areas = []
    means = []
    sigmas = []
    bkgs = []
    for i, ind in enumerate(inds):
        imin = ind-50 if ind>50 else 0
        imax = ind+50 if ind<len(x)-50 else len(x)-1
        xfit, yfit = x[imin:imax], y[imin:imax]
        area, mean, sigma = GetSeed(xfit, yfit)
        popt, pcov = curve_fit(gaus_b, xfit, yfit, p0=[area, mean, sigma, bkg])
        area, mean, sigma, bkg = popt
        areas.append(round(area, 3))
        means.append(round(mean, 3))
        sigmas.append(round(sigma, 3))
        bkgs.append(round(bkg, 2))
    areas = np.array(areas)
    means = np.array(means)
    sigmas = np.array(sigmas)
    bkgs = np.array(bkgs)
    return areas, means, sigmas, bkgs

def TestFit(x, y, temp):
    d0 = round(min(x), 2)
    d = round(max(x), 2)

    area, mean, sigma = GetSeed(x, y)
    bkg, err = GetBkg(y)
    popt, pcov = curve_fit(gaus_b, x, y, p0=[area, mean, sigma, bkg])

    area, mean, sigma, bkg = popt
    Yhat = np.array([gaus_b(ix, area, mean, sigma, bkg) for ix in x])
    res = Yhat - y
    chi2 = LSE(x, y, area, mean, sigma, bkg, err)
    
    fig = plt.figure(figsize=(8,7))
    gs = gridspec.GridSpec(2, 1,  height_ratios=[5, 1]) 

    ax1 = plt.subplot(gs[0])
    plt.plot(x, y, 'k.:', label='data T = {} K'.format(temp))
    plt.plot(x, gaus_b(x, *popt), 'r+:', label='fit T = {} K'.format(temp))
    plt.legend()
    plt.title('Peak at {} - {} A for T = {} K'.format(d0, d, temp))
    frame = pylab.gca()
    frame.axes.get_xaxis().set_ticks([])
    plt.ylabel('Intensity')
    plt.annotate('Fit Results for T = {} K:\nA = {}\n$\mu$ = {}\n$\sigma$ = {}\nChi2={}'.format(temp,
                                                                                       round(area,2),
                                                                                       round(mean,2),
                                                                                       round(sigma,4),
                                                                                       round(chi2, 2) ),
                 xy=(d0, max(y)*.7))

    ax2 = plt.subplot(gs[1])
    plt.errorbar(x, res, yerr=err, fmt='b.')
    plt.plot([d0, d], [0, 0])
    plt.xlabel('dspacing (Angstrom)')
    plt.ylabel('Residues')
    plt.subplots_adjust(hspace=.003)
    plt.show()
    return
