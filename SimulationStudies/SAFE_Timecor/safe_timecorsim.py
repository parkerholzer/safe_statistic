import numpy as np
import pandas as pd
import h5py
import scipy.interpolate as interpolate
from abfeature_functions import lowerSNR
from astropy.io import fits

def broaden(wvl, flx, pcount, step, resol):
    fwhm = wvl[pcount]/resol
    disp = wvl[pcount+1] - wvl[pcount]
    nhalf = int(3.3972872*0.5*fwhm/disp)
    gpro = np.sqrt(np.log(2)/np.pi) * disp / (0.5*fwhm) * np.exp(-((np.sqrt(np.log(2))/(0.5*fwhm))*disp*(np.arange(2*nhalf+1) - nhalf))**2)
    if pcount+step <= len(wvl):
        w2 = wvl[pcount+step-1]
    else:
        w2 = wvl[-1]
    ind_pix = np.where((wvl >= wvl[pcount]) & (wvl <= w2))[0]
    step = len(ind_pix)
    ssnip = flx[ind_pix]
    if(pcount == 0):
        spad = np.array([ssnip[0]]*(nhalf+2) + list(ssnip) + list(flx[np.arange(pcount+step-1, pcount+step+nhalf+1)]))
    elif pcount+step+nhalf+2 <= len(wvl):
        spad = np.array(list(flx[np.arange(pcount - nhalf - 3, pcount-1)]) + list(ssnip) + list(flx[np.arange(pcount+step-1, pcount+step+nhalf+1)]))
    else:
        spad = np.array(list(flx[np.arange(pcount - nhalf - 3, pcount-1)]) + list(ssnip) + [ssnip[-1]]*(nhalf+2))
    s_conv = np.convolve(spad, gpro)
    s_conv = s_conv[int((len(s_conv)-step)/2):int(len(s_conv) - int((len(s_conv)-step)/2))]
    return s_conv

def wave_match(wvl1, flx1, targetwvl):
    wvlrng = np.where((targetwvl >= np.min(wvl1)) & (targetwvl <= np.max(wvl1)))[0]
    f = interpolate.interp1d(wvl1, flx1, kind="cubic", fill_value=0)
    return f(targetwvl[wvlrng])

spec = h5py.File("chris/res-1000-1years_full_id1.h5", "r")
tau = np.arange(366)
tauceti = fits.open("10700spec/10700_191007.1130.fits")

wvl = []
wmin = tauceti[1].data[0][12][0]
for od in range(len(tauceti[0].data[:,0])):
    nonan = np.where(~np.isnan(tauceti[0].data[od,:]))[0]
    keep = np.where(tauceti[1].data[od][12][nonan] > wmin)[0]
    wvl = wvl + list(tauceti[1].data[od][12][nonan][keep])
    wmin = np.max(np.array(wvl))
wvl = np.array(wvl)
wvl = wvl[(wvl > 5000) & (wvl < 6500)]

wvl2 = np.array(spec['lambdas'])
keep = np.where((wvl2 >= 4990) & (wvl2 <= 6510))[0]
tmp = wave_match(wvl2[keep], np.array(spec['quiet']/np.max(spec['quiet']))[keep], wvl)
for t in tau:
    flx0 = np.array(spec['active'][t,keep])
    flx = np.hstack([list(broaden(wvl2[keep], flx0, p, 325, 137500)) 
                     for p in np.arange(0, len(keep), 325)])
    flx = lowerSNR(wave_match(wvl2[keep], flx/np.max(flx), wvl), 250)
    s = pd.DataFrame({"Wavelength": wvl, "Flux": flx, "Uncertainty": 1/np.sqrt(tmp)})
    s.to_csv("chris2/chrissim_%d.csv"%int(t))
temp = pd.DataFrame({"Wavelength": wvl, "Flux": tmp})
temp.to_csv("chris2/chrissim_template.csv")
