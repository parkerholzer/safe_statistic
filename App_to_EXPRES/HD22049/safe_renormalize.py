import numpy as np
import pandas as pd
from abfeature_functions import findabsorptionfeatures
from multiprocessing import Pool

template = pd.read_csv("22049smoothtemp1.csv")

keep = np.where((template.Wavelength.values >= 4000) & (template.Wavelength.values <= 7000))[0]
wvbounds, minwvs, minflxs, maxflxs = findabsorptionfeatures(template.Wavelength.values[keep], template.Flux.values[keep], pix_range=7, minlinedepth=0.002, alpha=0.10, gamma=0.08)

import glob
filenames = [f for f in glob.glob("22049spec/*ctd.csv")]
filenames = np.array(filenames)[np.argsort([float(f.split("22049_")[1].split("ctd")[0]) for f in filenames])]
#filenames = np.array(filenames)[np.argsort([float(f.split(".csv")[0].split("chris/")[1]) for f in filenames])]
vels = pd.read_csv("22049spec/22049.txt")
vels["FILENAME"] = np.array([str(f) for f in vels.FILENAME.values])
filenames2 = np.array([f.split('ctd')[0].split('spec/')[1] + '.fits' for f in filenames])
notmissing = np.array([i for i in range(len(vels.FILENAME.values)) if vels.FILENAME.values[i] in filenames2])
vels = vels.iloc[notmissing,:]

assert np.array_equal(filenames2, vels.FILENAME.values)
SPECTRA = [pd.read_csv(f) for f in filenames]
vels = vels.V.values/100
vels = vels - np.mean(vels)

#doppfact = np.sqrt((1 - vels/299792458)/(1 + vels/299792458))
doppfact = 1 + vels/299792458
for i in range(len(vels)):
    SPECTRA[i]["Wavelength"] = SPECTRA[i]["Wavelength"]/doppfact[i]

filenames = [f.split(".csv")[0] + "_2.csv" for f in filenames]

def medn(wvl, flx, outputwvl, windowsize):
    w = np.where((wvl >= outputwvl - windowsize/2) & (wvl <= outputwvl + windowsize/2) & (~np.isnan(flx)))[0]
    iqr = np.percentile(flx[w], 75) - np.percentile(flx[w], 25)
    good = np.where(flx[w] >= np.max((np.percentile(flx[w], 25) - 1.0*iqr, 0.95)))[0]
    if len(good) <= 2:
        print("Count Error")
        return 1.0
    else:
        return np.median(flx[w][good])

def normalize(t):
    skeep = np.where((SPECTRA[t].Wavelength.values >= np.min(template.Wavelength.values[keep])) & (SPECTRA[t].Wavelength.values <= np.max(template.Wavelength.values[keep])) & (~np.isnan(SPECTRA[t].Flux.values)))[0]
    ftrs = np.hstack(np.array([np.where(((SPECTRA[t].Wavelength.values[skeep] >= bnds[0]) & (SPECTRA[t].Wavelength.values[skeep] <= bnds[1])) | ((SPECTRA[t].Wavelength.values[skeep] >= 4845) & (SPECTRA[t].Wavelength.values[skeep] <= 4880)) | ((SPECTRA[t].Wavelength.values[skeep] >= 5160) & (SPECTRA[t].Wavelength.values[skeep] <= 5200)) | ((SPECTRA[t].Wavelength.values[skeep] >= 5260) & (SPECTRA[t].Wavelength.values[skeep] <= 5290)) | ((SPECTRA[t].Wavelength.values[skeep] >= 6545) & (SPECTRA[t].Wavelength.values[skeep] <= 6585)))[0] for bnds in wvbounds]))
    cntm = np.setdiff1d(np.arange(len(skeep)), ftrs)
    continuum0 = np.array([medn(SPECTRA[t].Wavelength.values[skeep][cntm], SPECTRA[t].Flux.values[skeep][cntm], k, 8) for k in SPECTRA[t].Wavelength.values[skeep][cntm][np.arange(0, len(cntm), 20)]])
    SPECTRA[t]["Continuum"] = np.array([np.nan]*SPECTRA[t].shape[0])
    SPECTRA[t].Continuum.values[skeep] = np.interp(SPECTRA[t].Wavelength.values[skeep], SPECTRA[t].Wavelength.values[skeep][cntm][np.arange(0, len(cntm), 20)], continuum0)
    SPECTRA[t].to_csv(filenames[t])
    return "complete" #(SPECTRA[t], filenames[t])

output = Pool(processes=19).map(normalize, np.arange(len(SPECTRA)))

print(output)

#for o in output:
#    o[0].to_csv(o[1])
