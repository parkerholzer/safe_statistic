import numpy as np
import pandas as pd
from wave_match import wave_match2
from scipy.optimize import curve_fit
def gaussfunc(x, mu, sigma):
    return np.exp(-((x - mu)**2)/(2*(sigma**2)))
def gauss1func(x, cntm, a1, mu1, sig1):
    return cntm - a1*gaussfunc(x, mu1, sig1)
def gauss2func(x, cntm, a1, a2, mu1, mu2, sig1, sig2):
    return cntm - a1*gaussfunc(x, mu1, sig1) - a2*gaussfunc(x, mu2, sig2)
def gauss3func(x, cntm, a1, a2, a3, mu1, mu2, mu3, sig1, sig2, sig3):
    return cntm - a1*gaussfunc(x, mu1, sig1) - a2*gaussfunc(x, mu2, sig2) - a3*gaussfunc(x, mu3, sig3)
from abfeature_functions import doppshift, lowerSNR
from statsmodels.api import OLS, WLS
from scipy.special import eval_hermite
def hermgauss(x, n, mu=0, sig=1):
    x = np.array(x)
    c = 1/np.sqrt(sig*(2**n)*np.math.factorial(n)*np.sqrt(np.pi))
    return c*eval_hermite(n, (x-mu)/sig)*np.exp(-((x - mu)**2)/(2*sig**2))
from multiprocessing import Pool



template = pd.read_csv("3651smoothtemp2.csv")
template = template[~np.isnan(template.Flux.values)]
#template = template.iloc[np.arange(0, template.shape[0], 10),:]

bestrv = 55290.2108
doppfact = 1+bestrv/299792458
Features = pd.read_csv("SOAP_Features.csv")
Features = Features[(Features["Wv_lbounds"] > 5000) & (Features["Wv_ubounds"] < template.Wavelength.values[-1])]
Features["Wv_lbounds"] = doppfact*Features.Wv_lbounds.values
Features["Wv_ubounds"] = doppfact*Features.Wv_ubounds.values
Features["MinWvl"] = doppfact*Features.MinWvl.values

print("Checkpoint 1")

w = np.where((template.Wavelength.values > 4990) & (template.Wavelength.values < 6710))[0]
wvl = template.Wavelength.values[w]
flx = template.Flux.values[w]

amps = []
mus = []
sigs = []
cntms = []
badftrs = []
for i in range(1,len(Features.Wv_lbounds.values)-1):
    if len(np.where((wvl >= Features.Wv_lbounds.values[i]) & (wvl <= Features.Wv_ubounds.values[i]))[0]) < 4:
        badftrs.append(i)
        continue
    #print(i)
    try:
        cntm_0 = 1.04
        w = np.where((wvl >= Features.Wv_lbounds.values[i-1]) & (wvl <= Features.Wv_ubounds.values[i-1]))[0]
        a1_0 = 1 - np.min(flx[w])
        mu1_0 = wvl[w][np.argmin(flx[w])]
        sig1_0 = (Features.Wv_ubounds.values[i-1] - Features.Wv_lbounds.values[i-1])/5
        w = np.where((wvl >= Features.Wv_lbounds.values[i]) & (wvl <= Features.Wv_ubounds.values[i]))[0]
        a2_0 = 1 - np.min(flx[w])
        mu2_0 = wvl[w][np.argmin(flx[w])]
        sig2_0 = (Features.Wv_ubounds.values[i] - Features.Wv_lbounds.values[i])/5
        w = np.where((wvl >= Features.Wv_lbounds.values[i+1]) & (wvl <= Features.Wv_ubounds.values[i+1]))[0]
        a3_0 = 1 - np.min(flx[w])
        mu3_0 = wvl[w][np.argmin(flx[w])]
        sig3_0 = (Features.Wv_ubounds.values[i+1] - Features.Wv_lbounds.values[i+1])/5
        w = np.where((wvl >= Features.Wv_lbounds.values[i-1]) & (wvl <= Features.Wv_ubounds.values[i+1]))[0]
        pars, cov = curve_fit(gauss3func, wvl[w], flx[w], p0 = [cntm_0, a1_0, a2_0, a3_0, mu1_0, mu2_0, mu3_0, sig1_0, sig2_0, sig3_0], bounds = ([0.9, 0,0,0,Features.Wv_lbounds.values[i-1],Features.Wv_lbounds.values[i], Features.Wv_lbounds.values[i+1], 0,0,0], [1.1,1,1,1,Features.Wv_ubounds.values[i-1],Features.Wv_ubounds.values[i], Features.Wv_ubounds.values[i+1], 5*sig1_0, 5*sig2_0, 5*sig3_0]))
        cntms.append(pars[0])
        amps.append(pars[2])
        mus.append(pars[5])
        sigs.append(pars[8])
    except:
        try:
            w = np.where((wvl >= Features.Wv_lbounds.values[i-1]) & (wvl <= Features.Wv_ubounds.values[i]))[0]
            pars, cov = curve_fit(gauss2func, wvl[w], flx[w], p0 = [cntm_0, a1_0, a2_0, mu1_0, mu2_0, sig1_0, sig2_0], bounds = ([0.9,0,0,Features.Wv_lbounds.values[i-1],Features.Wv_lbounds.values[i],0,0], [1.1,1,1,Features.Wv_ubounds.values[i-1],Features.Wv_ubounds.values[i], 5*sig1_0, 5*sig2_0]))
            cntms.append(pars[0])
            amps.append(pars[2])
            mus.append(pars[4])
            sigs.append(pars[6])
        except:
            try:
                w = np.where((wvl >= Features.Wv_lbounds.values[i]) & (wvl <= Features.Wv_ubounds.values[i+1]))[0]
                pars, cov = curve_fit(gauss2func, wvl[w], flx[w], p0 = [cntm_0, a2_0, a3_0, mu2_0, mu3_0, sig2_0, sig3_0], bounds = ([0.9,0,0,Features.Wv_lbounds.values[i],Features.Wv_lbounds.values[i+1],0,0], [1.1,1,1,Features.Wv_ubounds.values[i],Features.Wv_ubounds.values[i+1], 5*sig2_0, 5*sig3_0]))
                cntms.append(pars[0])
                amps.append(pars[1])
                mus.append(pars[3])
                sigs.append(pars[5])
            except:
                try:
                    w = np.where((wvl >= Features.Wv_lbounds.values[i]) & (wvl <= Features.Wv_ubounds.values[i]))[0]
                    pars, cov = curve_fit(gauss1func, wvl[w], flx[w], p0 = [cntm_0, a2_0, mu2_0, sig2_0], bounds = ([0.9,0,Features.Wv_lbounds.values[i],0],[1.1,1,Features.Wv_ubounds.values[i], 5*sig2_0]))
                    cntms.append(pars[0])
                    amps.append(pars[1])
                    mus.append(pars[2])
                    sigs.append(pars[3])
                except:
                    cntms.append(1)
                    amps.append(0)
                    mus.append(mu2_0)
                    sigs.append(sig2_0)

#Fit the first and the last features
w = np.where((wvl >= Features.Wv_lbounds.values[0]) & (wvl <= Features.Wv_ubounds.values[0]))[0]
a2_0 = 1 - np.min(flx[w])
mu2_0 = wvl[w][np.argmin(flx[w])]
sig2_0 = (Features.Wv_ubounds.values[0] - Features.Wv_lbounds.values[0])/5
w = np.where((wvl >= Features.Wv_lbounds.values[1]) & (wvl <= Features.Wv_ubounds.values[1]))[0]
a3_0 = 1 - np.min(flx[w])
mu3_0 = wvl[w][np.argmin(flx[w])]
sig3_0 = (Features.Wv_ubounds.values[1] - Features.Wv_lbounds.values[1])/5
w = np.where((wvl >= Features.Wv_lbounds.values[0]) & (wvl <= Features.Wv_ubounds.values[1]))[0]
pars, cov = curve_fit(gauss2func, wvl[w], flx[w], p0 = [1.04, a2_0, a3_0, mu2_0, mu3_0, sig2_0, sig3_0], bounds = ([0.9,0,0,Features.Wv_lbounds.values[0],Features.Wv_lbounds.values[1],0,0],[1.1,1,1,Features.Wv_ubounds.values[0],Features.Wv_ubounds.values[1], 5*sig2_0, 5*sig3_0]))


w = np.where((wvl >= Features.Wv_lbounds.values[-2]) & (wvl <= Features.Wv_ubounds.values[-2]))[0]
a2_0 = 1 - np.min(flx[w])
mu2_0 = wvl[w][np.argmin(flx[w])]
sig2_0 = (Features.Wv_ubounds.values[-2] - Features.Wv_lbounds.values[-2])/5
w = np.where((wvl >= Features.Wv_lbounds.values[-1]) & (wvl <= Features.Wv_ubounds.values[-1]))[0]
a3_0 = 1 - np.min(flx[w])
mu3_0 = wvl[w][np.argmin(flx[w])]
sig3_0 = (Features.Wv_ubounds.values[-1] - Features.Wv_lbounds.values[-1])/5
w = np.where((wvl >= Features.Wv_lbounds.values[-2]) & (wvl <= Features.Wv_ubounds.values[-1]))[0]
pars2, cov2 = curve_fit(gauss2func, wvl[w], flx[w], p0 = [1.04, a2_0, a3_0, mu2_0, mu3_0, sig2_0, sig3_0], bounds = ([0.9,0,0,Features.Wv_lbounds.values[-2],Features.Wv_lbounds.values[-1],0,0], [1.1,1,1,Features.Wv_ubounds.values[-2],Features.Wv_ubounds.values[-1], 5*sig2_0, 5*sig3_0]))
cntms = [pars[0]] + cntms + [pars2[0]]
amps = [pars[1]] + amps + [pars2[2]]
mus = [pars[3]] + mus + [pars2[4]]
sigs = [pars[5]] + sigs + [pars2[6]]

print("Checkpoint 2")

Features = Features.iloc[np.setdiff1d(np.arange(len(Features.Wv_lbounds.values)), np.array(badftrs)),:]
assert len(Features.Wv_lbounds.values) == len(cntms)

Features["Gauss_amp"] = np.array(amps)
Features["Gauss_mu"] = np.array(mus)
Features["Gauss_sig_true"] = np.array(list(Features.Gauss_sig.values))
Features["Gauss_sig"] = np.array(sigs)
Features["Continuum"] = np.array(cntms)

sig_cutoff = 2.5*np.percentile(Features.Gauss_sig.values, 75) - 1.5*np.percentile(Features.Gauss_sig.values, 25)
Features = Features[(Features.Gauss_sig.values <= sig_cutoff) & (Features.Gauss_amp.values > 0) & (Features.Gauss_sig.values >= 0.001) & (Features.Continuum.values > 0.98) & (Features.Continuum.values < 1.02) & (Features.Wv_ubounds.values - Features.Wv_lbounds.values > 6*Features.Gauss_sig.values)]

import glob
filenames = [f for f in glob.glob("3651spec/*_2.csv")]
filenames = np.array(filenames)[np.argsort([float(f.split("spec/")[1].split("ctd_2")[0].split("_")[1]) for f in filenames])]

SPECTRA = [pd.read_csv(f) for f in filenames]

for i in range(len(filenames)):
    SPECTRA[i]["Flux"] = SPECTRA[i]["Flux"]/SPECTRA[i]["Continuum"]
    SPECTRA[i]["Uncertainty"] = SPECTRA[i]["Uncertainty"]/SPECTRA[i]["Continuum"]

pixspc = template.Wavelength.values[1:] - template.Wavelength.values[:-1]
rightcutoffs = template.Wavelength.values[np.where(pixspc > np.percentile(pixspc, 75) + 15*(np.percentile(pixspc, 75) - np.percentile(pixspc, 25)))[0]]
rightcutoffs = np.hstack((rightcutoffs, template.Wavelength.values[-1]))
leftcutoffs = template.Wavelength.values[np.where(pixspc > np.percentile(pixspc, 75) + 15*(np.percentile(pixspc, 75) - np.percentile(pixspc, 25)))[0] + 1]
leftcutoffs = np.hstack((template.Wavelength.values[0], leftcutoffs))

print("Checkpoint 3")

def safefit(tau):
    wvl = []
    tempflx = []
    obsflx = []
    obsunc = []
    for i in range(len(leftcutoffs)):
        keep = np.where((SPECTRA[tau].Wavelength.values > leftcutoffs[i]) &
                        (SPECTRA[tau].Wavelength.values < rightcutoffs[i]))[0]
        tkeep = np.where((template.Wavelength.values >= leftcutoffs[i]) &
                        (template.Wavelength.values <= rightcutoffs[i]))[0]
        if len(tkeep) > 5 and len(keep) > 5:
            wvl = wvl + list(SPECTRA[tau].Wavelength.values[keep])
            obsflx = obsflx + list(SPECTRA[tau].Flux.values[keep])
            obsunc = obsunc + list(SPECTRA[tau].Uncertainty.values[keep])
            tempflx = tempflx + list(wave_match2(template.Wavelength.values[tkeep], template.Flux.values[tkeep], SPECTRA[tau].Wavelength.values[keep]))

    #obsunc = obsunc + list(np.sqrt(tempflx))

    nonans = np.where(~np.isnan(obsflx))[0]
    wvl = np.array(wvl)[nonans]
    obsflx = np.array(obsflx)[nonans]
    obsunc = np.array(obsunc)[nonans]
    tempflx = np.array(tempflx)[nonans]
    keep = np.where((wvl >= 5000) & (wvl <= 6700))[0]
    wvl = wvl[keep]
    obsflx = obsflx[keep]
    obsunc = obsunc[keep]
    tempflx = tempflx[keep]
    
    keep = []
    doppvar = np.array([])
    for i in range(len(Features.Gauss_mu.values)):
        w = np.where((wvl <= Features.Wv_ubounds.values[i]) & (wvl >= Features.Wv_lbounds.values[i]))[0]
        if len(w) > 3:
            keep = keep + list(w)
            diff = tempflx[w] - doppshift(wvl[w], tempflx[w], 7)
            mdl = OLS(diff, np.array([hermgauss(wvl[w], j, mu=Features.Gauss_mu.values[i], sig=Features.Gauss_sig.values[i]) for j in [0,1,2,3,4,5]]).T).fit()
            doppvar = np.hstack((doppvar, mdl.fittedvalues/7))
            
    Loadings = np.array(Features[["SOAPamp_0", "SOAPamp_2", "SOAPamp_3", "SOAPamp_4", "SOAPamp_5"]]).T
    X = np.array(list(doppvar))
    for j in [0,2,3,4,5]:
        v = np.array([])
        for i in range(len(Features.Wv_lbounds.values)):
            w = np.where((wvl <= Features.Wv_ubounds.values[i]) & (wvl >= Features.Wv_lbounds.values[i]))[0]
            if len(w) > 3:
                if j ==0:
                    v = np.hstack((v,Loadings[j,i]*hermgauss(wvl[w], j, mu=Features.Gauss_mu.values[i], sig=Features.Gauss_sig.values[i])))
                else:
                    v = np.hstack((v,Loadings[j-1,i]*hermgauss(wvl[w], j, mu=Features.Gauss_mu.values[i], sig=Features.Gauss_sig.values[i])))
        X = np.vstack((X, v))
        
    diff1 = tempflx[keep] - obsflx[keep]
    mdl1 = WLS(diff1, X.T, weights = 1/obsunc[keep]**2)
    influence = OLS(mdl1.endog, mdl1.exog).fit().get_influence().influence
    goodinf = np.where(np.abs(influence) <= np.median(influence) + np.percentile(influence, 75) + 50*(np.percentile(influence, 75) - np.percentile(influence, 25)))[0]
    mdl = WLS(diff1[goodinf], X.T[goodinf,:], weights = 1/obsunc[keep][goodinf]**2).fit()
    return mdl.params

avgcoefs = np.median(np.array(Pool(processes=19).map(safefit, np.arange(len(SPECTRA)))), axis=0)

print(avgcoefs)

def safefit(tau):
    wvl = []
    tempflx = []
    obsflx = []
    obsunc = []
    for i in range(len(leftcutoffs)):
        keep = np.where((SPECTRA[tau].Wavelength.values > leftcutoffs[i]) &
                        (SPECTRA[tau].Wavelength.values < rightcutoffs[i]))[0]
        tkeep = np.where((template.Wavelength.values >= leftcutoffs[i]) &
                        (template.Wavelength.values <= rightcutoffs[i]))[0]
        if len(tkeep) > 5 and len(keep) > 5:
            wvl = wvl + list(SPECTRA[tau].Wavelength.values[keep])
            obsflx = obsflx + list(SPECTRA[tau].Flux.values[keep])
            obsunc = obsunc + list(SPECTRA[tau].Uncertainty.values[keep])
            tempflx = tempflx + list(wave_match2(template.Wavelength.values[tkeep], template.Flux.values[tkeep], SPECTRA[tau].Wavelength.values[keep]))

    #obsunc = obsunc + list(np.sqrt(tempflx))
    nonans = np.where(~np.isnan(obsflx))[0]
    wvl = np.array(wvl)[nonans]
    obsflx = np.array(obsflx)[nonans]
    obsunc = np.array(obsunc)[nonans]
    tempflx = np.array(tempflx)[nonans]
    keep = np.where((wvl >= 5000) & (wvl <= 6700))[0]
    wvl = wvl[keep]
    obsflx = obsflx[keep]
    obsunc = obsunc[keep]
    tempflx = tempflx[keep]
    
    keep = []
    doppvar = np.array([])
    for i in range(len(Features.Gauss_mu.values)):
        w = np.where((wvl <= Features.Wv_ubounds.values[i]) & (wvl >= Features.Wv_lbounds.values[i]))[0]
        if len(w) > 3:
            keep = keep + list(w)
            diff = tempflx[w] - doppshift(wvl[w], tempflx[w], 7)
            mdl = OLS(diff, np.array([hermgauss(wvl[w], j, mu=Features.Gauss_mu.values[i], sig=Features.Gauss_sig.values[i]) for j in [0,1,2,3,4,5]]).T).fit()
            doppvar = np.hstack((doppvar, mdl.fittedvalues/7))
            
    Loadings = np.array(Features[["SOAPamp_0", "SOAPamp_2", "SOAPamp_3", "SOAPamp_4", "SOAPamp_5"]]).T
    X = np.array(list(doppvar))
    for j in [0,2,3,4,5]:
        v = np.array([])
        for i in range(len(Features.Wv_lbounds.values)):
            w = np.where((wvl <= Features.Wv_ubounds.values[i]) & (wvl >= Features.Wv_lbounds.values[i]))[0]
            if len(w) > 3:
                if j ==0:
                    v = np.hstack((v,Loadings[j,i]*hermgauss(wvl[w], j, mu=Features.Gauss_mu.values[i], sig=Features.Gauss_sig.values[i])))
                else:
                    v = np.hstack((v,Loadings[j-1,i]*hermgauss(wvl[w], j, mu=Features.Gauss_mu.values[i], sig=Features.Gauss_sig.values[i])))
        X = np.vstack((X, v))
        
    tempflx[keep] = tempflx[keep] - np.dot(X.T, avgcoefs.T)
    diff1 = tempflx[keep] - obsflx[keep]
    mdl1 = WLS(diff1, X.T, weights = 1/obsunc[keep]**2)
    influence = OLS(mdl1.endog, mdl1.exog).fit().get_influence().influence
    goodinf = np.where(np.abs(influence) <= np.median(influence) + np.percentile(influence, 75) + 50*(np.percentile(influence, 75) - np.percentile(influence, 25)))[0]
    print("Number of regression points = %d"%int(len(goodinf)))
    mdl = WLS(diff1[goodinf], X.T[goodinf,:], weights = 1/obsunc[keep][goodinf]**2).fit()

    A = np.identity(len(mdl.params))[1:,:]
    return [mdl.f_test(A).pvalue, mdl.f_test(A).fvalue[0][0]] + list(mdl.params) + list(mdl.tvalues)

rslts = np.array(Pool(processes=19).map(safefit, np.arange(len(SPECTRA))))
pvals = pd.DataFrame({"Filename": filenames, "Pvals": rslts[:,0], "SAFE": rslts[:,1], "b1": rslts[:,2], "b0": rslts[:,3], "b2": rslts[:,4],"b3": rslts[:,5],"b4": rslts[:,6],"b5": rslts[:,7], "t1": rslts[:,8],"t0": rslts[:,9],"t2": rslts[:,10],"t3": rslts[:,11],"t4": rslts[:,12],"t5": rslts[:,13]})

pvals.to_csv("3651SAFE.csv")
