import numpy as np
import glob
import sys
sys.path.append("../")
from abfeature_functions import lowerSNR, doppshift
from statsmodels.api import OLS, WLS
from scipy.special import eval_hermite
from multiprocessing import Pool
from pandas import read_csv, DataFrame
def hermgauss(x, n, mu=0, sig=1):
    x = np.array(x)
    c = 1/np.sqrt(sig*(2**n)*np.math.factorial(n)*np.sqrt(np.pi))
    return c*eval_hermite(n, (x-mu)/sig)*np.exp(-((x - mu)**2)/(2*sig**2))
def gaussfunc(x, mu, sigma):
    return np.exp(-((x - mu)**2)/(2*(sigma**2)))
def gauss1func(x, cntm, a1, mu1, sig1):
    return cntm - a1*gaussfunc(x, mu1, sig1)
def gauss2func(x, cntm, a1, a2, mu1, mu2, sig1, sig2):
    return cntm - a1*gaussfunc(x, mu1, sig1) - a2*gaussfunc(x, mu2, sig2)
def gauss3func(x, cntm, a1, a2, a3, mu1, mu2, mu3, sig1, sig2, sig3):
    return cntm - a1*gaussfunc(x, mu1, sig1) - a2*gaussfunc(x, mu2, sig2) - a3*gaussfunc(x, mu3, sig3)


#READ IN SPECTRA
filenames = [f for f in glob.glob("../../Methods/SOAP/*00.csv")]
phs = []
for f in filenames:
    phs.append(float(f.split("phase_")[1].split(".csv")[0]))
filenames = np.array(filenames)[np.argsort(np.array(phs))]

def readspec(fname):
    spec = read_csv(fname)
    spec["Flux"] = spec.Flux.values/np.max(spec.Flux.values)
    return spec

with Pool(processes=19) as p:
    #p = Pool(processes=15)
    SPECTRA = p.map(readspec, filenames)

print("Checkpoint 1")
template = read_csv("../../Methods/SOAP/integrated_spectrum_full_reso_spot_prot_25.0_size_0.010_lat_0_phase_-0.5000.csv")
template["Flux"] = template.Flux.values/np.max(template.Flux.values)

#FIND ABSORPTION FEATURES 
keep = np.where((template.Wavelength.values > 4470) & (template.Wavelength.values < 6700))[0]#4470 6750
wvl = template.Wavelength.values[keep]
flx = template.Flux.values[keep]

Features = read_csv("../../Methods/SOAP/SOAP_Features.csv")

#PROJECT THE DIFFERENCE SPECTRUM DUE TO A DOPPLER-SHIFT ONTO THE SPACE SPANNED BY THE HERMITE-GAUSSIAN FUNCTIONS
#OF DEGREE 0 THROUGH 5

def hg_dopp(i):
    w = np.where((wvl <= Features.Wv_ubounds.values[i]) & (wvl >= Features.Wv_lbounds.values[i]))[0]
    diff = flx[w] - doppshift(wvl[w], flx[w], 10)
    mdl = OLS(diff, np.array([hermgauss(wvl[w], j, mu=Features.Gauss_mu.values[i], 
                                           sig=Features.Gauss_sig.values[i]) for j in [0,1,2,3,4,5]]).T).fit()
    return [w, mdl.fittedvalues]

with Pool(processes=19) as p:
    #p = Pool(processes=15)
    doppoutput = np.array(p.map(hg_dopp, np.arange(len(Features.Wv_lbounds.values))))

print("Checkpoint 2")
        
keep2 = np.hstack(doppoutput[:,0])
doppvar = np.hstack(doppoutput[:,1])
print("n = :", len(keep2))

Loadings = np.array([Features.SOAPamp_0.values, Features.SOAPamp_2.values, Features.SOAPamp_3.values, Features.SOAPamp_4.values, Features.SOAPamp_5.values])


#BUILD THE DESIGN MATRIX

def build_design(j):
    v = np.array([])
    for i in range(len(Features.Wv_lbounds.values)):
        w = np.where((wvl <= Features.Wv_ubounds.values[i]) & (wvl >= Features.Wv_lbounds.values[i]))[0]
        if j ==0:
            v = np.hstack((v,Loadings[j,i]*hermgauss(wvl[w], j, mu=Features.Gauss_mu.values[i], 
                                                     sig=Features.Gauss_sig.values[i])))
        else:
            v = np.hstack((v,Loadings[j-1,i]*hermgauss(wvl[w], j, mu=Features.Gauss_mu.values[i], 
                                                       sig=Features.Gauss_sig.values[i])))
    return v

with Pool(processes=5) as p:
    #p = Pool(processes=5)
    X = np.array(p.map(build_design, np.array([0,2,3,4,5])))
    X = np.vstack((doppvar, X))

print("Checkpoint 3")
        
#ESTIMATE THE POWER OF THE TEST THROUGH 1000 SIMULATIONS AT EACH PHASE
    
Dpvals = []
Spvals = []
A = np.identity(X.shape[0])[1:,:]
def power_est(t):
    dopp_pvals = []
    spot_pvals = []
    for i in range(5000):
        diff1 = flx[keep2] - lowerSNR(doppshift(wvl[keep2], flx[keep2], 10*np.sin(2*np.pi*t/len(SPECTRA))),110)
        mdl1 = WLS(diff1, X.T, weights=1/flx[keep2]).fit()
        dopp_pvals.append(float(mdl1.f_test(A).pvalue))
        diff2 = flx[keep2] - lowerSNR(SPECTRA[t].Flux.values[keep][keep2], 150)
        mdl2 = WLS(diff2, X.T, weights=1/flx[keep2]).fit()
        spot_pvals.append(float(mdl2.f_test(A).pvalue))
    return [np.array(spot_pvals),np.array(dopp_pvals)]

with Pool(processes=19) as p:
    #p = Pool(processes=15)
    Pvals = np.array(p.map(power_est, np.arange(len(SPECTRA))))
print("Made it here!")
        
Dpvals = np.vstack(Pvals[:,1]).T
Spvals = np.vstack(Pvals[:,0]).T

DataFrame(Dpvals).to_csv("Dopp_Pvals_sn150_2.csv", index=False, header=False)
DataFrame(Spvals).to_csv("Spot_Pvals_sn150_p.csv", index=False, header=False)

