import numpy as np
import scipy.interpolate as interpolate


def wave_match(w1,s1,w2,s2,verbose=False):
    """Match template and observation wavelengths

    Parameters
    ----------
    w1, s1 : ndarray
        Observed spectrum
    w2, s2 : ndarray
        Template spectrum
    verbose: bool, optional
        If True, print additional information statements
    
    Returns
    -------
    w1_fine, s1_fine : ndarray
        Re-scaled observed spectrum
    w2_fine, s2_fine : ndarray
        Re-scaled template spectrum
    """
    dw1 = w1[1:] - w1[:-1]
    wmid = (w1[0] + w1[-1])/2.
    
    dw2 = w2[1:] - w2[:-1]
    
    # Determine new uniformly spaced model wavelength scale
    min_disp = min(dw2)                               # Smallest spacing in template spectrum
    deltaw = w2[-1] - w2[0]                           # Wavelength range
    nx = (int(deltaw/float(min_disp))+2)       # Number of pixels in uniform scale
    if nx%2==0: nx+=1                                 # nx should be odd
    w2_fine = w2[0] + deltaw * np.arange(nx)/(nx-1.)  # new wavelength scale (x_seg)
    
    s1_range=np.logical_and(w2_fine>min(w1), w2_fine<max(w1)) # Find range of observed spectrum
    w1_fine = w2_fine[s1_range]                               # Constrict template to wavelength range of observed spectrum
    
    if verbose:
        print("Smallest wavelength spacing in template is: {}".format(min_disp))
        print("Number of pixels in new uniform scale is: {}".format(nx))

    # Interpolate template spectrum onto the uniform wavelength scale
    f = interpolate.interp1d(w2,s2,kind="cubic",fill_value=0) # Template spectrum as interpolation template
    s2_fine = f(w2_fine)                                      # Interpolate onto finer wavelength scale
    f = interpolate.interp1d(w1,s1,kind="cubic",fill_value=0) # Observed spectrum as interpolation template
    s1_fine = f(w1_fine)                                      # Note: w2_fine and w1_fine should be the same

    return w1_fine, s1_fine, w2_fine, s2_fine

def wave_match2(wvl1, flx1, targetwvl):
    wvlrng = np.where((targetwvl >= np.min(wvl1)) & (targetwvl <= np.max(wvl1)))[0]
    f = interpolate.interp1d(wvl1, flx1, kind="cubic", fill_value=0)
    return f(targetwvl[wvlrng])
