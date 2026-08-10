'''
Code and comment from Parthapratim Mahapatra

This paper (https://arxiv.org/pdf/2106.07179) uses the RIT fit. However, a more accurate fit, the NRSurrogate fit, is now available, although it is currently valid only for mass ratios up to 6. 
In recent paper (https://arxiv.org/pdf/2406.06390), we use the NRSurrogate fit for mass ratios (\leq 6) and the RIT fit for mass ratios (> 6).
The NRSurrogate fit is available through the surfinBH package (I recommend installing it via conda install -c conda-forge surfinbh), while the RIT fit is now implemented in the precession package (which can be installed using pip install precession).
For your convenience, I've shared a code snippet below for computing the recoil (kick) velocity. Please feel free to modify it as needed.
'''

import numpy as np
import precession
import surfinBH

#Vkick RIT
def Vkick_RIT(m1,m2,a1,a2,costilt1,costilt2,phi12):
    q = m2/m1
    Vrecoil = precession.remnantkick(np.arccos(costilt1),np.arccos(costilt2),phi12,q,a1,a2,kms=True,full_output=False)
    return Vrecoil[0]

#NRSURROGATE Fit
def cartesian_spin_components(a1,a2,tilt1,tilt2,phi12):
    spin_1x=a1 * np.sin(tilt1)
    spin_1y=0.
    spin_1z=a1 * np.cos(tilt1)
    spin_2x=a2 * np.sin(tilt2) * np.cos(phi12)
    spin_2y=a2 * np.sin(tilt2) * np.sin(phi12)
    spin_2z=a2 * np.cos(tilt2)
    return spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z

fit_name = 'NRSur7dq4Remnant'
fit = surfinBH.LoadFits(fit_name)

#NRSUR Vkick
def Vkick_NRSURfit(m1,m2,a1,a2,costilt1,costilt2,phi12):
    cc = 2.99792458 * (10**5)
    q = m1/m2
    spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = cartesian_spin_components(a1,a2,np.arccos(costilt1),np.arccos(costilt2),phi12)
    chiA = [spin_1x, spin_1y, spin_1z]
    chiB = [spin_2x, spin_2y, spin_2z]
    vf, vf_err = fit.vf(q, chiA, chiB)
    return cc * np.sqrt((vf[0])**2. + (vf[1])**2. + (vf[2])**2.)

#Combined Kick fit
def Vkick_NRfit(m1,m2,a1,a2,costilt1,costilt2,phi12):
    mass_ratio = m1/m2
    if mass_ratio <= 6:
        Vkick = Vkick_NRSURfit(m1,m2,a1,a2,costilt1,costilt2,phi12)
    else:
        Vkick = Vkick_RIT(m1,m2,a1,a2,costilt1,costilt2,phi12)
    return Vkick

