import numpy as np
import astropy
import astropy.constants as const

# Constants, cgs

G_CGS = const.G.cgs.value          # cm^3 g^-1 s^-2
C_CGS = const.c.cgs.value          # cm s^-1
MSUN_CGS = const.M_sun.cgs.value   # g
DAY_CGS = 86400.0           # s
KM_CGS = 1.0e5              # cm


# Binary-remnant quantities

def mass_ratio_q(m1_msun, m2_msun):
    """
    Return q = lighter/heavier, so q <= 1.
    m1 is heavier, m2 is lighter
    """
    m_heavy = max(m1_msun, m2_msun)
    m_light = min(m1_msun, m2_msun)
    return m_light / m_heavy


def remnant_mass_simple(m1_msun, m2_msun, eps_rad=0.05):
    """
    Very simple remnant-mass estimate.
    Replace with outcomes from GWFISH ** done in notebook
    """
    return (1.0 - eps_rad) * (m1_msun + m2_msun)


def kick_mass_asymmetry_kms(q):
    """
    Mass-asymmetry recoil component from Chen & Dai's quoted formula.

    q should be lighter/heavier, q <= 1.
    Output is km/s.
    """
    A = 1.2e4      # km/s these values quoted in the paper. 
    B = -0.93

    A_term = A * q**2 * (1.0 - q) / (1.0 + q)**5
    B_term = 1.0 + B * q / (1.0 + q)**2 

    return (A_term * B_term)

def total_delay_days(*args, **kwargs):
    return total_delay_s(*args, **kwargs) / DAY_CGS #converting time into days (for comparison)

"""
Code from partapratim below: 
- including the spin kick into the kick velocity 
- including a remnant fit from lalsimulation 
"""

'''
This paper (https://arxiv.org/pdf/2106.07179) uses the RIT fit. However, a more accurate fit, the NRSurrogate fit, is now available, although it is currently valid only for mass ratios up to 6. 
In recent paper (https://arxiv.org/pdf/2406.06390), we use the NRSurrogate fit for mass ratios (\leq 6) and the RIT fit for mass ratios (> 6).
The NRSurrogate fit is available through the surfinBH package (I recommend installing it via conda install -c conda-forge surfinbh), while the RIT fit is now implemented in the precession package (which can be installed using pip install precession).
For your convenience, I've shared a code snippet below for computing the recoil (kick) velocity. Please feel free to modify it as needed.
'''

import numpy as np
import precession
import surfinBH

#Vkick RIT
def Vkick_RIT(m1,m2,chi1,chi2,costilt1,costilt2,phi12):
    q = m2/m1 #mass ratio using here is heavier/lighter, so q <= 1. may need to put in assurances that this will be the case? 
    Vrecoil = precession.remnantkick(np.arccos(costilt1),np.arccos(costilt2),phi12,q,chi1,chi2,kms=True,full_output=False)
    return Vrecoil #this is in km/s

#NRSURROGATE Fit
def cartesian_spin_components(chi1,chi2,tilt1,tilt2,phi12):
    spin_1x=chi1*np.sin(tilt1)
    spin_1y=0.
    spin_1z=chi1*np.cos(tilt1)
    spin_2x=chi2*np.sin(tilt2)*np.cos(phi12)
    spin_2y=chi2*np.sin(tilt2)*np.sin(phi12)
    spin_2z=chi2*np.cos(tilt2)
    return spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z

fit_name = 'NRSur7dq4Remnant'
fit = surfinBH.LoadFits(fit_name)

#NRSUR Vkick
def Vkick_NRSURfit(m1,m2,chi1,chi2,costilt1,costilt2,phi12):
    cc = 2.99792458e5 #kms
    q = m1/m2 
    spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = cartesian_spin_components(chi1,chi2,np.arccos(costilt1),np.arccos(costilt2),phi12)
    chiA = [spin_1x, spin_1y, spin_1z]
    chiB = [spin_2x, spin_2y, spin_2z]
    vf, vf_err = fit.vf(q, chiA, chiB) # remnant recoil kick and 1-sigma error estimate (units of c) this bit is in units of c. 
    return cc*np.sqrt((vf[0])**2.0 + (vf[1])**2.0 + (vf[2])**2.0) #now been converted into kms units - *c

#NRSUR Remnant Mass
def Mrem_NRSURfit(m1,m2,chi1,chi2,costilt1,costilt2,phi12):
    q = m1/m2
    if q <= 6:
        spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = cartesian_spin_components(chi1,chi2,np.arccos(costilt1),np.arccos(costilt2),phi12)
        chiA = [spin_1x, spin_1y, spin_1z]
        chiB = [spin_2x, spin_2y, spin_2z]
        mf, mf_err = fit.mf(q, chiA, chiB) # remnant mass and 1-sigma error estimate
        return mf * (m1 + m2) #as mf here is a fraction. check that this is not too long 
    else:
        mf = remnant_mass_simple(m1,m2,eps_rad=0.05) #revert back to simple model for mass ratio not trained on. 
        return mf 

#Combined Kick fit
def Vkick_NRfit(m1,m2,chi1,chi2,costilt1,costilt2,phi12):
    mass_ratio = m1/m2
    if mass_ratio <= 6:
        Vkick = Vkick_NRSURfit(m1,m2,chi1,chi2,costilt1,costilt2,phi12)
    else:
        Vkick = Vkick_RIT(m1,m2,chi1,chi2,costilt1,costilt2,phi12)
    return Vkick 
 

#m1 and m2 calculations 
def m1_m2_from_chirp_q(chirp_mass, q):
    """
    Calculate m1 and m2 from chirp mass and mass ratio - for calculations of remnant mass
    biztodo if this works put into python file for use in other scripts
    """

    m1_raw = chirp_mass * (1 + q)**(1/5) / (q**(3/5))
    m2_raw = m1_raw * q

    m1 = np.maximum(m1_raw, m2_raw)  # Ensure m1 is the larger mass
    m2 = np.minimum(m1_raw, m2_raw)  # Ensure m2 is the smaller mass   - this should always be true based on how I have defined the equations 

    return m1, m2 #in solar masses 

#############################################################
#luminosity calculations - to include these in the plotting #
#############################################################
    #from chen and dai

def jet_luminosity(Mdot, eta_jet):
    """
    Calculate the jet luminosity based on the mass accretion rate and efficiency.
    in cgs units 
    so luminosity is returned in erg/s, Mdot needs to be in cgs etc
    """
    c = const.c.cgs.value  # Speed of light in cm/s (ie cgs units) 
    Mdot_cgs = Mdot  # Mdot is already in cgs units (g/s) from bhl_rate_g_per_s function 
    L_jet = eta_jet * Mdot_cgs * c**2 
    return L_jet

def cocoon_energy(Mdot, eta_jet, t_breakout, H):
    """
    Calculating cocoon energy from the jet
    want everything in cgs again
    estimating v_jh/c is 0.1 so jet velocity is currently not needed 
    t_breakout is something calculated earlier - needs jet head velocity. 
    Mdot in cgs units from bhl_rate_g_per_s function, eta_jet is 0.1 for now
    """
    c = const.c.cgs.value
    beta_h = 3 * H / (5 * t_breakout * c) #new version of beta
    c = const.c.cgs.value
    L_jet = jet_luminosity(Mdot, eta_jet)
    E_cocoon = L_jet * t_breakout * (1 - (beta_h * t_breakout)) 
    return E_cocoon



def cocoon_luminosity_chen(Mdot, rho_agn, H, eta_jet, t_breakout):
    """
    Calculating the jet luminosity based on eq 29 of chen and dai 

    f_FB = 0.1 following Nakar & Piran (2017)
    kappa = 0.34 cm^2/g electron scattering opacity 
    volume of cocoon V_cj = pi * (0.1*H)**2 * H 
    E_c is calculated above - so I need to estimate breakout time and jet velocity to get this 
    m_cj = rho_agn * V_cj

    H is in cm 
    rho agn in g/cm^3 ?? - need to check 
    """  
    c = const.c.cgs.value
    E_cocoon = cocoon_energy(Mdot, eta_jet=0.1, t_breakout=t_breakout, H=H)
    f_FB = 0.1
    kappa = 0.34  # cm^2/g
    H = H  # cm, height of the AGN disk
    V_cj = np.pi * (0.05 * H)**2 * H #volume of the cocoon, assuming a cylindrical shape with radius 0.05H and height H - could change this if needed
    rho_agn = rho_agn  # g/cm^3, density of the AGN disk
    m_cj = rho_agn * V_cj 

    L_cocoon = 2 * np.pi * c * f_FB * E_cocoon * (V_cj)**(1/3) / (kappa * m_cj)

    return L_cocoon


# simple 

import numpy as np
from scipy.constants import c
from astropy.cosmology import Planck18

c = c * 100.0

G = 6.6743e-8
MSUN = 1.98847e33


def mdot_bhl(mass,rho_agn,vk,cs,):

    m = mass * MSUN #converting from Msun to grams 

    return (4* np.pi* G**2 * m**2 * rho_agn /(vk**2 +cs**2)**1.5) 


def jet_power(mass,rho_agn,vk,cs,f_bz=0.1,):

    return (f_bz * mdot_bhl(mass, rho_agn, vk, cs) * c**2)


def cocoon_luminosity(mass,rho_agn,vk,cs,epsilon_x=0.03,):

    return (epsilon_x*jet_power(mass,rho_agn,vk,cs,))


def cocoon_temperature_keV(mass, rho_agn, vk, cs):

    Lj = jet_power(mass, rho_agn, vk, cs)

    return (1.0 * ( Lj / 1e44 )**0.15)


def cocoon_duration(mass, rho_agn, vk, cs):

    Lj = jet_power(mass, rho_agn, vk, cs)

    return (1000.0 * ( Lj / 1e44)**(-0.2))


####################### back to time delay calculations     

def bhl_radius_cm(M_total_msun, vk_kms, cs_kms):
    """
    Bondi-Hoyle-Lyttleton radius:
        r_BHL = G M / (v_k^2 + c_s^2)
        M is total mass of BBH 
        density and sound speed of the AGN disk
        v rel is relative velocty between the BBh center of mass and the ambient gas - can be considered as the kick velocityu under the models/parameters tested in paper

    """
    M = M_total_msun * MSUN_CGS
    vk = vk_kms * KM_CGS
    cs = cs_kms * KM_CGS
    v_eff_sq = vk**2 + cs**2

    return G_CGS * M / v_eff_sq


def bhl_rate_g_per_s(M_rem_msun, rho_agn, vk_kms, cs_kms):
    """
    Bondi-Hoyle-Lyttleton accretion rate:
        Mdot_BHL = 4 pi G^2 M^2 rho / (v_k^2 + c_s^2)^(3/2)

    rho_agn in g cm^-3.
    input M_rem_msun needs to be in solar masses 
    """
    M = M_rem_msun * MSUN_CGS
    vk = vk_kms * KM_CGS
    cs = cs_kms * KM_CGS
    v_eff_sq = vk**2 + cs**2

    return 4.0 * np.pi * G_CGS**2 * M**2 * rho_agn / v_eff_sq**(3/2)

#cavity radius - estimating to be 0.6H 

def cavity_radius_cm_06H(H_cm):
    """
    Cavity radius:
        r_cav = 0.8 H
        H is the scale height of the disk
    """
    return 0.6 * H_cm

# Timescales

def t_kick_s(r_cav_cm, vk_kms):
    """
    Cavity crossing time:
        t_kick = r_cav / v_k
    """
    vk = vk_kms * KM_CGS
    return r_cav_cm / vk


def t_bhl_s(M_rem_msun, vk_kms, cs_kms, n_acc=1.0, use_v_eff=True):
    """
    Jet-formation/accretion timescale.

    Chen & Dai write approximately:
        t_BHL ~ r_BHL / v_k ~ G M / v_k^3

    n_acc accounts for 'a few' BHL/accretion times. - biz may need to change the n_acc value, if I can find a value here which would work better - compare results which come out and see what this needs to be 
    If use_v_eff=True, divide by sqrt(v_k^2 + c_s^2) instead of v_k. biztodo check where this effictive value is coming from (not going to use yet)
    """
    r_bhl = bhl_radius_cm(M_rem_msun, vk_kms, cs_kms) #needs total mass not remnant mass  - check what units it wants 

    vk = vk_kms * KM_CGS
    cs = cs_kms * KM_CGS

    if use_v_eff:
        v_cross = np.sqrt(vk**2 + cs**2)
    else:
        v_cross = vk

    return n_acc * r_bhl / v_cross



def t_breakout_s(H, rho_agn, M_rem_msun, eta_jet, vk_kms, cs_kms, theta_0=0.17):
    """
    Jet breakout timescale, from Chen & Dai eq 14, and the assumptions in Bromberg et all (2011)
        breakout occurs when the jet head reaches the scale height of the disk, H
        Mdot in solar masses, eta jet is 0.1 for now
    """
    Mdot = bhl_rate_g_per_s(M_rem_msun, rho_agn, vk_kms, cs_kms)
    luminosity_jet = jet_luminosity(Mdot, eta_jet) 
    t_bre = (3/5) * H**(5/3) * (rho_agn * theta_0 / luminosity_jet)**(1/3) 

    return t_bre


def total_delay_s(
    M_rem_msun,
    M_smbh_msun,
    H_cm,
    rho_agn,
    vk_kms,
    cs_kms,
    eta_jet=0.1,
    theta_0=0.17,
    n_acc=1.0,
    use_v_eff_for_t_bhl=True
):
    """
    Total delay:
        t_delay = t_kick + n_acc*t_BHL + t_breakout + t_diff

    rho_agn is included because you will often call this function
    together with luminosity/detectability, but in this simple t_BHL
    prescription rho_agn does not enter the delay directly.
    """
    r_cav = cavity_radius_cm_06H( #biztodo: check with the cavity radius bit, if this is overkill replace with prescription here
        H_cm=H_cm
    )

    tkick = t_kick_s(r_cav, vk_kms) #cavity crossing time, if there is a cavity. 
    tbhl = t_bhl_s( #jet formation/accretion timescale,
        M_rem_msun=M_rem_msun,
        vk_kms=vk_kms,
        cs_kms=cs_kms,
        n_acc=n_acc,
        use_v_eff=use_v_eff_for_t_bhl
    )

    
    c = 2.99792458e10 # speed of light in cm/s
    kappa = 0.34 # opacity in cm^2/g, electron scattering opacity 

    t_breakout = t_breakout_s(H=H_cm, rho_agn=rho_agn, M_rem_msun=M_rem_msun, vk_kms=vk_kms, cs_kms=cs_kms, eta_jet=0.1, theta_0=0.17) #jet breakout timescale, from Chen & Dai eq 14, and the assumptions in Bromberg et all (2011)

    beta_h = 3 * H_cm / (5 * t_breakout * c) #jet head velocity, from Chen and Dai eq 14, and the assumptions in Bromberg et all (2011)

    t_diff_s = 1 / (kappa * rho_agn * beta_h**2 * c) # eq 15 in Chen and Dai 

    return tkick + tbhl + t_diff_s + t_breakout


