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

#biz removed this- dont think I want to choose kicks just caluclatethem based on masses.
# in further equatuions the kick is calculated based on the masses and the mass asymmetry. 
# def choose_kick_velocity_kms(
#     m1_msun,
#     m2_msun,
#     vk_user_kms=None,
#     use_mass_asymmetry_only=True
# ):
#     """
#     For first tests:
#     - if vk_user_kms is supplied, use that;
#     - otherwise use the mass-asymmetry kick only.

#     Later, replace this with a full spin-dependent recoil fit.
#     """
#     if vk_user_kms is not None:
#         return np.asarray(vk_user_kms)

#     q = mass_ratio_q(m1_msun, m2_msun)

#     if use_mass_asymmetry_only:
#         return kick_mass_asymmetry_kms(q)

#     raise NotImplementedError(
#         "Full spin-dependent kicks should be inserted here."
#     )

# AGN and BHL quantities 

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


def hill_radius_cm(M_rem_msun, M_smbh_msun, R_cm):
    """
    Hill radius:
        r_Hill = R * (M_rem / (3 M_SMBH))^(1/3)
        can be approximated to 0.6H where H is the scale height of the disk
    """
    return R_cm * (M_rem_msun / (3.0 * M_smbh_msun))**(1.0 / 3.0)


def cavity_radius_cm(
    M_rem_msun,
    M_smbh_msun,
    R_cm,
    H_cm,
    vk_kms,
    cs_kms,
    mode="min_bhl_hill"
):
    """
    Alternative prescriptions for the cavity radius.
    Options for how the cavity radius it set, if it is set, etc biztodo: is this overkill/ just set to 0.8H for now/ find a better prescription 

    mode options:
    - "none": no cavity, r_cav = 0
    - "min_bhl_hill": r_cav = min(r_BHL, r_Hill)
    - "0.6H": r_cav = 0.6 H
    - float/int: user-supplied radius in cm
    """
    if mode == "none":
        return 0.0

    if isinstance(mode, (float, int)):
        return float(mode)

    r_bhl = bhl_radius_cm(M_rem_msun, vk_kms, cs_kms)
    r_hill = hill_radius_cm(M_rem_msun, M_smbh_msun, R_cm)

    if mode == "min_bhl_hill":
        return np.minimum(r_bhl, r_hill)

    if mode == "0.6H":
        return 0.6 * H_cm

    raise ValueError(f"Unknown cavity mode: {mode}")

# Timescales

def t_kick_s(r_cav_cm, vk_kms):
    """
    Cavity crossing time:
        t_kick = r_cav / v_k
    """
    vk = vk_kms * KM_CGS
    return r_cav_cm / vk


def t_bhl_s(M_rem_msun, vk_kms, cs_kms, n_acc=3.0, use_v_eff=False):
    """
    Jet-formation/accretion timescale.

    Chen & Dai write approximately:
        t_BHL ~ r_BHL / v_k ~ G M / v_k^3

    n_acc accounts for 'a few' BHL/accretion times. - biz may need to change the n_acc value, if I can find a value here which would work better - compare results which come out and see what this needs to be 
    If use_v_eff=True, divide by sqrt(v_k^2 + c_s^2) instead of v_k. biztodo check where this effictive value is coming from (not going to use yet)
    """
    r_bhl = bhl_radius_cm(M_rem_msun, vk_kms, cs_kms)

    vk = vk_kms * KM_CGS
    cs = cs_kms * KM_CGS

    if use_v_eff:
        v_cross = np.sqrt(vk**2 + cs**2)
    else:
        v_cross = vk

    return n_acc * r_bhl / v_cross


def t_exit_disk_s(H_cm, vk_kms):
    """
    Approximate minimum time for kicked remnant to leave the AGN disk:
        t_exit ~ H / v_k

    Useful as a consistency check:
        jet formation inside the disk requires t_BHL < t_exit.

    biztodo: check if this is necessary? could leave for now
    """
    vk = vk_kms * KM_CGS
    return H_cm / vk


def total_delay_s(
    M_rem_msun,
    M_smbh_msun,
    R_cm,
    H_cm,
    rho_agn,
    vk_kms,
    cs_kms,
    cavity_mode="min_bhl_hill",
    n_acc=3.0,
    t_breakout_s=0.0,
    t_diff_s=0.0,
    use_v_eff_for_t_bhl=False
):
    """
    Total delay:
        t_delay = t_kick + n_acc*t_BHL + t_breakout + t_diff

    rho_agn is included because you will often call this function
    together with luminosity/detectability, but in this simple t_BHL
    prescription rho_agn does not enter the delay directly.
    """
    r_cav = cavity_radius_cm( #biztodo: check with the cavity radius bit, if this is overkill replace with prescription here
        M_rem_msun=M_rem_msun,
        M_smbh_msun=M_smbh_msun,
        R_cm=R_cm,
        H_cm=H_cm,
        vk_kms=vk_kms,
        cs_kms=cs_kms,
        mode=cavity_mode
    )

    tkick = t_kick_s(r_cav, vk_kms) #cavity crossing time, if there is a cavity. 
    tbhl = t_bhl_s( #jet formation/accretion timescale,
        M_rem_msun=M_rem_msun,
        vk_kms=vk_kms,
        cs_kms=cs_kms,
        n_acc=n_acc,
        use_v_eff=use_v_eff_for_t_bhl
    )

    return tkick + tbhl + t_breakout_s + t_diff_s


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
    cc = 2.99792458e5
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
        mf, mf_err = fit.mf(q, chiA, chiB) # remnant mass and 1-sigma error estimate - need to check what unit this is in biztodo
        return mf
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

    m1 = chirp_mass * (1 + q)**(1/5) / (q**(3/5))
    m2 = m1 * q

    m1 = np.maximum(m1, m2)  # Ensure m1 is the larger mass
    m2 = np.minimum(m1, m2)  # Ensure m2 is the smaller mass   - this should always be true based on how I have defined the equations 

    return m1, m2 #in solar masses 