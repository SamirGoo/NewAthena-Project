import numpy as np
from scipy.constants import c
from astropy.cosmology import Planck18
import astropy.constants as const


c = const.c.cgs.value # speed of light in cm per s
G = const.G.cgs.value # gravitational constant in cm^3 per gram per s^2
MSUN = const.M_sun.cgs.value  # convert solar masses to grams


def mdot_bhl(
    mass, # In units of solar masses
    rho_agn, # in grams per cm cubed
    vk, # in cm / s
    cs, # in cm / s
):

    m = mass * MSUN # mass in grams

    return (
        4.0
        * np.pi
        * G**2
        * m**2
        * rho_agn
        /
        (
            vk**2 +
            cs**2
        )**1.5
    )


def jet_luminosity(
    mass,
    rho_agn,
    vk,
    cs,
    f_bz=0.1,
):
    return (
        f_bz *
        mdot_bhl(
            mass,
            rho_agn,
            vk,
            cs,
        )
        *
        c**2
    )


# from chen and dai
def get_beta_h(H, t_breakout):
    return 3 * H / (5 * t_breakout * c)


# from chen and dai
def breakout_time(mass,
    rho_agn,
    vk,
    cs,
    H,
    f_bz=0.1,
    theta_0=0.17
):

    Mdot = mdot_bhl(mass, rho_agn, vk, cs)
    Lj = jet_luminosity(mass, rho_agn, vk, cs, f_bz)

    return (3/5) * H**(5/3) * (rho_agn * theta_0 / Lj)**(1/3)


# from chen and dai
def cocoon_energy(mass,
                  rho_agn,
                  vk,
                  cs,
                  H,
                  f_bz=0.1):


    t_bre = breakout_time(mass, rho_agn, vk, cs, H, f_bz)
    Lj = jet_luminosity(mass, rho_agn, vk, cs, f_bz)
    beta_h = get_beta_h(H, t_bre)

    return Lj * t_bre * (1 - beta_h)


def cocoon_mass(f_FB, Ec, beta_cj=0.7):
    return f_FB * Ec / (beta_cj**2 * c**2)


def cocoon_volume(H, t_bre, beta_c=0.7):
    return np.pi * get_beta_h(H, t_bre) * beta_c**2 * c**3 * t_bre**3


# from eq 29 of chen and dai
def cocoon_luminosity(
    mass,
    rho_agn,
    vk,
    cs,
    H,
    f_bz=0.1,
):

    Ec = cocoon_energy(mass, rho_agn, vk, cs, H, f_bz)
    f_FB = 0.1 # Nakar & Piran 2017
    kappa = 0.34 # cm^2 g^-1
    t_bre = breakout_time(mass, rho_agn, vk, cs, H, f_bz)
    V_cj = cocoon_volume(H, t_bre)
    m_cj = cocoon_mass(f_FB, Ec) # rho_agn * V_cj # mass of the cocoon

    return 2 * np.pi * c * f_FB * Ec * V_cj**(1/3) / (kappa * m_cj)


# chen and dai calculations eq 15
def cocoon_duration(
    mass,
    rho_agn,
    vk,
    cs,
    H,
    f_bz=0.1
):
    kappa = 0.34 # cm^2 g^-1
    t_bre = breakout_time(mass, rho_agn, vk, cs, H, f_bz)
    beta_h = get_beta_h((H, t_bre)

    return 1 / (kappa * rho_agn * beta_h**2 * c)


# chen and dai calculation
def cocoon_temperature_keV(
    mass,
    rho_agn,
    vk,
    cs,
    H,
    f_bz=0.1,
):

    Ec = cocoon_energy(mass, rho_agn, vk, cs, H, f_bz)
    t_bre = breakout_time(mass, rho_agn, vk, cs, H, f_bz)
    t_duration = cocoon_duratione(mass, rho_agn, vk, cs, H, f_bz)

    kappa = 0.34 # cm^2 g^-1
    f_FB = 0.1 # Nakar & Piran 2017

    V_cj = cocoon_volume(H, t_bre)
    m_cj = cocoon_mass(f_FB, Ec) # rho_agn * V_cj # mass of the cocoon

    beta_h = get_beta_h(H, t_bre)

    a = 7.5657 * 1e-15 # radiation density constant, erg cm^-3 K^-4
    kB = 1.380649e-16 # erg/K
    kB_eV = 8.617333262e-5 # eV/K

    TBB_hbre = np.power(18/(7*a) * rho_agn * beta_h**2 * c**2, 1/4) # units of K

    n_BB = a * TBB_hbre**3 / 3 / kB # units of?
    ndot_ph = 3.5e36 * rho_agn**2 * TBB_hbre**(-0.5) # units of?

    eta = n_BB / t_duration / ndot_ph

    TBB_hbre_eV = TBB_hbre * kB_eV

    TBB_hbre_keV = TBB_hbre_eV * 1e-3 # units of keV

    try:
        if eta >= 1:
            ymax = 3.0 * (rho_agn / 10**(-9))**(-0.5) * np.power(TBB_hbre_eV / 100, 9/4)
            compton_corrected = max(1.0, 0.5 * np.log(ymax) * (1.6 + np.log(ymax)))
            TBB_comp = TBB_hbre * eta**2 / compton_corrected**2
            TBB_comp_keV = TBB_comp * kB_eV * 1e-3
            TBB_hbre_keV = min(TBB_comp_keV, 100)
    except:
        map = (eta >= 1)
        ymax = 3.0 * (rho_agn / 10**(-9))**(-0.5) * np.power(TBB_hbre_eV / 100, 9/4)
        compton_corrected = np.array([max(1.0, 0.5 * np.log(ym) * (1.6 + np.log(ym))) for ym in ymax])
        TBB_comp = TBB_hbre * eta**2 / compton_corrected**2
        TBB_comp_keV = TBB_comp * kB_eV * 1e-3
        TBB_hbre_keV = np.array([min(T1kev, 100) if e >= 1 else T2kev for e, T1kev, T2kev in zip(eta, TBB_comp_keV, TBB_hbre_keV)])

    return TBB_hbre_keV


