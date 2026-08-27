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


    t_breakout = breakout_time(mass, rho_agn, vk, cs, H, f_bz)
    Lj = jet_luminosity(mass, rho_agn, vk, cs, f_bz)
    beta_h = get_beta_h(H, t_breakout)

    return Lj * t_breakout * (1 - beta_h)


def cocoon_mass(f_FB, Ec, beta_cj=0.7):
    return f_FB * Ec / (beta_cj**2 * c**2)


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
    V_cj = np.pi * (0.05 * H)**2 * H # volume of the cocoon - radius assumed to be 5% of disk height
    m_cj = rho_agn * V_cj # cocoon_mass(f_FB, Ec) # rho_agn * V_cj # mass of the cocoon

    return 2 * np.pi * c * f_FB * Ec * V_cj**(1/3) / (kappa * m_cj)



def cocoon_mass(f_FB, Ec, beta_cj=0.7):
    return f_FB * Ec / (beta_cj**2 * c**2)


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
    t_b = breakout_time(mass, rho_agn, vk, cs, H, f_bz)

    kappa = 0.34 # cm^2 g^-1
    f_FB = 0.1 # Nakar & Piran 2017
    r_cj = 0.05 * H
    V_cj = np.pi * r_cj**2 * H # volume of the cocoon - radius assumed to be 5% of disk height
    m_cj = rho_agn * V_cj # mass of the cocoon
    #print("m_cj 1:", m_cj)
    #m_cj = cocoon_mass(f_FB, Ec)
    #print("m_cj 2:", m_cj)
    a = 7.5657 * 1e-15 # radiation density constant, erg cm^-3 K^-4
    kB = 1.380649e-16 # erg/K
    kB_eV = 8.617333262e-5 # eV/K

    TBB_cj = np.power(f_FB * Ec / (4 * a * V_cj), 1/4) # units of K

    n_BB = a * TBB_cj**3 / 3 / kB # units of?
    ndot_ph = 3.5e36 * rho_agn**2 * TBB_cj**(-0.5) # units of?
    eta_cj = n_BB / t_b / ndot_ph

    TBB_cj_eV = TBB_cj * kB_eV

    Tcj_b_keV = TBB_cj_eV * 1e-3 # units of keV

    try:
        if eta_cj >= 1:
            ymax = 3.0 * (rho_agn / 10**(-9))**(-0.5) * np.power(TBB_cj_eV / 100, 9/4)
            compton_corrected = max(1.0, 0.5 * np.log(ymax) * (1.6 + np.log(ymax)))
            Tcj_comp = TBB_cj * eta_cj**2 / compton_corrected**2
            Tcj_comp_keV = Tcj_comp * kB_eV * 1e-3
            Tcj_b_keV = min(Tcj_comp_keV, 100)
    except:
        map = (eta_cj >= 1)
        ymax = 3.0 * (rho_agn / 10**(-9))**(-0.5) * np.power(TBB_cj_eV / 100, 9/4)
        compton_corrected = np.array([max(1.0, 0.5 * np.log(ym) * (1.6 + np.log(ym))) for ym in ymax])
        Tcj_comp = TBB_cj * eta_cj**2 / compton_corrected**2
        Tcj_comp_keV = Tcj_comp * kB_eV * 1e-3
        Tcj_b_keV = np.array([min(Tcjckev, 100) if ecj >= 1 else Tbbcjkev * 1e-3 for ecj, Tcjckev, Tbbcjkev in zip(eta_cj, Tcj_comp_keV, TBB_cj_eV)])

    # How much has cocoon expanded by in time t_b?
    v_cj = np.sqrt(f_FB * Ec / m_cj)
    r_diff = np.sqrt(kappa * m_cj * v_cj / (4 * np.pi * c))

    Tcj = Tcj_b_keV * V_cj ** (1/3) / r_diff

    return Tcj


# chen and dai calculations eq 30
def cocoon_duration(
    mass,
    rho_agn,
    vk,
    cs,
    H,
    f_bz=0.1
):

    Ec = cocoon_energy(mass, rho_agn, vk, cs, H, f_bz)
    f_FB = 0.1 # Nakar & Piran 2017
    kappa = 0.34 # cm^2 g^-1
    V_cj = np.pi * (0.05 * H)**2 * H # volume of the cocoon - radius assumed to be 5% of disk height
    m_cj = rho_agn * V_cj # cocoon_mass(f_FB, Ec) # rho_agn * V_cj # mass of the cocoon
    beta_cj = np.sqrt(f_FB * Ec / (m_cj * c**2)) # rearranged eq 29 for this

    return np.sqrt(kappa * m_cj / (4 * np.pi * beta_cj * c**2))
