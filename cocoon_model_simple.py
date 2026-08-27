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
        4
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


def cocoon_luminosity(
    mass,
    rho_agn,
    vk,
    cs,
    H,
    epsilon_x=0.03,
):

    return (
        epsilon_x
        *
        jet_luminosity(
            mass,
            rho_agn,
            vk,
            cs,
        )
    )


def cocoon_temperature_keV(
    mass,
    rho_agn,
    vk,
    cs,
    H
):

    Lj = jet_luminosity(
        mass,
        rho_agn,
        vk,
        cs,
    )

    return (
        1.0
        *
        (
            Lj /
            1e44
        )**0.15
    )


def cocoon_duration(
    mass,
    rho_agn,
    vk,
    cs,
    H
):

    Lj = jet_luminosity(
        mass,
        rho_agn,
        vk,
        cs,
    )

    return (
        1000.0
        *
        (
            Lj / 1e44
        )**(-0.2)
    )
