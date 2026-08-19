import numpy as np
from scipy.constants import c
from astropy.cosmology import Planck18

c = c * 100.0 # speed of light in cm per s

G = 6.6743e-8 # gravitational constant in cm^3 per gram per s^2
MSUN = 1.98847e33 # convert solar masses to grams


def mdot_bhl(
    mass, # In units of solar masses
    rho_agn, # in grams per cm cubed
    vk, # in cm / s
    cs, # in cm / s
):

    m = mass * MSUN

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


def jet_power(
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
    epsilon_x=0.03,
):

    return (
        epsilon_x
        *
        jet_power(
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
):

    Lj = jet_power(
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
):

    Lj = jet_power(
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
