import numpy as np
from astropy.cosmology import Planck18

KEV_TO_ERG = 1.60218e-9


def thermal_bremsstrahlung(
    energy_keV,
    luminosity,
    kT,
):

    model = (
        energy_keV**-1
        *
        np.exp(
            -energy_keV / kT
        )
    )

    norm = np.trapz(
        energy_keV *
        model,
        energy_keV,
    )

    model *= (
        luminosity
        /
        (
            norm *
            KEV_TO_ERG
        )
    )

    return model


def observed_flux(
    energy_obs,
    source_model,
    z,
):

    dl = (
        Planck18
        .luminosity_distance(z)
        .cgs
        .value
    )

    return (
        source_model
        /
        (
            4*np.pi*dl**2
            *
            (1+z)
        )
    )
