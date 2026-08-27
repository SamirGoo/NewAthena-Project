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
        energy_keV * model,
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


def thermal_blackbody(
    energy_keV,
    luminosity,
    kT,
):
    # Prevent overflow in np.exp for high energies
    x = energy_keV / kT
    x = np.clip(x, 1e-5, 700)

    # 1. Photon spectrum shape: E^2 / (exp(E/kT) - 1)
    model = (energy_keV**2) / (np.exp(x) - 1)

    # 2. Energy spectrum shape for integration: E * photon_shape
    norm = np.trapz(
        energy_keV * model,
        energy_keV,
    )

    # 3. Scale by total luminosity (erg/s) converted from the keV integral
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
