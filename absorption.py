import numpy as np


def photoelectric_sigma(
    energy_keV,
):

    return (
        2e-22
        *
        energy_keV**(-2.6)
    )


def transmission(
    energy_keV,
    nh_mw=3e20,
    nh_agn=1e23,
    covering=0.8,
):

    sigma = photoelectric_sigma(
        energy_keV
    )

    tau_mw = nh_mw * sigma

    tau_agn = nh_agn * sigma

    mw = np.exp(-tau_mw)

    agn = (
        covering
        *
        np.exp(-tau_agn)
        +
        (1 - covering)
    )

    return mw * agn
