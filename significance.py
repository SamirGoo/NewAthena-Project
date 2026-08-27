import numpy as np

from athena_wfi import AthenaWFI

from agn_model import (
    agn_grid_interps,
    get_disk_properties
)

from cocoon_model_simple import (
    cocoon_luminosity,
    cocoon_temperature_keV,
    cocoon_duration,
)

from spectrum import (
    thermal_bremsstrahlung,
    thermal_blackbody,
    observed_flux,
)

from absorption import transmission

from kicks import Vkick_NRfit
from utils import Mrem_NRSURfit

import matplotlib.pyplot as plt


def get_remnant_mass_and_kicks(source_properties):

    remnant_kicks = []
    remnant_mass = []


    for i, m1 in enumerate(source_properties['mass_1']):
        remnant_kicks.append(
                             Vkick_NRfit(source_properties['mass_1'][i],
                             source_properties['mass_2'][i],
                             source_properties['a_1'][i],
                             source_properties['a_2'][i],
                             np.cos(source_properties['tilt_1'][i]),
                             np.cos(source_properties['tilt_2'][i]),
                             source_properties['phi_12'][i])
                             )
        remnant_mass.append(
                             Mrem_NRSURfit(source_properties['mass_1'][i],
                             source_properties['mass_2'][i],
                             source_properties['a_1'][i],
                             source_properties['a_2'][i],
                             np.cos(source_properties['tilt_1'][i]),
                             np.cos(source_properties['tilt_2'][i]),
                             source_properties['phi_12'][i])
                             )

    return np.array(remnant_mass), np.array(remnant_kicks)


def get_observation_significance(detector_model, source_properties, M_SMBH, fractional_rbbh):

    Eobs = detector_model.energy
    Erest = np.array([Eobs * (1 + z) for z in source_properties['redshift']])

    # Remnant mass in solar masses
    remnant_mass, vk = get_remnant_mass_and_kicks(source_properties)
    print(remnant_mass)
    vk *= 1e5 # kick units are km/s, converted to cm/s

    # Set up agn model
    smbh_grid = np.logspace(6, 10, 10)
    dimless_rmin, dimless_rmax, log_rho_interp, log_cs_interp = agn_grid_interps(smbh_grid)
    dimless_rbbh = dimless_rmin + (fractional_rbbh * (dimless_rmax - dimless_rmin))

    rho_agn, cs, H = get_disk_properties(log_rho_interp, log_cs_interp, M_SMBH, dimless_rbbh)
    # convert to cgs
    rho_agn *= 1e-3
    cs *= 1e2
    H *= 1e2

    Lx = cocoon_luminosity(
                remnant_mass,
                rho_agn,
                vk,
                cs,
                H
            )

    kT = cocoon_temperature_keV(
                remnant_mass,
                rho_agn,
                vk,
                cs,
                H
            )

    duration = cocoon_duration(
                remnant_mass,
                rho_agn,
                vk,
                cs,
                H
            )

    source_spec = np.array([thermal_bremsstrahlung(Eresti, Lxi, kTi) * transmission(Eresti) for Eresti, Lxi, kTi in zip(Erest, Lx, kT)])

    exposure = np.array([min(10000., d) for d in duration])

    bkg_counts = np.array([detector_model.background_counts(e) for e in exposure])

    B = np.array([b.sum() for b in bkg_counts])

    flux = np.array([observed_flux(Eobs, spec, z) for spec, z in zip(source_spec, source_properties['redshift'])])

    src_counts = np.array([detector_model.fold(f, e) for f, e in zip(flux, exposure)])

    S = np.array([s.sum() for s in src_counts])

    significance = (
                    S /
                    np.sqrt(
                        S + B
                    )
    )
    fig = plt.figure()
    plt.scatter(vk, significance)
    plt.xlabel('vk (cm/s)')
    plt.ylabel('significance')
    plt.savefig('vk_vs_sig.png')
    plt.close()
    return np.array(significance)

