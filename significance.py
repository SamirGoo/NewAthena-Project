import numpy as np

from athena_wfi import AthenaWFI

from cocoon_model import (
    cocoon_luminosity,
    cocoon_temperature_keV,
    cocoon_duration,
)

from spectrum import (
    thermal_bremsstrahlung,
    observed_flux,
)

from absorption import transmission

from kicks import Vkick_NRfit


def get_kicks(source_properties):

    remnant_kicks = []
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
    return np.array(remnant_kicks)


def get_observation_significance(detector_model, source_properties):

    Eobs = detector_model.energy
    Erest = np.array([Eobs * (1 + z) for z in source_properties['redshift']])

    # Remnant mass in solar masses
    remnant_mass = (source_properties['mass_1'] + source_properties['mass_2']) * 0.95 # IRS todo make this computation more sophisticated
    vk = get_kicks(source_properties) * 1e5 # kick units are km/s, converted to cm/s
    cs = 1e7 # sound speed - IRS this is constant for all AGN, is this reasonable or should it vary with density?

    Lx = cocoon_luminosity(
                remnant_mass,
                source_properties['rho_agn'],
                vk,
                cs,
            )

    kT = cocoon_temperature_keV(
                remnant_mass,
                source_properties['rho_agn'],
                vk,
                cs,
            )

    duration = cocoon_duration(
                remnant_mass,
                source_properties['rho_agn'],
                vk,
                cs,
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
    return np.array(significance)

