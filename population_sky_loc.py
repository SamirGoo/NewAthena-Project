import numpy as np
from GWFish.modules.detection import Network, Detector
from GWFish.modules.horizon import horizon
from GWFish.modules.fishermatrix import compute_network_errors, sky_localization_percentile_factor, compute_detector_fisher
import GWFish.modules as gwf_mods
import pandas as pd
import pathlib
import os

import bilby as bb
import matplotlib
import matplotlib.pylab as plt

import json

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
from significance import get_observation_significance


athena = AthenaWFI(
    "rsp/NewAthena_WFI_13rows_LDA_wo_filter_FoVAvg_20260511.rsp",
    "bkgd/NewAthena_WFI_13rows_LDA_20260528_bkgd_sum_9asec_wo_filter_FoVAvg.pha",
)

base_dir = "."
regenerate = True

N_pop = 100000

if regenerate:

    # One CE at LLO, one CE at Gingin (Australia), ET in Sardinia
    detectors = ['CE1', 'CE2', 'ET']

    results = {'detected_idxs':[],
           'netw_snrs':[],
           'errors':[],
           'sky_locs':[]}

    # Draw sets of parameters from BBHPriorDict
    prior = bb.gw.prior.BBHPriorDict()
    prior['mass_1'].minimum = 50
    prior['mass_1'].maximum = 10000
    prior['mass_2'].minimum = 50
    prior['mass_2'].maximum = 10000
    prior['chirp_mass'].minimum = 30
    prior['chirp_mass'].maximum = 10000
    prior['luminosity_distance'].minimum = 50
    prior['luminosity_distance'].maximum = 25924.2 # z = 3, 47647.9 for z = 5, 106192.4 for z = 10, probably need to decrease for more detectable sources
    prior['SMBH_mass'] = bb.prior.LogUniform(name='SMBH_mass', minimum=1e6, maximum=1e10)
    prior['fractional_r'] = bb.prior.LogUniform(name='fractional_r', minimum=1e-10, maximum=1)
    prior['geocent_time'] = bb.prior.Uniform(name='geocent_time', minimum=0, maximum=np.pi * 10**7)

    _pop_samples = prior.sample(N_pop)
    _pop_samples['total_mass'] = bb.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(_pop_samples['chirp_mass'], _pop_samples['mass_ratio'])
    _pop_samples['redshift'] = bb.gw.conversion.luminosity_distance_to_redshift(_pop_samples['luminosity_distance'])
    _pop_samples['mass_1'], _pop_samples['mass_2'] = bb.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(_pop_samples['chirp_mass'], _pop_samples['mass_ratio'])

    # Filter for only those that are detectable by NewAthena to slightly speed up the computation
    # this is where we add new NewAthena spectrum modeling
    sigma_thresh = 5
    significance = get_observation_significance(athena, _pop_samples, _pop_samples['SMBH_mass'], _pop_samples['fractional_r'])
    # get rid of nans for now, before we debug
    detectable_map = (significance > sigma_thresh) & (~np.isnan(significance))

    print("fraction of this population detectable by NewAthena:", len(np.array(_pop_samples['total_mass'])[detectable_map])/len(np.array(_pop_samples['total_mass'])))

    # We need to get rid of duplicate parameters
    # Removing spin parameters for now: 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl'
    wanted_params = ['mass_ratio', 'chirp_mass', 'luminosity_distance', 'theta_jn', 'psi', 'ra', 'dec', 'geocent_time']
    pop_samples = {k: [x for x, m in zip(v, detectable_map) if m] for k, v in _pop_samples.items() if k in wanted_params}

    total_mass = np.array(_pop_samples['total_mass'])[detectable_map]

    const_90 = sky_localization_percentile_factor(90)
    const_50 = sky_localization_percentile_factor(50)

    network = gwf_mods.detection.Network(detector_ids=detectors,
                                         detection_SNR=(0., 12.),
                                         config=pathlib.Path(base_dir + '/detectors.yaml'))


    gwfish_input_data = pd.DataFrame.from_dict({k:v*np.array([1.]) for k, v in pop_samples.items()})

    results['detected_idxs'], results['netw_snrs'], results['errors'], results['sky_locs'] = compute_network_errors(
        network=network,
        parameter_values=gwfish_input_data,
        f_ref=10,
        waveform_model='IMRPhenomXAS',
        save_matrices=False,
    )

    results['sky_percentiles_90'] = results['sky_locs'] * const_90
    results['sky_percentiles_50'] = results['sky_locs'] * const_50

    pop_samples['total_mass'] = total_mass.tolist()

    with open("data.json", "w") as f:
        for key in results:
            results[key] = results[key].tolist()
        data = {"pop_samples": pop_samples, "results": results}
        json.dump(data, f)

else:

    with open("data.json", "r") as f:
        data = json.load(f)

print(len(data['pop_samples']['total_mass'])/N_pop)
print(len(data['results']['sky_percentiles_90'])/N_pop)


detected_idxs_3G = data['results']['detected_idxs']
within_WFI_map = np.array(data['results']['sky_percentiles_90'])[detected_idxs_3G] < 0.7
within_10deg_map = np.array(data['results']['sky_percentiles_90'])[detected_idxs_3G] < 10


print("fraction of detectable with sky area fully in WFI:",
len(np.array(data['pop_samples']['total_mass'])[detected_idxs_3G][within_WFI_map])/len(np.array(data['pop_samples']['total_mass'])[detected_idxs_3G]))

print("fraction of detectable with sky area in 10 deg^2:",
len(np.array(data['pop_samples']['total_mass'])[detected_idxs_3G][within_10deg_map])/len(np.array(data['pop_samples']['total_mass'])[detected_idxs_3G]))

Fig = plt.figure()
plt.scatter(np.array(data['pop_samples']['total_mass'])[detected_idxs_3G], np.array(data['results']['sky_percentiles_90'])[detected_idxs_3G],
            label='detectable by 3G+NewAthena', alpha=0.5, c=np.array(data['results']['netw_snrs'])[detected_idxs_3G], cmap='viridis',
            norm=matplotlib.colors.LogNorm())
plt.colorbar(label='Network SNR')
plt.axhline(0.7, label='NewAthena WFI span (single tiling)', color='k', ls='--')
plt.axhline(10, label='10 deg$^2$', color='k', ls=':')
plt.legend(loc='upper left')
plt.xlabel('total mass [M$_\odot$]')
plt.ylabel('90% sky loc [deg$^2$]')
plt.xscale('log')
plt.yscale('log')
plt.savefig('Mt_vs_sky_loc.pdf')
plt.close()

