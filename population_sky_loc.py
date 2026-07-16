import numpy as np
from GWFish.modules.detection import Network, Detector
from GWFish.modules.horizon import horizon
from GWFish.modules.fishermatrix import compute_network_errors, sky_localization_percentile_factor
import GWFish.modules as gwf_mods
import pandas as pd
import pathlib
import os

import bilby as bb
import matplotlib
import matplotlib.pylab as plt

import json
import utils

base_dir  = os.getcwd()

regenerate = True

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
    prior['luminosity_distance'].maximum = 106192.4 # z = 10
    N_pop = 100
    _pop_samples = prior.sample(N_pop)
    _pop_samples['geocent_time'] = np.zeros(N_pop)
    _pop_samples['total_mass'] = bb.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(_pop_samples['chirp_mass'], _pop_samples['mass_ratio'])

    # Filter for only those that are detectable by NewAthena to slightly speed up the computation
    Flux = utils.X_ray_luminosity_flux(np.array(_pop_samples['total_mass']), np.array(_pop_samples['luminosity_distance']))
    detectable_map = Flux > utils.F_lim

    print("fraction detectable by NewAthena:", len(np.array(_pop_samples['total_mass'])[detectable_map])/len(np.array(_pop_samples['total_mass'])))

    pop_samples = {k: [x for x, m in zip(v, detectable_map) if m] for k, v in _pop_samples.items()}

    const_90 = sky_localization_percentile_factor(90)
    const_50 = sky_localization_percentile_factor(50)

    network = gwf_mods.detection.Network(detector_ids=detectors,
                                         detection_SNR=(0., 12.),
                                         config=pathlib.Path(base_dir + '/detectors.yaml'))

    total_mass = pop_samples.pop('total_mass')

    gwfish_input_data = pd.DataFrame.from_dict({k:v*np.array([1.]) for k, v in pop_samples.items()})

    results['detected_idxs'], results['netw_snrs'], results['errors'], results['sky_locs'] = compute_network_errors(
        network=network,
        parameter_values=gwfish_input_data,
        f_ref=10,
        waveform_model='IMRPhenomXPHM',
        save_matrices=True,
        save_matrices_path=pathlib.Path(os.path.join(base_dir,
                                                 'GWFish_analysis',
                                                 'BBH',
                                                 'Fisher_matrices')),
        matrix_naming_postfix="test"
    )

    results['sky_percentiles_90'] = results['sky_locs'] * const_90
    results['sky_percentiles_50'] = results['sky_locs'] * const_50

    pop_samples['total_mass'] = total_mass

    with open("data.json", "w") as f:
        for key in results:
            results[key] = results[key].tolist()
        data = {"pop_samples": pop_samples, "results": results}
        json.dump(data, f)

else:

    with open("data.json", "r") as f:
        data = json.load(f)

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
plt.savefig('Mt_vs_sky_loc.png')
plt.close()

