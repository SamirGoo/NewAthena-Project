import bilby as bb
import numpy as np
from GWFish.modules.detection import Network, Detector
from GWFish.modules.horizon import horizon
from GWFish.modules.fishermatrix import compute_network_errors, compute_detector_fisher
import GWFish.modules as gwf_mods
import pandas as pd
import pathlib
import os

from astropy.cosmology import Planck18
from astropy.cosmology import z_at_value
import astropy.units as u

import utils

import matplotlib.pyplot as plt


base_dir = os.getcwd()

# Detectors are one CE at LLO location, one CE at Gingin (Australia), ET in Sardinia
detectors = ['CE1', 'CE2', 'ET']
ls = [':', '-.', '--']

#new max horizon: dL=608465.4969298993, z=48.99134890039851 for ra=4.310664640963899, dec=-0.7176488681791958

params = {
    'total_mass': 900,
    'mass_ratio': 0.9,
    'theta_jn': 0.0,
    'phase': 2.8,
    'geocent_time': 0.00475200053340101,
    'ra': 4.310664640963899,
    'dec': -0.7176488681791958,
    'psi': 0.2,}


network = gwf_mods.detection.Network(detector_ids=detectors,
                                     detection_SNR=(0., 12.),
                                     config=pathlib.Path(base_dir + '/detectors.yaml'))
dets = {}
for det_name in detectors:
    dets[det_name] = Detector(det_name,
                              config=pathlib.Path(base_dir + '/detectors.yaml'))

total_mass_range = np.logspace(1.5, 4.4, 200)
hzn_store = {'nwk': [], 'CE1': [], 'CE2': [], 'ET': [], "NA": []}

for tmass in total_mass_range:
    params['total_mass'] = tmass
    params['mass_1'], params['mass_2'] = bb.gw.conversion.total_mass_and_mass_ratio_to_component_masses(
                                             mass_ratio=params['mass_ratio'], total_mass=params['total_mass'])
    hzn, rz = horizon(params, network, waveform_model='IMRPhenomXPHM', target_SNR=10.0)
    print("tmass: {}, m1: {}, m2: {},  horizon dL: {}, z: {}".format(tmass, params['mass_1'], params['mass_2'], hzn, rz))
    hzn_store['nwk'].append(rz)
    for det_name in detectors:
        try:
            hzni, rzi = horizon(params, dets[det_name], waveform_model='IMRPhenomXPHM', target_SNR=10.0)
            hzn_store[det_name].append(rzi)
        except:
            print(det_name, "dropped")
    NA_dist = utils.max_distance_NewAthena(tmass, fj=0.7**2)
    rz_na = z_at_value(Planck18.luminosity_distance, NA_dist * u.Mpc)
    hzn_store["NA"].append(rz_na)
    print("NA redshift: {}".format(rz_na))

fig = plt.figure()
for i, det_name in enumerate(detectors):
    plt.loglog(total_mass_range[0:len(hzn_store[det_name])], hzn_store[det_name], label=det_name, color='k', ls=ls[i], alpha=0.2)
plt.loglog(total_mass_range, hzn_store['nwk'], label='CE1+CE2+ET', color='k')
plt.loglog(total_mass_range, hzn_store['NA'], label='NewAthena', color='c')
plt.ylim(1.5e-2, 1.5e2)
plt.xlim(min(total_mass_range), max(total_mass_range))
plt.fill_between(total_mass_range, y1=1, y2=3, color='r', alpha=0.25)
plt.fill_between(total_mass_range, y1=2, y2=2.5, color='r', alpha=0.25, label='peak AGN activity')
plt.legend()
plt.xlabel("total mass [M$_\odot$]")
plt.ylabel("redshift horizon")
plt.savefig("horizon_3G_network.png")
plt.close()


