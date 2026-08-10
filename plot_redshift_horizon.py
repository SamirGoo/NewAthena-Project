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


import matplotlib.pyplot as plt


base_dir = "."

# Detectors are one CE at LLO location, one CE at Gingin (Australia), ET in Sardinia
detectors = ['CE1', 'CE2', 'ET']
detectors_2G = ['LLO', 'LHO', 'VIR']

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

network_2G = gwf_mods.detection.Network(detector_ids=detectors_2G,
                                     detection_SNR=(0., 12.),
                                     config=pathlib.Path(base_dir + '/detectors.yaml'))

dets = {}
for det_name in detectors:
    dets[det_name] = Detector(det_name,
                              config=pathlib.Path(base_dir + '/detectors.yaml'))

dets_2G = {}
for det_name in detectors_2G:
    dets_2G[det_name] = Detector(det_name,
                              config=pathlib.Path(base_dir + '/detectors.yaml'))

total_mass_range = np.logspace(1.5, 4.4, 200)
hzn_store = {'nwk': [], 'CE1': [], 'CE2': [], 'ET': []}
hzn_store_2G = {'nwk': [], 'LLO': [], 'LHO': [], 'VIR': []}

for tmass in total_mass_range:
    params['total_mass'] = tmass
    params['mass_1'], params['mass_2'] = bb.gw.conversion.total_mass_and_mass_ratio_to_component_masses(
                                             mass_ratio=params['mass_ratio'], total_mass=params['total_mass'])

    hzn, rz = horizon(params, network, waveform_model='IMRPhenomXAS', target_SNR=12.0)
    print("tmass: {}, m1: {}, m2: {},  horizon dL: {}, z: {}".format(tmass, params['mass_1'], params['mass_2'], hzn, rz))
    hzn_store['nwk'].append(rz)
    for det_name in detectors:
        try:
            hzni, rzi = horizon(params, dets[det_name], waveform_model='IMRPhenomXAS', target_SNR=12.0)
            hzn_store[det_name].append(rzi)
        except:
            print(det_name, "dropped")

    hzn_2G, rz_2G = horizon(params, network_2G, waveform_model='IMRPhenomXAS', target_SNR=12.0)
    print("tmass: {}, m1: {}, m2: {},  horizon dL: {}, z: {}".format(tmass, params['mass_1'], params['mass_2'], hzn_2G, rz_2G))
    hzn_store_2G['nwk'].append(rz_2G)
    for det_name in detectors_2G:
        try:
            hzni_2g, rzi_2g = horizon(params, dets_2G[det_name], waveform_model='IMRPhenomXAS', target_SNR=12.0)
            hzn_store_2G[det_name].append(rzi_2g)
        except:
            print(det_name, "dropped")

# Read NewAthena data
# Format: (f"{mass}, {zmaxmin}, {zmaxmean}, {zmaxmax}\n")
NA_files_per_vk = {
    "50 km/s": 'newathena_horizon_vk_0.txt',
    "500 km/s": 'newathena_horizon_vk_1.txt',
    "5000 km/s": 'newathena_horizon_vk_2.txt'
}

# Three shades
colors = plt.cm.viridis([0.15, 0.55, 0.90])

fig = plt.figure()

for i, (k, f) in enumerate(NA_files_per_vk.items()):
    data = np.loadtxt(f, delimiter=",")
    masses   = data[:, 0]
    zmaxmins = data[:, 1]
    zmaxmeans = data[:, 2]
    zmaxmaxs = data[:, 3]
    plt.fill_between(masses, zmaxmins, zmaxmaxs, alpha=0.25, color=colors[i])
    plt.loglog(masses, zmaxmeans, label=k, color=colors[i])


for i, det_name in enumerate(detectors):
    plt.loglog(total_mass_range[0:len(hzn_store[det_name])], hzn_store[det_name], label=det_name, color='k', ls=ls[i], lw=1)
plt.loglog(total_mass_range, hzn_store['nwk'], label='CE1+CE2+ET', color='k')

for i, det_name in enumerate(detectors_2G):
    plt.loglog(total_mass_range[0:len(hzn_store_2G[det_name])], hzn_store_2G[det_name], label=det_name, color='lightgrey', ls=ls[i], lw=1)

plt.loglog(total_mass_range, hzn_store_2G['nwk'], label='LLO+LHO+VIR', color='lightgrey')

plt.ylim(1e-1, 1e2)
plt.xlim(min(total_mass_range), max(total_mass_range))
plt.fill_between(total_mass_range, y1=1, y2=3, color='r', alpha=0.25)
plt.fill_between(total_mass_range, y1=2, y2=2.5, color='r', alpha=0.25, label='peak AGN activity')
plt.legend(ncols=3, loc="lower center")
plt.xlabel("total mass [M$_\odot$]")
plt.ylabel("redshift horizon")
plt.savefig("horizon_3G_network.pdf")
plt.close()


