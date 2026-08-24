#plotting for the poster

#define 3 scenarios for AGN.
#use isobels code for agn modelling

import GWFish.modules as gw
import pandas as pd
from astropy.cosmology import Planck18
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.constants as const

# Import file of time delay calcs
import timedelaycalcs as df

# Import file of AGN model

import agn_model as agn


# Scenarios 

scenario_1 = {
    "Mbh": 1e7,
    "le": 0.1,
    "alpha": 0.1
}

scenario_2 = {
    "Mbh": 1e8,
    "le": 0.5,
    "alpha": 0.01
}

scenario_3 = {
    "Mbh": 1e9,
    "le": 1.0,
    "alpha": 0.01
}

# For each scenario compute the AGN model 

disk1 = agn.get_agn_model(mbh=scenario_1["Mbh"], le=scenario_1["le"], alpha=scenario_1["alpha"])

disk2 = agn.get_agn_model(mbh=scenario_2["Mbh"], le=scenario_2["le"], alpha=scenario_2["alpha"])

disk3 = agn.get_agn_model(mbh=scenario_3["Mbh"], le=scenario_3["le"], alpha=scenario_3["alpha"])

# for each of these i have disk.rho and disk.cs which are functions of radius.
# Assuming that all the mergers happen at the same radius 

R_rs = 1000  #10-10^6 in Chen, 10^2-10^3 schw radius? so using 1000 for now

disk1_rho = np.interp(R_rs, disk1.R/disk1.Rs, disk1.rho)
disk1_cs = np.interp(R_rs, disk1.R/disk1.Rs, disk1.cs) / 1000 #converting to kms as this is needed for later
disk2_rho = np.interp(R_rs, disk2.R/disk2.Rs, disk2.rho)
disk2_cs = np.interp(R_rs, disk2.R/disk2.Rs, disk2.cs) / 1000
disk3_rho = np.interp(R_rs, disk3.R/disk3.Rs, disk3.rho)
disk3_cs = np.interp(R_rs, disk3.R/disk3.Rs, disk3.cs) / 1000

# print("Disk 1 rho:", disk1_rho)
# print("Disk 1 cs:", disk1_cs)
# print("Disk 2 rho:", disk2_rho)
# print("Disk 2 cs:", disk2_cs)
# print("Disk 3 rho:", disk3_rho)
# print("Disk 3 cs:", disk3_cs)

#now we have AGN params to use later. Now need to repeat the time delay calcs for each of these scenarios 

#Setting params

from scipy.stats import qmc

sampler = qmc.LatinHypercube(d=5)
sample = sampler.random(n=200) #increasing to 200 as only doing 3 disk scenarios. 

M = 50 + sample[:,0]*(1000-50)
Q = 0.25 + sample[:,1]*(0.95-0.25)
S1 = sample[:,2]*0.9
S2 = sample[:,3]*0.9
Z = 2.0 + sample[:,4]*(2.5-2.0) #redshift range from 2.0 to 2.5, to be within isobels ranges

N = M.size

parameters = {
    'chirp_mass': M.flatten(),  
    'mass_ratio': Q.flatten(), 
    'luminosity_distance': np.full(N, Planck18.luminosity_distance(Z.flatten()).value), #changing to redshift 2 to be within isobels ranges 
    'theta_jn': np.full(N, 0.0),
    'ra': np.full(N, 4.310664640963899),
    'dec': np.full(N, -0.7176488681791958),
    'psi': np.full(N, 0.2),
    'phase': np.full(N, 2.8),
    'geocent_time': np.full(N, 0.00475200053340101),
    'a_1':S1.flatten(), 
    'a_2':S2.flatten(), 
    'lambda_1':np.full(N, 0), 
    'lambda_2':np.full(N, 0),
    'tilt_1': np.full(N, 0), #testing aligned spins first, so costilt1 = 1, tilt1 = 0, tilt2 = 0, phi12 = 0
    'tilt_2': np.full(N, 0),
    'phi_12': np.full(N, 0)}
parameters = pd.DataFrame(parameters)
parameters

#setting fisher params, waveform, and reference frequency 
from GWFish.modules.waveforms import LALFD_Waveform

waveform_class = LALFD_Waveform
waveform_name = "IMRPhenomXAS"
f_ref = 10.

fisher_parameters = ['chirp_mass', 'mass_ratio', 'a_1', 'a_2' ]

detectors = ['ET', 'CE1', 'CE2']

for i in range(N):
    source = parameters.iloc[[i]]

    detected, network_snr, parameter_errors, sky_localization = gw.fishermatrix.compute_network_errors(
            network = gw.detection.Network(detector_ids = detectors, config="detectors.yaml", detection_SNR = (0., 12.0)),
            parameter_values = source,
            fisher_parameters=fisher_parameters, 
            waveform_model = waveform_name,
            f_ref = 10.,
            
            # use_duty_cycle = False, # default is False anyway
            save_matrices = True, # default is False anyway, put True if you want Fisher and covariance matrices in the output
            save_matrices_path = f'outputs/gwfishMQSpin/poster/source_{i}' #Mass Ratio, Chirp Mass, Spins outputs 
            ) 

#looping over detectors and sources to get samples for each param
chirp_mass_samples = []
mass_ratio_samples = []
chirp_mass_covariances = []
mass_ratio_covariances = []
a_1_samples = []
a_2_samples = []
a_1_covariances = []
a_2_covariances = []

for i in range(len(parameters)):

    C_i = np.load(f"outputs/gwfishMQSpin/poster/source_{i}/inv_fisher_matrices.npy")[0]

    theta0 = (parameters.iloc[i][fisher_parameters].values.astype(float)) #for caluclating the mean

    samples_i = np.random.multivariate_normal(mean = theta0, cov = C_i, size = 1000) #could increase the size here but not doing yet to keep speeds high 

    chirp_mass_samples.append(samples_i[:, 0])
    mass_ratio_samples.append(samples_i[:, 1])
    a_1_samples.append(samples_i[:, 2])
    a_2_samples.append(samples_i[:, 3])
    chirp_mass_covariances.append(np.sqrt(C_i[0,0])) #fixing how covariances are stored 
    mass_ratio_covariances.append(np.sqrt(C_i[1,1]))
    a_1_covariances.append(np.sqrt(C_i[2,2]))
    a_2_covariances.append(np.sqrt(C_i[3,3]))

#
costilt1 = 1.0 #defining this based on the values in the dataframe, could include them in the loop if needed later on
costilt2 = 1.0
phi12 = 0.0

#At this point, looping over samples and computing the time delays for each of the 3 scenarios

summaries = []

#we know the mass of the supermassive BH we set for each disk scenario 

mass_smbh = [1e7, 1e8, 1e9] #in solar masses for each disk scenario

for source_idx in range(len(chirp_mass_samples)):
    

    for disk, disk_name, m_smbh in [(disk1, "disk1", mass_smbh[0]), (disk2, "disk2", mass_smbh[1]), (disk3, "disk3", mass_smbh[2])]:

        chirp_samples = chirp_mass_samples[source_idx]
        mass_ratio_samples_source = mass_ratio_samples[source_idx]
        a_1_samples_source = a_1_samples[source_idx]
        a_2_samples_source = a_2_samples[source_idx]

        remnant_mass_samples = []
        kick_velocity_samples = []
        delay_time_samples = []
        chi_eff_samples = []
        H_samples = []
        rho_samples = []

        luminosity_chen_samples = []
        luminosity_simple_samples = []

        H = disk.Rs * 0.1 #assuming H = 0.1 Rs for all scenarios, could change this to be a function of the disk params if needed
        rho_agn = np.interp(R_rs, disk.R/disk.Rs, disk.rho)
        c_s = np.interp(R_rs, disk.R/disk.Rs, disk.cs) / 1000 #converting to kms

        M_smbh = m_smbh

        for chirp, q, chi1, chi2 in zip(chirp_samples, mass_ratio_samples_source, a_1_samples_source, a_2_samples_source):

            if not (0 < q <= 1):
                continue

            if chirp <= 0:
                continue

            if not (0 <= chi1 <= 1 and 0 <= chi2 <= 1):
                continue

            m1, m2 = df.m1_m2_from_chirp_q(chirp, q)
            M_rem = df.Mrem_NRSURfit(m1, m2, chi1, chi2, costilt1=costilt1, costilt2=costilt2, phi12=phi12)
            Vkick = df.Vkick_NRfit(m1, m2, chi1, chi2, costilt1=costilt1, costilt2=costilt2, phi12=phi12)
            chi_eff = (m1 * chi1 * costilt1 + m2 * chi2 * costilt2) / (m1 + m2) #calculating effective spin param for each sample

            r_cav = df.cavity_radius_cm(
                M_rem_msun=M_rem,
                M_smbh_msun=M_smbh,
                R_cm=1e16,
                H_cm=H,
                vk_kms=Vkick,
                cs_kms=c_s,
                mode="0.6H"
            )
            t_delay = df.total_delay_s( 
                M_rem_msun=M_rem,
                M_smbh_msun=M_smbh,
                R_cm=1e16,
                H_cm=H,
                rho_agn=rho_agn, 
                vk_kms=Vkick,
                cs_kms=c_s,
                eta_jet=0.1,
                theta_0=0.17,
                cavity_mode="0.6H",
                n_acc=1.0
            )

            remnant_mass_samples.append(M_rem)
            kick_velocity_samples.append(Vkick)
            delay_time_samples.append(t_delay / 86400) # Convert seconds to days
            chi_eff_samples.append(chi_eff)
            H_samples.append(H)
            rho_samples.append(rho_agn)

            #luminosity calculations below: 
            Mdot = df.bhl_rate_g_per_s(M_rem, rho_agn=rho_agn, vk_kms=Vkick, cs_kms=c_s) #mrem here needs to be in solar masses
            luminosity_chen = df.cocoon_luminosity_chen(Mdot,rho_agn=rho_agn, H=H, eta_jet=0.1, t_breakout=df.t_breakout_s(H, rho_agn, M_rem, vk_kms=Vkick, cs_kms=c_s, eta_jet=0.1, theta_0=0.17))

            luminosity_simple = df.cocoon_luminosity(M_rem, rho_agn, Vkick, cs=c_s)  #i think mrem here also needs to be in solar masses

            luminosity_chen_samples.append(luminosity_chen)
            luminosity_simple_samples.append(luminosity_simple)



        median_delay = np.median(delay_time_samples)
        delay_p16 = np.percentile(delay_time_samples, 16)
        delay_p84 = np.percentile(delay_time_samples, 84)

        median_kick = np.median(kick_velocity_samples)
        kick_p16 = np.percentile(kick_velocity_samples, 16)
        kick_p84 = np.percentile(kick_velocity_samples, 84)

        median_Mrem = np.median(remnant_mass_samples)
        Mrem_p16 = np.percentile(remnant_mass_samples, 16)
        Mrem_p84 = np.percentile(remnant_mass_samples, 84)

        median_chi_eff = np.median(chi_eff_samples)
        chi_eff_p16 = np.percentile(chi_eff_samples, 16)
        chi_eff_p84 = np.percentile(chi_eff_samples, 84)

        median_H = np.median(H_samples)
        median_rho = np.median(rho_samples)

        luminosity_chen_median = np.median(luminosity_chen_samples)
        luminosity_chen_p16 = np.percentile(luminosity_chen_samples, 16)
        luminosity_chen_p84 = np.percentile(luminosity_chen_samples, 84)

        luminosity_simple_median = np.median(luminosity_simple_samples)
        luminosity_simple_p16 = np.percentile(luminosity_simple_samples, 16)
        luminosity_simple_p84 = np.percentile(luminosity_simple_samples, 84)


        summaries.append({
            "disk": disk_name,
            "source": source_idx,
            "chirp_mass": parameters.iloc[source_idx]["chirp_mass"],
            "mass_ratio": parameters.iloc[source_idx]["mass_ratio"],
            "a_1": parameters.iloc[source_idx]["a_1"],
            "a_2": parameters.iloc[source_idx]["a_2"],
            "luminosity_distance": parameters.iloc[source_idx]["luminosity_distance"],
            "median_chi_eff": median_chi_eff,
            "chi_eff_p16": chi_eff_p16,
            "chi_eff_p84": chi_eff_p84,
            "median_delay": median_delay,
            "delay_p16": delay_p16,
            "delay_p84": delay_p84,
            "median_kick": median_kick,
            "kick_p16": kick_p16,
            "kick_p84": kick_p84,
            "median_Mrem": median_Mrem,
            "Mrem_p16": Mrem_p16,
            "Mrem_p84": Mrem_p84, 
            "H": median_H,
            "rho_agn": median_rho,
            "luminosity_chen": luminosity_chen_median,
            "chen_p16": luminosity_chen_p16,
            "chen_p84": luminosity_chen_p84,
            "luminosity_simple": luminosity_simple_median,
            "simple_p16": luminosity_simple_p16,
            "simple_p84": luminosity_simple_p84
        })

summary = pd.DataFrame(summaries)

#plotting with these summary values 

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

fig, ax = plt.subplots(figsize=(8,6))

# Viridis colours
cmap = plt.get_cmap("viridis")
colours = cmap(np.linspace(0, 1, 4)) #cutting out the yellow, hard to see on the plot, so using 4 colours and skipping the 3rd one

for colour, disk_name in zip([colours[0], colours[1], colours[3]], ["disk1", "disk2", "disk3"]):

    subset = summary[summary["disk"] == disk_name].copy()

    # important: sort by remnant mass
    subset = subset.sort_values("median_Mrem")

    x = subset["median_Mrem"].values

    y = subset["median_delay"].values
    y_low = subset["delay_p16"].values
    y_high = subset["delay_p84"].values

    # median line
    ax.scatter(
        x,
        y,
        lw=2,
        color=colour,
        label=disk_name
    )

    # uncertainty band
    ax.fill_between(
        x,
        y_low,
        y_high,
        color=colour,
        alpha=0.25
    )

ax.set_xlabel(r"Remnant mass [$M_\odot$]")
ax.set_ylabel("Delay time [days]")

ax.legend()

ax.set_yscale("log")   # probably useful for delays
ax.set_xscale("log")   # seeing how this affects the look of the plot. times are still muuch larger than I am expecting. maybe need to look into which part of the time delay is contriburing this? 

plt.tight_layout()
plt.savefig("poster/delay_vs_remnant_mass.png")
plt.show()


######## plotting luminosity 

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 6))


linestyles = {
    "chen": "-",
    "simple": ":"
}

for colour, disk_name in zip([colours[0], colours[1], colours[3]], ["disk1", "disk2", "disk3"]):

    subset = summary[summary["disk"] == disk_name].copy()
    subset = subset.sort_values("median_Mrem")

    x = subset["median_Mrem"].values

    # Chen luminosity
    ax.scatter(
        x,
        subset["luminosity_chen"],
        color=colour,
        linestyle="-",
        lw=2,
        label=f"{disk_name} (Chen)"
    )

    # uncertainty band for chen luminosity
    y_low = subset["chen_p16"].values
    y_high = subset["chen_p84"].values
    ax.fill_between(
        x,
        y_low,
        y_high,
        color=colour,
        alpha=0.25
    )

    # # Simple luminosity removing simple luminosity for now - think something may be wrong here
    # ax.plot(
    #     x,
    #     subset["luminosity_simple"],
    #     color=colour,
    #     linestyle=":",
    #     lw=2,
    #     label=f"{disk_name} (Simple)"
    # )

ax.set_xlabel(r"Remnant mass [$M_\odot$]")
ax.set_ylabel(r"Luminosity [erg s$^{-1}$]")

ax.set_yscale("log")

ax.legend()
plt.tight_layout()

plt.savefig("poster/luminosity_vs_remnant_mass.png")
plt.show()
