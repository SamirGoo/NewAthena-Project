#going to do the time delay analysis here so that I can run this on the cluster 
import GWFish.modules as gw
import pandas as pd
from astropy.cosmology import Planck18
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.constants as const

# Import file of time delay calcs
import timedelaycalcs as df

#grid of parameters 
chirp_masses = np.linspace(50, 1000, 5)  

q_grid = np.linspace(0.25,0.95,5)

a1_grid = np.linspace(0.0, 0.99, 5)  

a2_grid = np.linspace(0.0, 0.99, 5) 

M, Q, S1, S2 = np.meshgrid(chirp_masses, q_grid, a1_grid, a2_grid, indexing="ij")

N = M.size

#adding new paramaters which are needed, tilt 1 and 2, phi 12 etc 
parameters = {
    'chirp_mass': M.flatten(),  
    'mass_ratio': Q.flatten(), 
    'luminosity_distance': np.full(N, Planck18.luminosity_distance(0.00980).value),
    'theta_jn': np.full(N, 2.545065595974997),
    'ra': np.full(N, 3.4461599999999994),
    'dec': np.full(N, -0.4080839999999999),
    'psi': np.full(N, 0.),
    'phase': np.full(N, 0.),
    'geocent_time': np.full(N, 1187008882.4),
    'a_1':S1.flatten(), 
    'a_2':S2.flatten(), 
    'lambda_1':np.full(N, [368.17802383555687]), 
    'lambda_2':np.full(N, [586.5487031450857]),
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

detectors = ['CE1', 'CE2', 'ET']

print(type(detectors))

for det in detectors:

        print(det)
        print(type(det))

        for i in range(N):
                source = parameters.iloc[[i]]

                detected, network_snr, parameter_errors, sky_localization = gw.fishermatrix.compute_network_errors(
                        network = gw.detection.Network(detector_ids = [det], config="detectors.yaml", detection_SNR = (0., 12.0)),
                        parameter_values = source,
                        fisher_parameters=fisher_parameters, 
                        waveform_model = waveform_name,
                        f_ref = 10.,
                        
                        # use_duty_cycle = False, # default is False anyway
                        save_matrices = True, # default is False anyway, put True if you want Fisher and covariance matrices in the output
                        save_matrices_path = f'outputs/gwfishMQSpin/{det}/source_{i}' #Mass Ratio, Chirp Mass, Spins outputs 
                        ) 

#looping over detectors and sources to get samples for each param
chirp_mass_samples = {det: [] for det in detectors}
mass_ratio_samples = {det: [] for det in detectors}
chirp_mass_covariances = {det: [] for det in detectors}
mass_ratio_covariances = {det: [] for det in detectors}
a_1_samples = {det: [] for det in detectors}
a_2_samples = {det: [] for det in detectors}
a_1_covariances = {det: [] for det in detectors}
a_2_covariances = {det: [] for det in detectors}

for det in detectors:

    for i in range(len(parameters)):

        C_i = np.load(f"outputs/gwfishMQSpinNUR/{det}/source_{i}/inv_fisher_matrices.npy")[0]

        theta0 = (parameters.iloc[i][fisher_parameters].values.astype(float)) #for caluclating the mean

        samples_i = np.random.multivariate_normal(mean = theta0, cov = C_i, size = 1000)

        chirp_mass_samples[det].append(samples_i[:, 0])
        mass_ratio_samples[det].append(samples_i[:, 1])
        a_1_samples[det].append(samples_i[:, 2])
        a_2_samples[det].append(samples_i[:, 3])
        chirp_mass_covariances[det].append(C_i[:, 0])
        mass_ratio_covariances[det].append(C_i[:, 1])
        a_1_covariances[det].append(C_i[:, 2])
        a_2_covariances[det].append(C_i[:, 3])

#
costilt1 = 1.0 #defining this based on the values in the dataframe, could include them in the loop if needed later on
costilt2 = 1.0
phi12 = 0.0

# source_posteriors = {det: {} for det in detectors}
summaries = {det: [] for det in detectors}

#now performing the time delay calcualtions here - for plotting 

for det in detectors:

    # source_posteriors[det] = {}
    summaries[det] = []

    for source_idx in range(len(chirp_mass_samples[det])):
        chirp_samples = chirp_mass_samples[det][source_idx]
        mass_ratio_samples_source = mass_ratio_samples[det][source_idx]
        a_1_samples_source = a_1_samples[det][source_idx]
        a_2_samples_source = a_2_samples[det][source_idx]

        remnant_mass_samples = []
        kick_velocity_samples = []
        delay_time_samples = []
        chi_eff_samples = []

        for chirp, q, chi1, chi2 in zip(chirp_samples, mass_ratio_samples_source, a_1_samples_source, a_2_samples_source):
            m1, m2 = df.m1_m2_from_chirp_q(chirp, q)
            M_rem = df.Mrem_NRSURfit(m1, m2, chi1, chi2, costilt1=costilt1, costilt2=costilt2, phi12=phi12)
            Vkick = df.Vkick_NRfit(m1, m2, chi1, chi2, costilt1=costilt1, costilt2=costilt2, phi12=phi12)
            chi_eff = (m1 * chi1 * costilt1 + m2 * chi2 * costilt2) / (m1 + m2) #calculating effective spin param for each sample

            r_cav = df.cavity_radius_cm(
                M_rem_msun=M_rem,
                M_smbh_msun=1e8,
                R_cm=1e16,
                H_cm=1e14,
                vk_kms=Vkick,
                cs_kms=30.0,
                mode="0.6H"
            )
            t_delay = df.total_delay_s(
                M_rem_msun=M_rem,
                M_smbh_msun=1e8,
                R_cm=1e16,
                H_cm=1e14,
                rho_agn=1e-5, #range 10^-3 to 10^-8
                vk_kms=Vkick,
                cs_kms=30.0,
                cavity_mode="0.6H",
                n_acc=3.0
            )

            remnant_mass_samples.append(M_rem)
            kick_velocity_samples.append(Vkick)
            delay_time_samples.append(t_delay / df.DAY_CGS)
            chi_eff_samples.append(chi_eff)

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

        # source_posteriors[det][source_idx] = {
        #     "remnant_mass_samples": np.array(remnant_mass_samples),
        #     "remnant_mass_median": np.median(remnant_mass_samples),
        #     "spin_1_median": np.median(a_1_samples_source),
        #     "spin_2_median": np.median(a_2_samples_source),
        #     "kick_velocity_samples": np.array(kick_velocity_samples),
        #     "delay_time_samples": np.array(delay_time_samples),
        # }

        summaries[det].append({
            "source": source_idx,
            "chirp_mass": parameters.iloc[source_idx]["chirp_mass"],
            "mass_ratio": parameters.iloc[source_idx]["mass_ratio"],
            "a_1": parameters.iloc[source_idx]["a_1"],
            "a_2": parameters.iloc[source_idx]["a_2"],
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
            "Mrem_p84": Mrem_p84
        })


summaries = {
    det: pd.DataFrame(rows)
    for det, rows in summaries.items()
}

#now we can perform the plotting of these time delay distributions

#plotting delay time vs remnant mass, 

cmap = "vidris"

colours = {
    "CE1": cmap(0.1),
    "CE2": cmap(0.5),
    "ET": cmap(0.9)
}

plt.figure(figsize=(8, 6))

for det in detectors:

    summary = summaries[det]

    delay_yerr = np.vstack([
        summary["median_delay"] - summary["delay_p16"],
        summary["delay_p84"] - summary["median_delay"]
    ])

    mrem_xerr = np.vstack([
        summary["median_Mrem"] - summary["Mrem_p16"],
        summary["Mrem_p84"] - summary["median_Mrem"]
    ])

    plt.scatter(
        summary["median_Mrem"],
        summary["median_delay"],
        color=colours[det],
        s=80,
        alpha=0.8,
        label=det
    )

    plt.errorbar(
        summary["median_Mrem"],
        summary["median_delay"],
        yerr=delay_yerr,
        fmt="none",
        color=colours[det],
        alpha=0.3
    )

plt.xlabel(r"Remnant mass ($M_\odot$)")
plt.ylabel("Delay time (days)")
plt.yscale("log")
plt.legend()

plt.tight_layout()
plt.savefig("outputs/gwfishMQSpinNUR/delay_time_vs_remnant_mass.png")


# plotting delay time vs spin (these are the 2 biggest impacts on kick velocity??)

plt.figure(figsize=(8, 6))

for det in detectors:

    summary = summaries[det]

    delay_yerr = np.vstack([
        summary["median_delay"] - summary["delay_p16"],
        summary["delay_p84"] - summary["median_delay"]
    ])

    spin_xerr = np.vstack([
        summary["median_chi_eff"] - summary["chi_eff_p16"],
        summary["chi_eff_p84"] - summary["median_chi_eff"]
    ])

    plt.scatter(
        summary["median_chi_eff"],
        summary["median_delay"],
        color=colours[det],
        s=80,
        alpha=0.8,
        label=det
    )

    plt.errorbar(
        summary["median_chi_eff"],
        summary["median_delay"],
        yerr=delay_yerr,
        fmt="none",
        color=colours[det],
        alpha=0.3
    )

plt.xlabel(r"Effective spin ($\chi_{\text{eff}}$)")
plt.ylabel("Delay time (days)")
plt.yscale("log")
plt.legend()

plt.tight_layout()
plt.savefig("outputs/gwfishMQSpinNUR/delay_time_vs_remnant_mass.png")


