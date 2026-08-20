#going to do the time delay analysis - combined network from the 3 detectors
import GWFish.modules as gw
import pandas as pd
from astropy.cosmology import Planck18
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.constants as const

# Import file of time delay calcs
import timedelaycalcs as df

# #grid of parameters 
# chirp_masses = np.linspace(50, 1000, 5)  

# q_grid = np.linspace(0.25,0.95,5)

# a1_grid = np.linspace(0.0, 0.9, 5)  #reducing the maximum spin in case at some point it is rounding to 1? 

# a2_grid = np.linspace(0.0, 0.9, 5) 

# M, Q, S1, S2 = np.meshgrid(chirp_masses, q_grid, a1_grid, a2_grid, indexing="ij")

from scipy.stats import qmc

sampler = qmc.LatinHypercube(d=5)
sample = sampler.random(n=25) #reducing samples to add in AGN params

M = 50 + sample[:,0]*(1000-50)
Q = 0.25 + sample[:,1]*(0.95-0.25)
S1 = sample[:,2]*0.9
S2 = sample[:,3]*0.9
Z = 2.0 + sample[:,4]*(2.5-2.0) #redshift range from 2.0 to 2.5, to be within isobels ranges

N = M.size

#adding new paramaters which are needed, tilt 1 and 2, phi 12 etc 
parameters = {
    'chirp_mass': M.flatten(),  
    'mass_ratio': Q.flatten(), 
    'luminosity_distance': np.full(N, Planck18.luminosity_distance(Z.flatten()).value), #changing to redshift 2 to be within isobels ranges 
    'theta_jn': np.full(N, 2.545065595974997),
    'ra': np.full(N, 3.4461599999999994),
    'dec': np.full(N, -0.4080839999999999),
    'psi': np.full(N, 0.),
    'phase': np.full(N, 0.),
    'geocent_time': np.full(N, 1187008882.4),
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

detectors = ['ET', 'CE1', 'CE2'] # ,'CE1', 'CE2' removing for now to run quick 

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
            save_matrices_path = f'outputs/gwfishMQSpin/AGN/source_{i}' #Mass Ratio, Chirp Mass, Spins outputs 
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

    C_i = np.load(f"outputs/gwfishMQSpin/AGN/source_{i}/inv_fisher_matrices.npy")[0]

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

#making a grid of parameters for the AGN posteriors  - 10 samples, not wanting to increase by too much at this point. 

rho_grid = np.logspace(-8, -3, len(chirp_mass_samples[1])) #in g/cm^3
H_grid = np.linspace(1e13, 1e15, len(chirp_mass_samples[1])) #in cm


#now performing the time delay calcualtions here - for plotting 
summaries = []

for source_idx in range(len(chirp_mass_samples)):
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

    for chirp, q, chi1, chi2, H, rho_agn in zip(chirp_samples, mass_ratio_samples_source, a_1_samples_source, a_2_samples_source, H_grid, rho_grid):

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
            M_smbh_msun=1e8,
            R_cm=1e16,
            H_cm=H,
            vk_kms=Vkick,
            cs_kms=30.0,
            mode="0.6H"
        )
        t_delay = df.total_delay_s(
            M_rem_msun=M_rem,
            M_smbh_msun=1e8,
            R_cm=1e16,
            H_cm=1e14,
            rho_agn=rho_agn, #range 10^-3 to 10^-8
            vk_kms=Vkick,
            cs_kms=30.0,
            cavity_mode="0.6H",
            n_acc=3.0
        )

        remnant_mass_samples.append(M_rem)
        kick_velocity_samples.append(Vkick)
        delay_time_samples.append(t_delay / 86400) # Convert seconds to days
        chi_eff_samples.append(chi_eff)
        H_samples.append(H)
        rho_samples.append(rho_agn)

        #luminosity calculations below: 
        Mdot = df.bhl_rate_g_per_s(M_rem, rho_agn=rho_agn, vk_kms=Vkick, cs_kms=30.0) #mrem here needs to be in solar masses
        luminosity_chen = df.cocoon_luminosity_chen(Mdot,rho_agn=rho_agn, H=H, eta_jet=0.1, t_breakout=1e10)

        luminosity_simple = df.cocoon_luminosity(M_rem, rho_agn, Vkick, cs=30.0)  #i think mrem here also needs to be in solar masses

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

    # source_posteriors[det][source_idx] = {
    #     "remnant_mass_samples": np.array(remnant_mass_samples),
    #     "remnant_mass_median": np.median(remnant_mass_samples),
    #     "spin_1_median": np.median(a_1_samples_source),
    #     "spin_2_median": np.median(a_2_samples_source),
    #     "kick_velocity_samples": np.array(kick_velocity_samples),
    #     "delay_time_samples": np.array(delay_time_samples),
    # }

    summaries.append({
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

#now we can perform the plotting of these time delay distributions

#plotting delay time vs scale height, colored by density

cmap = plt.get_cmap("viridis")



plt.figure(figsize=(8, 6))


delay_yerr = np.vstack([
    summary["median_delay"] - summary["delay_p16"],
    summary["delay_p84"] - summary["median_delay"]
])

plt.scatter(
    summary["H"],
    summary["median_delay"],
    c=summary["rho_agn"],  # Color by density
    cmap=cmap,
    s=80,
    alpha=0.8
)

plt.errorbar(
    summary["H"],
    summary["median_delay"],
    yerr=delay_yerr,
    fmt="none",
    alpha=0.3
)

plt.xlabel(r"Scale height (H, cm)")
plt.ylabel("Delay time (days)")
plt.yscale("log")
plt.colorbar(label="AGN density (g/cm^3)")

plt.tight_layout()
plt.savefig("outputs/gwfishMQSpin/AGN/delaytime_H.png")


# plotting delay time vs density

plt.figure(figsize=(8, 6))

delay_yerr = np.vstack([
    summary["median_delay"] - summary["delay_p16"],
    summary["delay_p84"] - summary["median_delay"]
])

plt.scatter(
    summary["rho_agn"],
    summary["median_delay"],
    c=summary["H"],  # Color by scale height
    cmap=cmap,
    s=80,
    alpha=0.8
)

plt.errorbar(
    summary["rho_agn"],
    summary["median_delay"],
    yerr=delay_yerr,
    fmt="none",
    alpha=0.3
)

plt.xlabel(r"AGN density (g/cm^3)")
plt.ylabel("Delay time (days)")
plt.yscale("log")
plt.colorbar(label="Scale height (cm)")

plt.tight_layout()
plt.savefig("outputs/gwfishMQSpin/AGN/delaytime_density.png")


#with these varying params now plotting the luminosity 

# plotting luminosity vs density - do i want to add some errors into luminosity? 

plt.figure(figsize=(8, 6))

chen_yerr = np.vstack([
    summary["luminosity_chen"] - summary["chen_p16"],
    summary["chen_p84"] - summary["luminosity_chen"]
])

plt.scatter(
    summary["rho_agn"],
    summary["luminosity_chen"],
    c=summary["H"],  # Color by scale height
    cmap=cmap,
    s=50,
    alpha=0.8,
    marker="o",
    label="Chen cocoon model"
)

plt.errorbar(
    summary["rho_agn"],
    summary["luminosity_chen"],
    yerr=chen_yerr,
    fmt="none",
    alpha=0.3
)

simple_yerr = np.vstack([
    summary["luminosity_simple"] - summary["simple_p16"],
    summary["simple_p84"] - summary["luminosity_simple"]
])

plt.scatter(
    summary["rho_agn"],
    summary["luminosity_simple"],
    c=summary["H"],
    cmap=cmap,
    s=50,
    alpha=0.8,
    marker="^",
    label="Simple cocoon model"
)

plt.errorbar(
    summary["rho_agn"],
    summary["luminosity_simple"],
    yerr=simple_yerr,
    fmt="none",
    alpha=0.3
)

plt.xlabel(r"AGN density (g/cm^3)")
plt.ylabel("Luminosity (erg/s)")
plt.yscale("log")
plt.xscale("log")
plt.colorbar(label="Scale height (cm)")
plt.legend()

plt.tight_layout()
plt.savefig("outputs/gwfishMQSpin/AGN/luminosity_density_H.png")
