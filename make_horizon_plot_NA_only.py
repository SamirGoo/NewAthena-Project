import numpy as np
import matplotlib.pyplot as plt
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

athena = AthenaWFI(
    "rsp/NewAthena_WFI_13rows_LDA_wo_filter_FoVAvg_20260511.rsp",
    "bkgd/NewAthena_WFI_13rows_LDA_20260528_bkgd_sum_9asec_wo_filter_FoVAvg.pha",
)

# Number of AGN parameters and BBH positions to vary
N_agn = 500

# SMBH masses
log_SMBH_masses = np.random.uniform(
    6,
    10,
    N_agn
)
SMBH_masses = np.power(10, log_SMBH_masses)

# Set up agn model - need grid of masses to interpolate over, cut to 3 for testing
M_SMBH_GRID = np.logspace(6, 10, 10)
dimless_rmin, dimless_rmax, log_rho_interp, log_cs_interp = agn_grid_interps(M_SMBH_GRID)

# BBH location in disk, units of Schwarzchild radius of SMBH.
# Only go out to mid disk to avoid low densities where we don't see anything
log_bbh_dimless_radii = np.random.uniform(
    np.log10(dimless_rmin),
    0.5 * np.log10(dimless_rmax),
    N_agn
)
bbh_dimless_radii = np.power(10, log_bbh_dimless_radii)

# Get sound speeds and densities
rho_agn_grid, cs_grid, H_grid = get_disk_properties(log_rho_interp, log_cs_interp, SMBH_masses, bbh_dimless_radii)
# Convert from SI (kg m^-3, m s^-1, m) to cgs (g cm^-3, cm s^-1, cm)
rho_agn_grid *= 1e-3
cs_grid *= 1e2
H_grid *= 1e2


# BBH remnant masses
masses = np.logspace(
    1.5,
    4.4,
    200,
)

# redshift range
z_grid = np.logspace(
    -2,
    2,
    200,
)

Eobs = athena.energy
Erest = [Eobs * (1 + z) for z in z_grid]

# kick velocities references - https://arxiv.org/abs/2106.07179, https://pure.mpg.de/rest/items/item_3626410_5/component/file_3626411/content
# kick units are cm/s
kick_velocities = {
    "50 km/s": 0.5e7,
    "500 km/s": 5e7,
    "5000 km/s": 5e8,
}

colors = ['red', 'blue', 'green', 'magenta']

for i, (label, vk) in enumerate(kick_velocities.items()):

    horizon = {'min': [], 'mean': [], 'max': [], 'maxmax': []}

    outfile = open(f"newathena_horizon_vk_{i}.txt", "w")

    for mass in masses:

        zmax_range = []

        for rho_agn, cs, H in zip(rho_agn_grid, cs_grid, H_grid):

            print()

            zmax = 0

            Lx = cocoon_luminosity(
                mass,
                rho_agn,
                vk,
                cs,
                H,
            ) # in erg/s

            kT = cocoon_temperature_keV(
                mass,
                rho_agn,
                vk,
                cs,
                H
            ) # in keV

            duration = cocoon_duration(
                mass,
                rho_agn,
                vk,
                cs,
                H
            ) # in s

            print("Lx:", Lx, "kT:", kT, "T Kelvin:", kT * 1.16045e7, "duration:", duration)

            exposure = min(
                10000.,
                duration,
            )

            bkg_counts = (
                athena.background_counts(
                        exposure
                )
            )

            B = bkg_counts.sum()

            for zi, z in enumerate(z_grid):

                source_spec = (
                    thermal_bremsstrahlung(
                        Erest[zi],
                        Lx,
                        kT,
                    )
                )
                source_spec *= transmission(
                    Erest[zi]
                )
                flux = observed_flux(
                    Eobs,
                    source_spec,
                    z,
                )

                src_counts = athena.fold(
                    flux,
                    exposure,
                )

                S = src_counts.sum()

                significance = (
                    S /
                    np.sqrt(
                        S + B
                    )
                )

                if significance > 5:
                    zmax = z
                else:
                    break

            # get a zmax for each rho sample
            zmax_range.append(zmax)

        # want to store mean, max and min zmax
        zmaxmin = np.nanpercentile(zmax_range, 5, axis=0)
        zmaxmean = np.nanmean(zmax_range, axis=0)
        zmaxmax = np.nanpercentile(zmax_range, 95, axis=0)
        zmaxmaxmax = max(zmax_range)
        print(min(zmax_range), max(zmax_range))

        horizon['min'].append(
            zmaxmin
        )
        horizon['mean'].append(
            zmaxmean
        )
        horizon['max'].append(
            zmaxmax
        )
        horizon['maxmax'].append(
            zmaxmax
        )

        outfile.write(f"{mass}, {zmaxmin}, {zmaxmean}, {zmaxmax}, {zmaxmaxmax}\n")

    plt.plot(
        masses,
        horizon['mean'],
        color=colors[i]
    )

    plt.plot(
        masses,
        horizon['maxmax'],
        color=colors[i],
        lw=4
    )

    plt.fill_between(
        masses,
        horizon['min'],
        horizon['max'],
        label=label,
        alpha=0.15,
        color=colors[i]
    )

    outfile.close()

plt.xscale("log")
plt.yscale("log")

plt.xlabel(
    "Remnant mass [$M_\\odot$]"
)

plt.ylabel(
    "NewAthena horizon redshift"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "athena_horizon.png",
    dpi=300,
)

plt.close()

