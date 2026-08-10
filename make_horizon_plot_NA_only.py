import numpy as np
import matplotlib.pyplot as plt
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


athena = AthenaWFI(
    "rsp/NewAthena_WFI_13rows_LDA_wo_filter_FoVAvg_20260511.rsp",
    "bkgd/NewAthena_WFI_13rows_LDA_20260528_bkgd_sum_9asec_wo_filter_FoVAvg.pha",
)

# agn density in g / cm^3 - vary this: inner disk can be 10^-8 to 10^-3, outer disk can go down to 10^-11 or lower.
# motivation for sampling log-uniform distribution: mass density correlates with distribution of BHs? More likely to be in inner disk, especially if massive?
log_rho_agn_samples = np.random.uniform(-8, -3, 100)
rho_agn_samples = np.power(10, log_rho_agn_samples)
cs = 1e7 # sound speed

masses = np.logspace(
    1.5,
    4.4,
    200,
)

z_grid = np.logspace(
    -1,
    2,
    300,
)

Eobs = athena.energy
Erest = [Eobs * (1 + z) for z in z_grid]

# kick velocities references - https://arxiv.org/abs/2106.07179, https://pure.mpg.de/rest/items/item_3626410_5/component/file_3626411/content
# ask parthapratim for kick estimation code from Appendix A of first paper, otherwise code up myself
# kick units are cm/s
kick_velocities = {
    "50 km/s": 0.5e7,
    "500 km/s": 5e7,
    "5000 km/s": 5e8,
}

colors = ['red', 'blue', 'green', 'magenta']

for i, (label, vk) in enumerate(kick_velocities.items()):

    horizon = {'min': [], 'mean': [], 'max': []}

    outfile = open(f"newathena_horizon_vk_{i}.txt", "w")

    for mass in masses:

        zmax_range = []

        for rho_agn in rho_agn_samples:

            zmax = 0

            Lx = cocoon_luminosity(
                mass,
                rho_agn,
                vk,
                cs,
            )

            kT = cocoon_temperature_keV(
                mass,
                rho_agn,
                vk,
                cs,
            )

            duration = cocoon_duration(
                mass,
                rho_agn,
                vk,
                cs,
            )

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
        zmaxmin = np.percentile(zmax_range, 5, axis=0)
        zmaxmean = np.mean(zmax_range, axis=0)
        zmaxmax = np.percentile(zmax_range, 95, axis=0)

        horizon['min'].append(
            zmaxmin
        )
        horizon['mean'].append(
            zmaxmean
        )
        horizon['max'].append(
            zmaxmax
        )

        outfile.write(f"{mass}, {zmaxmin}, {zmaxmean}, {zmaxmax}\n")

    plt.plot(
        masses,
        horizon['mean'],
        color=colors[i]
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

