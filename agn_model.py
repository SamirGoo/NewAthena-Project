from pagn import Sirko
import pagn.constants as ct
import numpy as np

from scipy.interpolate import RegularGridInterpolator


def get_agn_model(mbh = 1e8,           # 10^8 solar-mass SMBH
                  le = 0.5,                      # Eddington ratio
                  mdot = None,                   # let the model infer Mdot from le
                  alpha = 0.01,                  # Shakura-Sunyaev alpha
                  X = 0.7,                       # hydrogen mass fraction
                  b = 0,                         # alpha-disk case
                  opacity = "combined",          # combined opacity tables
                  n_resolution = 10_000):

    model = Sirko.SirkoAGN(
        Mbh=mbh * ct.MSun,
        le=le,
        Mdot=mdot,
        alpha=alpha,
        X=X,
        b=b,
        opacity=opacity,
    )
    model.solve_disk(N=n_resolution)
    return model


def agn_grid_interps(Mbh_grid,
                  le = 0.5,                      # Eddington ratio
                  mdot = None,                   # let the model infer Mdot from le
                  alpha = 0.01,                  # Shakura-Sunyaev alpha
                  X = 0.7,                       # hydrogen mass fraction
                  b = 0,                         # alpha-disk case
                  opacity = "combined",          # combined opacity tables
                  n_resolution = 10_000):

    disk0 = get_agn_model(Mbh_grid[0], le, mdot, alpha, X, b, opacity, n_resolution)
    r_grid_dimensionless = disk0.R/disk0.Rs # build interpolator on R scaled to Rs as this is a constant regardless of SMBH mass
    rho_grid = np.zeros((len(Mbh_grid), len(r_grid_dimensionless)))
    cs_grid = np.zeros((len(Mbh_grid), len(r_grid_dimensionless)))

    # Precompute all disks
    for i, Mbh in enumerate(Mbh_grid):

        disk = get_agn_model(Mbh, le, mdot, alpha, X, b, opacity, n_resolution)

        # interpolate onto common dimensionless radius grid
        rho_grid[i] = np.interp(r_grid_dimensionless, disk.R/disk.Rs, disk.rho)
        cs_grid[i]  = np.interp(r_grid_dimensionless, disk.R/disk.Rs, disk.cs)

    # Build 2D interpolators
    log_rho_interp = RegularGridInterpolator(
        (np.log10(Mbh_grid), np.log10(r_grid_dimensionless)),
        np.log10(rho_grid),
    )

    log_cs_interp = RegularGridInterpolator(
        (np.log10(Mbh_grid), np.log10(r_grid_dimensionless)),
        np.log10(cs_grid),
    )

    return min(r_grid_dimensionless), max(r_grid_dimensionless), log_rho_interp, log_cs_interp


def get_disk_properties(log_rho_interp, log_cs_interp, Mbh, r_dimensionless):

    points = np.column_stack([
        np.log10(Mbh),
        np.log10(r_dimensionless)
    ])

    rho = 10**log_rho_interp(points)
    cs  = 10**log_cs_interp(points)

    return rho, cs


