import numpy as np


G = 6.67430e-8 # cm^3/g/s^2
c = 2.99792458e10 # cm/s
M_sun = 1.989e33 # g
Mpc_to_cm = 3.085677581e24
k_B = 1.380649e-16           # Boltzmann constant (erg/K)
m_p = 1.6726219e-24          # Proton mass (g)
F_lim = 5e-17 # erg/cm^2/s, NewAthena X-ray flux limit


def sound_speed(T, gamma=5/3, mu=0.6):
    return np.sqrt(gamma * k_B * T / (mu * m_p))  # cm/s


def gravitational_radius(M):
    return G * M / c**2  # cm


def orbital_radius(R, M_AGN):
    return R * gravitational_radius(M_AGN) # cm


def keplerian_velocity(M_AGN, R):
    r = orbital_radius(R, M_AGN)
    return np.sqrt(G * M_AGN / r)  # cm/s


def blanford_znajek(M_binary, rho_AGN, v_k, c_s, f):
    Mdot = (4 * np.pi * G**2 * M_binary**2 * rho_AGN) / ((v_k**2 + c_s**2)**1.5)
    L_j = f * Mdot * c**2
    return L_j


def X_ray_luminosity_flux(M, # Msun, Binary total mass
                          d_L, # Mpc, binary distance
                          T=1e4, # K, temperature, minimum assumed
                          R_orb=1e4, # Gravitational radius to orbital radius
                          fx=0.01, # X-ray conversion efficiency, maximum assumed
                          fj=0.1, # Jet conversion efficiency
                          p_AGN=1e-10, # g/cm^3, AGN mass density, maximum assumed,
                          M_AGN=1e8, # Msun, AGN mass
                          ):
    M_AGN_g = M_AGN * M_sun

    c_s = sound_speed(T) # cm / s
    v_k = keplerian_velocity(M_AGN_g, R_orb) # cm / s

    M_g = M * M_sun # g
    L_j = blanford_znajek(M_g, p_AGN, v_k, c_s, fj) # g m^2 / s^3
    L_x = fx * L_j # erg / s
    d = d_L * Mpc_to_cm # cm
    F = L_x / (4 * np.pi * d**2)
    return F # erg/cm^2/s



def max_distance_NewAthena(M, # Msun, Binary total mass
                           T=1e4, # K, temperature, minimum assumed
                           R_orb=1e4, # Gravitational radius to orbital radius
                           fx=0.01, # X-ray conversion efficiency, maximum assumed
                           fj=0.1, # Jet conversion efficiency
                           p_AGN=1e-10, # g/cm^3, AGN mass density, maximum assumed
                           M_AGN=1e8, # Msun, AGN mass
                           ):

    M_AGN_g = M_AGN * M_sun

    c_s = sound_speed(T) # cm / s
    v_k = keplerian_velocity(M_AGN_g, R_orb) # cm / s

    M_g = M * M_sun # g
    L_j = blanford_znajek(M_g, p_AGN, v_k, c_s, fj) # g m^2 / s^3
    L_x = fx * L_j # erg / s
    d = np.sqrt(L_x / (4.0 * np.pi * F_lim)) # cm
    return d / Mpc_to_cm # Mpc


