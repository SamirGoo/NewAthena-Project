import numpy as np

def shakura_sunyaev_point(R_physical, M_BH_solar, m_dot=0.1, alpha=0.1):
    """
    Calculates the local sound speed and midplane gas density for a standard 
    Shakura-Sunyaev AGN disk at a specific physical radius.
    
    Parameters:
    -----------
    R_physical : float
        The radius where you want the calculation, in Astronomical Units (AU).
    M_BH_solar : float
        The mass of the central Black Hole in units of Solar Masses (M_sun).
    m_dot : float, optional
        Accretion rate normalized to the Eddington rate (dot{M} / dot{M}_edd). Default 0.1.
    alpha : float, optional
        The Shakura-Sunyaev viscosity parameter. Default 0.1.
        
    Returns:
    --------
    dict
        A dictionary containing the local zone type, sound speed (km/s), 
        and gas midplane density (g/cm^3).
    """
    # Physical and Astronomical Constants
    G = 6.67430e-8      # Gravitational constant (cm^3 g^-1 s^-2)
    c = 2.99792e10      # Speed of light (cm/s)
    M_sun = 1.98847e33  # Solar mass (g)
    AU_to_cm = 1.4959787e13 # 1 AU in cm
    
    # 1. Transform variables into the standard dimensionless format
    M_8 = M_BH_solar / 1e8
    R_g = (G * (M_BH_solar * M_sun)) / (c**2) # Gravitational radius in cm
    r_cm = R_physical * AU_to_cm
    r_star = r_cm / R_g # Dimensionless radius (r / R_g)
    
    # ISCO / No-slip boundary condition check (for a non-spinning Schwarzschild BH)
    if r_star <= 6.0:
        return {
            "Zone": "Inside ISCO (Inside Edge)",
            "Sound Speed (cm/s)": 0.0,
            "Gas Density (g/cm^3)": 0.0,
            "Valid": False
        }
        
    # Boundary correction function J(r)
    J = 1.0 - np.sqrt(6.0 / r_star)
    
    # 2. Calculate the Transition Boundaries (in units of r_star)
    r_rad_gas = 100.0 * (alpha * M_8)**(2.0/21.0) * (m_dot)**(16.0/21.0)
    r_scat_Kram = 1350.0 * (m_dot)**(2.0/3.0)
    r_self_grav = 3100.0 * (alpha)**(2.0/9.0) * (M_8)**(-2.0/9.0) * (m_dot)**(4.0/9.0)
    
    # 3. Determine the zone and calculate cs (km/s) and rho (g/cm^3)
    if r_star > r_self_grav:
        # Disk undergoes gravitational fragmentation
        zone = "Beyond Self-Gravity Radius (Fragmented/Star-Forming)"
        c_s = None
        rho = None
        valid = False
        
    elif r_star <= r_rad_gas:
        # Zone 1 (Innermost): Radiation Pressure & Electron Scattering
        zone = "Innermost Zone (Radiation Pressure)"
        c_s = (3.0 / (2.0 * np.sqrt(3.0))) * (m_dot / alpha) * (c / 1e5) * J
        # Density scaling for radiation zone
        rho = 4.3e-12 * (alpha * M_8)**(-1) * m_dot**(-2) * r_star**(1.5) * J**(-2)
        valid = True
        
    elif r_star <= r_scat_Kram:
        # Zone 2 (Middle): Gas Pressure & Electron Scattering
        zone = "Middle Zone (Gas Pressure, Ionized)"
        c_s = 62.4 * alpha**(-0.1) * m_dot**(0.2) * M_8**(-0.1) * r_star**(-0.9) * J**(0.2)
        rho = 2.0e-10 * alpha**(-0.7) * m_dot**(0.4) * M_8**(-0.7) * r_star**(-1.65) * J**(0.4)
        valid = True
        
    else:
        # Zone 3 (Outer): Gas Pressure & Kramers' Opacity
        zone = "Outer Zone (Gas Pressure, Neutral Atoms)"
        c_s = 28.5 * alpha**(-0.1) * m_dot**(0.3) * M_8**(-0.1) * r_star**(-1.125) * J**(0.3)
        rho = 6.2e-9 * alpha**(-0.7) * m_dot**(0.55) * M_8**(-0.7) * r_star**(-1.875) * J**(0.55)
        valid = True

    return {
        "Zone": zone,
        "Sound Speed (cm/s)": round(c_s * 10**5, 2) if c_s is not None else "N/A",
        "Gas Density (g/cm^3)": f"{rho:.2e}" if rho is not None else "N/A",
        "Valid": valid,
        "Radius (Rg)": round(r_star, 1),
        "Self-Gravity Limit (Rg)": round(r_self_grav, 1)
    }

'''
# --- EXAMPLE USAGE ---
result = shakura_sunyaev_point(R_physical=20.0, M_BH_solar=1e8, m_dot=0.1, alpha=0.1)

print("--- AGN Disk Conditions ---")
for key, val in result.items():
    print(f"{key}: {val}")
'''


