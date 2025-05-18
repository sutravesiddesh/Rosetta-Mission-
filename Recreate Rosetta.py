import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

mu = 1.32712440018e20  # Gravitational parameter for the Sun in km^3/s^2

# def lambert_solver(r1, r2, dt, mu):
#     # Flatten input position vectors
#     r1, r2 = r1.flatten(), r2.flatten()
    
#     # Compute magnitudes of position vectors
#     r1_norm, r2_norm = np.linalg.norm(r1), np.linalg.norm(r2)
    
#     # Calculate cosine and sine of true anomaly difference
#     cos_dnu = np.dot(r1, r2) / (r1_norm * r2_norm)
#     sin_dnu = np.sign(np.cross(r1, r2)[-1]) * np.sqrt(1 - cos_dnu**2)
    
#     # Compute constant A for Lambert's equation
#     A = sin_dnu * np.sqrt(r1_norm * r2_norm / (1 - cos_dnu))
    
#     # Initialize z for Newton-Raphson iteration
#     z = 1.0
    
#     # Iterate to solve for z using Newton-Raphson method
#     for _ in range(50):
#         # Compute Stumpff functions C and S
#         C = (1 - np.cos(np.sqrt(z))) / z if z > 1e-6 else 0.5 - z / 24
#         S = (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z)**3) if z > 1e-6 else 1/6 - z / 120
        
#         # Calculate y parameter
#         y = r1_norm + r2_norm + A * (z * S - 1) / np.sqrt(C)
        
#         # Compute function F and its derivative dFdz
#         F = (y / C)**1.5 * S + A * np.sqrt(y) - np.sqrt(mu) * dt
#         dFdz = (y / C)**1.5 * (1 / (2 * z) * (C - 3 * S / (2 * C)) + 3 * S**2 / (4 * C)) + A / 8 * (3 * S / C * np.sqrt(y) + A * np.sqrt(C / y))
        
#         # Update z using Newton-Raphson step
#         dz = F / dFdz if abs(dFdz) > 1e-10 else 0
#         z -= dz
        
#         # Check for convergence
#         if abs(dz) < 1e-8:
#             break
    
#     # If z is invalid, return simple velocity estimate
#     if z < 0 or np.isnan(z):
#         return (r2 - r1) / dt
    
#     # Compute chi and alpha for velocity calculation
#     chi = np.sqrt(z)
#     alpha = 1 - z * r1_norm / mu
    
#     # Calculate initial velocity vector v1
#     v1 = (r2 - r1 * (1 - chi**2 * C / alpha)) / (np.sqrt(mu) * dt * np.sqrt(C) / chi)
    
#     # If velocity is invalid, return simple velocity estimate
#     if np.any(np.isnan(v1)) or np.any(np.isinf(v1)):
#         return (r2 - r1) / dt
    
#     # Return computed velocity vector
#     return v1
def lambert_problem(r1, r2, dt, mu):
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    cos_theta = np.dot(r1, r2) / (r1_mag * r2_mag)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  

    A = np.sin(theta) * np.sqrt(r1_mag * r2_mag / (1 - np.cos(theta)))
    
    z = 0.1
    tol = 1e-8
    max_iter = 100

    for i in range(max_iter):
        C = stumpff_C(z)
        S = stumpff_S(z)

       
        sqrt_C = np.sqrt(max(C, 1e-8))
        y = r1_mag + r2_mag + A * (z * S - 1) / sqrt_C

        if y < 0:
            z += 0.1
            continue

        try:
            t_z = (y ** 1.5 * S + A * np.sqrt(y)) / np.sqrt(mu)
        except:
            break

        dt_diff = t_z - dt
        if abs(dt_diff) < tol:
            break


        if abs(z) < 1e-5:
            dtdz = np.sqrt(2) * y ** 1.5 / (40 * np.sqrt(mu))  
        else:
            denom = max(1e-8, 2 * z)
            dtdz = (np.sqrt(y) / np.sqrt(mu)) * (
                (1 / denom) * (C - 3 * S / (2 * C)) + (3 * S ** 2) / (4 * C)
            )

        if abs(dtdz) < 1e-10:
            break  

        z -= dt_diff / dtdz
        z = max(z, -100)  

    f = 1 - y / r1_mag
    g = A * np.sqrt(y / mu)
    g_dot = 1 - y / r2_mag

    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - f * r1) / g

    return v1, v2


def stumpff_C(z):
    if abs(z) < 1e-5:
        return 1/2 - z / 24 + z**2 / 720  
    elif z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    else:
        return (np.cosh(np.sqrt(-z)) - 1) / -z

def stumpff_S(z):
    if abs(z) < 1e-5:
        return 1/6 - z / 120 + z**2 / 5040
    elif z > 0:
        return (np.sqrt(z) - np.sin(np.sqrt(z))) / (z**1.5)
    else:
        return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / ((-z)**1.5)

def two_body_dynamics(t, y, mu):
    r = y[:3]
    v = y[3:]
    r_norm = np.linalg.norm(r)
    a = -mu * r / r_norm**3
    return np.concatenate((v, a))

def propagate_orbit(r0, v0, t_span, mu, num_points=1000):
    y0 = np.concatenate((r0, v0))
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    sol = solve_ivp(two_body_dynamics, t_span, y0, args=(mu,), t_eval=t_eval, rtol=1e-9)
    return sol.t, sol.y[:3, :].T



if __name__ == "__main__":

    spice.furnsh("E:\\ISAE Supaero\\Siddy\\Rossetta Project\\project_root\\kernels\\naif0012.tls")
    spice.furnsh("E:\\ISAE Supaero\\Siddy\\Rossetta Project\\project_root\\kernels\\de430.bsp")

    # print(spice.kdata(0, "spk"))  # See first SPK kernel loaded

    et_launch = spice.utc2et("2004-03-02")
    et_flyby1 = spice.utc2et("2005-03-04")

    state_launch, _ = spice.spkezr("EARTH", et_launch, "ECLIPJ2000", "NONE", "SUN")
    state_flyby1, _ = spice.spkezr("EARTH", et_flyby1, "ECLIPJ2000", "NONE", "SUN")

    v1, v2 = lambert_problem(state_launch[:3], state_flyby1[:3], et_flyby1 - et_launch, mu)
    print("Initial velocity vector (v1):", v1)
    print("Final velocity vector (v2):", v2)

    t0 = et_launch
    tf = et_flyby1
    duration = tf - t0

    # Lambert path
    t_array, lambert_positions = propagate_orbit(state_launch[:3], v1, (et_launch, et_flyby1), mu)