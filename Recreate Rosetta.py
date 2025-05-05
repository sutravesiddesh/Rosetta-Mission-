import spiceypy as spice

def lambert_problem(r1, r2, dt, mu):
    """
    Solve the Lambert problem using the universal variable method.
    :param r1: Initial position vector (numpy array).
    :param r2: Final position vector (numpy array).
    :param dt: Time of flight (seconds).
    :param mu: Gravitational parameter (km^3/s^2).
    :return: Tuple of initial and final velocity vectors (v1, v2).
    """
    # Placeholder for the actual implementation
    # This function should return the calculated velocities based on the Lambert problem

    


    return None, None





spice.furnsh("E:\\ISAE Supaero\\Siddy\\Rossetta Project\\project_root\\kernels\\naif0012.tls")
spice.furnsh("E:\\ISAE Supaero\\Siddy\\Rossetta Project\\project_root\\kernels\\de430.bsp")

print(spice.kdata(0, "spk"))  # See first SPK kernel loaded

et_launch = spice.utc2et("2004-03-02")
et_flyby1 = spice.utc2et("2005-03-04")

state_launch, _ = spice.spkezr("EARTH", et_launch, "ECLIPJ2000", "NONE", "SUN")
state_flyby1, _ = spice.spkezr("EARTH", et_flyby1, "ECLIPJ2000", "NONE", "SUN")

print(state_launch)
print(state_flyby1)