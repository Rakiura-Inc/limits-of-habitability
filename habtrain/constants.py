"""Physical constants, unit ratios and time offsets the notebooks share, each with its source.

SI units unless the name says otherwise.
"""

# CODATA 2018 (G), IAU 2015 nominal (the Sun's radius), NASA planetary fact sheets (the rest).
G = 6.674e-11  # gravitational constant, m^3 kg^-1 s^-2
M_SUN = 1.989e30  # kg
R_SUN = 6.957e8  # m
M_EARTH = 5.972e24  # kg
R_EARTH = 6.371e6  # m, Earth's mean radius
DAY = 86400.0  # seconds in a day
DAYS_PER_YEAR = 365.25  # a Julian year

# Ratios the notebooks use to change units (NASA planetary fact sheet; the Sun's radius, IAU 2015).
EARTH_INSOLATION_W_M2 = 1361  # the starlight Earth gets
EARTH_RADII_PER_SUN_RADIUS = 109.08
EARTH_RADII_PER_JUPITER_RADIUS = 11.2
EARTH_MASSES_PER_JUPITER_MASS = 317.8
EARTH_DENSITY_G_CM3 = 5.514

BTJD = 2457000.0  # TESS time (BTJD) is the barycentric Julian date minus this
TWO_MINUTES_DAYS = 120 / DAY  # one TESS 2-minute exposure
