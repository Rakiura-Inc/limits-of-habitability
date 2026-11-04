"""Physical constants and time offsets shared by the fits, in SI units unless named otherwise."""

G = 6.674e-11  # gravitational constant, m^3 kg^-1 s^-2
M_SUN = 1.989e30  # kg
R_SUN = 6.957e8  # m
M_EARTH = 5.972e24  # kg
R_EARTH = 6.371e6  # m
M_JUP = 1.898e27  # kg
DAY = 86400.0  # seconds in a day

BTJD = 2457000.0  # TESS time (BTJD) is the barycentric Julian date minus this
TWO_MINUTES = 120 / 86400  # one TESS 2-minute exposure, in days
