"""The arithmetic and the fit, checked against numbers you can work out by hand."""
import numpy as np

import habtrain as ht


def test_wobble_matches_earth_and_jupiter():
    # Earth moves the Sun by about 9 cm/s; Jupiter by about 12.5 m/s.
    assert abs(ht.fit.wobble(1, 365.25, 1) - 0.0895) < 0.002
    assert abs(ht.fit.wobble(317.8, 4332.6, 1) - 12.47) < 0.1


def test_mass_from_wobble_inverts_wobble():
    k = ht.fit.wobble(10, 4.0, 0.9)
    assert abs(ht.fit.mass_from_wobble(k, 4.0, 0.9) - 10) < 1e-6


def test_fit_recovers_a_planet_we_made():
    t, v, err = ht.fit.fake_velocities(10.0, 7.0, n=60, error=1.0, jitter=1.0, seed=3)
    f = ht.fit.fit_circular(t, v, err, P=7.0, T0=0.0, steps=2000)
    k, lo, hi = f.value("K")
    assert abs(k - 10.0) < 3 * max(lo, hi)


def test_in_transit_marks_only_the_transit():
    t = np.array([0.0, 0.02, 0.5, 1.0])     # days; the transit is at 0 and 1 for P = 1
    m = ht.fit.in_transit(t, P=1.0, T0=0.0, duration_hours=2.0, margin_hours=0.0)
    assert m.tolist() == [True, True, False, True]
