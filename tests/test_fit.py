"""The arithmetic and the fit, checked against numbers you can work out by hand."""

import numpy as np
import pytest

import habtrain as ht


def test_wobble_matches_earth_and_jupiter():
    # Earth moves the Sun by about 9 cm/s; Jupiter by about 12.5 m/s.
    assert ht.fit.wobble(1, 365.25, 1) == pytest.approx(0.0895, abs=0.002)
    assert ht.fit.wobble(317.8, 4332.6, 1) == pytest.approx(12.47, abs=0.1)


def test_an_eccentric_orbit_wobbles_more():
    assert ht.fit.wobble(10, 30, 1, e=0.6) == pytest.approx(ht.fit.wobble(10, 30, 1) / 0.8)


@pytest.mark.parametrize("e", [0.0, 0.5])
def test_mass_from_wobble_inverts_wobble(e):
    k = ht.fit.wobble(10, 4.0, 0.9, e=e)
    assert ht.fit.mass_from_wobble(k, 4.0, 0.9, e=e) == pytest.approx(10)


def test_circular_orbit_passes_through_zero_at_transit_and_peaks_a_quarter_later():
    v = ht.fit.circular([0.0, 2.5, 7.5, 10.0], K=3.0, P=10.0, T0=0.0)
    assert v == pytest.approx([0.0, -3.0, 3.0, 0.0], abs=1e-12)


def test_fake_velocities_are_repeatable_and_sorted():
    t1, v1, e1 = ht.fit.fake_velocities(5.0, 3.0, n=30, error=1.5, seed=7)
    t2, v2, _ = ht.fit.fake_velocities(5.0, 3.0, n=30, error=1.5, seed=7)
    assert np.array_equal(v1, v2)
    assert np.array_equal(t1, t2)
    assert np.all(np.diff(t1) >= 0)
    assert np.all(e1 == 1.5)


def test_telescope_index_numbers_telescopes_alphabetically():
    names, idx = ht.fit.telescope_index(["HIRES", "APF", "HIRES"])
    assert names == ["APF", "HIRES"]
    assert idx.tolist() == [1, 0, 1]


def test_settled_needs_fifty_autocorrelation_times():
    assert ht.fit.settled(10.0, 501)
    assert not ht.fit.settled(10.0, 500)
    assert not ht.fit.settled(float("nan"), 10_000)


def test_longest_autocorr_is_nan_when_it_cannot_be_estimated():
    class Broken:
        def get_autocorr_time(self, tol):
            raise ValueError

    assert np.isnan(ht.fit.longest_autocorr(Broken()))


def test_longest_autocorr_can_skip_nan():
    class Sampler:
        def get_autocorr_time(self, tol):
            return np.array([3.0, np.nan, 5.0])

    assert np.isnan(ht.fit.longest_autocorr(Sampler()))
    assert ht.fit.longest_autocorr(Sampler(), ignore_nan=True) == 5.0


def test_fit_recovers_a_planet_we_made():
    t, v, err = ht.fit.fake_velocities(10.0, 7.0, n=60, error=1.0, jitter=1.0, seed=3)
    f = ht.fit.fit_circular(t, v, err, P=7.0, T0=0.0, steps=2000)
    k, lo, hi = f.value("K")
    assert abs(k - 10.0) < 3 * max(lo, hi)
    assert f.names == ["K", "offset telescope", "jitter telescope"]
    assert f.chain.shape == (2000 * 3 // 4 * 32, 3)
    assert np.all(f.chain[:, 2] > 0)  # jitter is reported in m/s, not as its log


def test_fit_gives_each_telescope_its_own_offset():
    t, v, err = ht.fit.fake_velocities(8.0, 5.0, n=80, error=1.0, jitter=0.5, seed=11)
    telescopes = np.where(np.arange(80) % 2, "APF", "HIRES")
    v = v + np.where(telescopes == "APF", 20.0, -15.0)
    f = ht.fit.fit_circular(t, v, err, P=5.0, T0=0.0, telescopes=telescopes, steps=1500)
    assert f.telescopes == ["APF", "HIRES"]
    assert f.value("offset APF")[0] == pytest.approx(20.0, abs=1.5)
    assert f.value("offset HIRES")[0] == pytest.approx(-15.0, abs=1.5)
    assert f.value("K")[0] == pytest.approx(8.0, abs=1.5)


def test_fit_value_is_the_median_and_68_percent_range():
    f = ht.fit.Fit(
        names=["x"], chain=np.arange(101.0)[:, None], telescopes=[], converged=True, autocorr_steps=1.0, steps=1
    )
    assert f.value("x") == pytest.approx((50.0, 34.0, 34.0))


def test_hours_from_transit_wraps_to_the_nearest_transit():
    hours = ht.fit.hours_from_transit([0.0, 0.25, 0.75, 1.0, 10.1], P=1.0, T0=0.0)
    assert hours == pytest.approx([0.0, 6.0, -6.0, 0.0, 2.4])


def test_in_transit_marks_only_the_transit():
    t = np.array([0.0, 0.02, 0.5, 1.0])  # days; the transit is at 0 and 1 for P = 1
    m = ht.fit.in_transit(t, P=1.0, T0=0.0, duration_hours=2.0, margin_hours=0.0)
    assert m.tolist() == [True, True, False, True]


def test_in_transit_with_no_known_duration_marks_nothing():
    assert not ht.fit.in_transit([0.0, 1.0], P=1.0, T0=0.0, duration_hours=float("nan")).any()


def test_periodogram_peaks_at_the_period_we_put_in():
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(0, 200, 300))
    y = np.sin(2 * np.pi * t / 17.0) + rng.normal(0, 0.3, t.size)
    periods, power = ht.fit.periodogram(t, y, 2, 100)
    assert periods[np.argmax(power)] == pytest.approx(17.0, rel=0.02)
    assert ht.fit.false_alarm(t, y, 2, 100) < 1e-10


def test_false_alarm_is_large_for_noise():
    rng = np.random.default_rng(6)
    t = np.sort(rng.uniform(0, 200, 100))
    assert ht.fit.false_alarm(t, rng.normal(0, 1, t.size), 2, 100) > 0.01


def test_rocky_radius_gives_earth_for_earth():
    assert ht.fit.rocky_radius(1.0) == pytest.approx(1.0, abs=0.01)
    assert ht.fit.rocky_radius(5.0, core_fraction=0.0) > ht.fit.rocky_radius(5.0)


def test_sigma_apart_combines_both_uncertainties():
    assert ht.fit.sigma_apart(10.0, 3.0, 5.0, 4.0) == pytest.approx(1.0)
