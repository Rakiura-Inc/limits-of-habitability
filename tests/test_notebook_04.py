"""Notebook 4's own model, log probability and fits, on planets we made, so the answer is known."""

import numpy as np
import pandas as pd
import pytest

import habtrain as ht

from .notebook_code import definitions


@pytest.fixture(scope="module")
def nb04():
    return definitions("04_weigh_a_planet.ipynb")


def test_circular_orbit_passes_through_zero_at_transit_and_peaks_a_quarter_later(nb04):
    velocities = nb04["circular_orbit"](np.array([0.0, 2.5, 7.5, 10.0]), 3.0, 10.0, 0.0)
    assert velocities == pytest.approx([0.0, -3.0, 3.0, 0.0], abs=1e-12)


def test_made_velocities_are_repeatable_and_in_date_order(nb04):
    times, velocities, errors = nb04["make_velocities"](5.0, 3.0)
    times_again, velocities_again, _ = nb04["make_velocities"](5.0, 3.0)
    assert np.array_equal(velocities, velocities_again)
    assert np.array_equal(times, times_again)
    assert np.all(np.diff(times) >= 0)
    assert np.all(errors == 2.0)


def test_log_probability_refuses_a_negative_wobble_or_a_wild_jitter(nb04):
    times, velocities, errors = nb04["make_velocities"](10.0, 7.0)
    data = (times, velocities, errors, 7.0, 0.0)
    assert np.isfinite(nb04["log_probability"]([10.0, 0.0, np.log(2.0)], *data))
    assert nb04["log_probability"]([-1.0, 0.0, 0.0], *data) == -np.inf
    assert nb04["log_probability"]([10.0, 0.0, 6.0], *data) == -np.inf


def test_part_1_fit_recovers_the_planet_we_made(nb04, capsys):
    times, velocities, errors = nb04["make_velocities"](10.0, 7.0, n_nights=60)
    start = nb04["least_squares_start"](times, velocities, 7.0, 0.0)
    assert start[0] == pytest.approx(10.0, abs=2.0)
    data = (times, velocities, errors, 7.0, 0.0)
    run = ht.fit.sample(nb04["log_probability"], start, ["K", "offset", "log jitter"], args=data)
    K = ht.fit.estimate(run.samples.K)
    assert abs(K.median - 10.0) < 3 * max(K.minus, K.plus)
    capsys.readouterr()


def test_part_2_fit_gives_each_telescope_its_own_offset(nb04):
    times, velocities, errors = nb04["make_velocities"](8.0, 5.0, n_nights=80, seed=11)
    telescope = np.where(np.arange(80) % 2, "APF", "HIRES")
    shifted = velocities + np.where(telescope == "APF", 20.0, -15.0)
    table = pd.DataFrame({"BJD": times, "RV": shifted, "eRV": errors, "Inst": telescope})
    fit = nb04["fit_circular_orbit"](table, 5.0, 0.0, steps=1500)
    assert list(fit.samples.columns) == [
        "K (m/s)",
        "offset APF (m/s)",
        "offset HIRES (m/s)",
        "jitter APF (m/s)",
        "jitter HIRES (m/s)",
    ]
    assert ht.fit.estimate(fit.samples["offset APF (m/s)"]).median == pytest.approx(20.0, abs=1.5)
    assert ht.fit.estimate(fit.samples["offset HIRES (m/s)"]).median == pytest.approx(-15.0, abs=1.5)
    assert ht.fit.estimate(fit.samples["K (m/s)"]).median == pytest.approx(8.0, abs=1.5)
    assert (fit.samples["jitter APF (m/s)"] > 0).all()  # reported in m/s, not as its log


def test_mass_from_wobble_gives_earth_and_jupiter(nb04):
    # Earth moves the Sun by about 9 cm/s; Jupiter by about 12.5 m/s.
    assert nb04["mass_from_wobble"](0.0895, 365.25, 1.0) == pytest.approx(1.0, rel=0.02)
    assert nb04["mass_from_wobble"](12.47, 4332.6, 1.0) == pytest.approx(317.8, rel=0.01)


def test_hours_from_transit_wraps_to_the_nearest_transit(nb04):
    hours = nb04["hours_from_transit"](np.array([0.0, 0.25, 0.75, 1.0, 10.1]), 1.0, 0.0)
    assert hours == pytest.approx([0.0, 6.0, -6.0, 0.0, 2.4])
    assert np.array_equal(hours, ht.fit.hours_from_transit([0.0, 0.25, 0.75, 1.0, 10.1], 1.0, 0.0))


def test_rocky_radius_gives_earth_for_an_earth_like_mix(nb04):
    assert nb04["rocky_radius"](1.0, 0.325) == pytest.approx(1.0, abs=0.01)
    assert nb04["rocky_radius"](5.0, 0.325) == ht.fit.rocky_radius(5.0)
