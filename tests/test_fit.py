"""habtrain.fit: summing up a posterior, running emcee, periodograms and the arithmetic around them."""

import logging

import numpy as np
import pytest

import habtrain as ht


def test_estimate_is_the_median_and_its_68_percent_range():
    estimate = ht.fit.estimate(np.arange(101.0))
    assert estimate == pytest.approx((50.0, 34.0, 34.0))
    assert (estimate.median, estimate.minus, estimate.plus) == pytest.approx((50.0, 34.0, 34.0))


def log_probability_of_a_line(params, x, y):
    slope, intercept = params
    return -0.5 * np.sum((y - slope * x - intercept) ** 2)


def test_sample_recovers_a_line_and_repeats_with_the_same_seed(capsys):
    x = np.linspace(0, 10, 50)
    y = 2.0 * x + 1.0 + np.random.default_rng(3).normal(0, 1.0, x.size)
    first = ht.fit.sample(log_probability_of_a_line, np.array([1.5, 0.0]), ["slope", "intercept"], args=(x, y))
    again = ht.fit.sample(log_probability_of_a_line, np.array([1.5, 0.0]), ["slope", "intercept"], args=(x, y))
    assert ht.fit.estimate(first.samples.slope).median == pytest.approx(2.0, abs=0.1)
    assert first.samples.equals(again.samples)
    assert first.samples.shape == (4000 * 3 // 4 * ht.fit.WALKERS, 2)
    assert first.converged
    assert "32 walkers, 4000 steps each; the first 1000 are dropped" in capsys.readouterr().out


def test_a_short_run_is_not_converged_and_emcee_says_why(caplog):
    x = np.linspace(0, 10, 50)
    with caplog.at_level(logging.WARNING):
        run = ht.fit.sample(log_probability_of_a_line, np.array([2.0, 1.0]), ["a", "b"], args=(x, 2 * x), steps=60)
    assert not run.converged
    assert run.steps == 60
    assert "autocorrelation time" in caplog.text


def test_hours_from_transit_wraps_to_the_nearest_transit():
    hours = ht.fit.hours_from_transit([0.0, 0.25, 0.75, 1.0, 10.1], 1.0, 0.0)
    assert hours == pytest.approx([0.0, 6.0, -6.0, 0.0, 2.4])


def test_in_transit_marks_only_the_transit():
    times = np.array([0.0, 0.02, 0.5, 1.0])  # days; the transits are at 0 and 1 for a 1-day period
    inside = ht.fit.in_transit(times, 1.0, 0.0, 2.0, margin_hours=0.0)
    assert inside.tolist() == [True, True, False, True]


def test_periodogram_peaks_at_the_period_we_put_in():
    rng = np.random.default_rng(5)
    times = np.sort(rng.uniform(0, 200, 300))
    values = np.sin(2 * np.pi * times / 17.0) + rng.normal(0, 0.3, times.size)
    periods, power = ht.fit.periodogram(times, values, shortest=2, longest=100)
    assert len(periods) == ht.fit.PERIODOGRAM_POINTS
    assert periods[np.argmax(power)] == pytest.approx(17.0, rel=0.02)


def test_rocky_radius_gives_earth_for_earth():
    assert ht.fit.rocky_radius(1.0) == pytest.approx(1.0, abs=0.01)
    assert ht.fit.rocky_radius(5.0, core_fraction=0.0) > ht.fit.rocky_radius(5.0)
