"""Plumbing around the fits the notebooks write out: running emcee, summing up a posterior, periodograms.

The models, likelihoods and formulas are not here. Notebook 4 writes the circular orbit, its log
probability and the mass from K in the open, and tess_02 writes the joint transit and velocity
model. This module runs emcee for them once they have been shown, and does the bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import emcee
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

# emcee's own test for a settled chain: it ran at least this many autocorrelation times.
AUTOCORR_TIMES = 50
WALKERS = 32
START_SPREAD = 1e-3  # how far the walkers start from the starting point, in each parameter's own units
PERIODOGRAM_POINTS = 5000
EARTH_CORE_FRACTION = 0.325  # the share of Earth's mass in its iron core (Zeng et al. 2016)
# The survey's transit times are uncertain by minutes, so a velocity this close to a transit counts as in it.
IN_TRANSIT_MARGIN_HOURS = 0.25


class Estimate(NamedTuple):
    """The middle of a posterior and how far its 68% range reaches below and above it."""

    median: float
    minus: float
    plus: float


def estimate(values) -> Estimate:
    """The median of `values` and the distances to its 16th and 84th percentiles (the middle 68%)."""
    low, middle, high = np.percentile(values, [16, 50, 84])
    return Estimate(median=middle, minus=middle - low, plus=high - middle)


@dataclass
class Run:
    """One emcee run: its samples (one column per parameter) and whether the chain settled."""

    samples: pd.DataFrame
    converged: bool
    autocorr_steps: float  # the longest autocorrelation time of any parameter, in steps
    steps: int


def sample(
    log_probability,
    start,
    names: list[str],
    *,
    args: tuple = (),
    steps: int = 4000,
    seed: int = 1,
) -> Run:
    """Run emcee the way notebook 4's first fit does, and return the samples after burn-in.

    WALKERS walkers start in a small ball (START_SPREAD) around `start`; the first quarter of the
    steps is dropped as burn-in. `seed` fixes both the starting ball and emcee's own random
    numbers, so the same call gives the same samples every time. Prints the settings it ran with.
    """
    rng = np.random.default_rng(seed)
    walker_starts = start + START_SPREAD * rng.standard_normal((WALKERS, len(start)))
    np.random.seed(seed)  # emcee takes its random numbers from numpy's, as it is built
    sampler = emcee.EnsembleSampler(WALKERS, len(start), log_probability, args=args)
    sampler.run_mcmc(walker_starts, steps, progress=False)
    burn_in = steps // 4
    print(f"emcee: {WALKERS} walkers, {steps} steps each; the first {burn_in} are dropped as burn-in.")
    return summarise(sampler, names, burn_in=burn_in)


def summarise(sampler, names: list[str], burn_in: int, thin: int = 1) -> Run:
    """The samples of a finished emcee run after `burn_in` steps, and whether it settled.

    emcee warns, without stopping, when the chain is shorter than 50 autocorrelation times.
    """
    autocorr_steps = float(np.nanmax(sampler.get_autocorr_time(quiet=True)))
    steps = sampler.iteration
    chain = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    return Run(
        samples=pd.DataFrame(chain, columns=names),
        converged=steps > AUTOCORR_TIMES * autocorr_steps,
        autocorr_steps=autocorr_steps,
        steps=steps,
    )


def hours_from_transit(times_bjd, period_days: float, transit_bjd: float):
    """Hours from the nearest mid-transit, for each time (days)."""
    orbits_since = (np.asarray(times_bjd, float) - transit_bjd) / period_days
    phase = (orbits_since + 0.5) % 1 - 0.5  # -0.5 to 0.5, with the transit at 0
    return phase * period_days * 24


def in_transit(
    times_bjd,
    period_days: float,
    transit_bjd: float,
    transit_hours: float,
    *,
    margin_hours: float = IN_TRANSIT_MARGIN_HOURS,
):
    """True for the times inside a transit, with `margin_hours` added either side."""
    return np.abs(hours_from_transit(times_bjd, period_days, transit_bjd)) < transit_hours / 2 + margin_hours


def periodogram(times, values, *, shortest: float, longest: float):
    """Lomb-Scargle power at 5,000 periods from `shortest` to `longest` (in the unit of `times`).

    The same astropy calls notebook 2 writes out. Returns (periods, power).
    """
    frequencies = np.linspace(1 / longest, 1 / shortest, PERIODOGRAM_POINTS)
    power = LombScargle(np.asarray(times, float), np.asarray(values, float)).power(frequencies)
    return 1 / frequencies, power


def rocky_radius(mass_earth, core_fraction: float = EARTH_CORE_FRACTION):
    """The radius (Earth radii) of a planet of rock and iron, for its mass (Earth masses).

    Zeng, Sasselov and Jacobsen 2016, ApJ 819, 127, as notebook 4 writes it out: R = (1.07 - 0.21
    x core fraction) x M^(1/3.7), fitted for about 1 to 8 Earth masses. Used here for the lines on
    the mass-radius plot.
    """
    return (1.07 - 0.21 * core_fraction) * np.asarray(mass_earth, float) ** (1 / 3.7)
