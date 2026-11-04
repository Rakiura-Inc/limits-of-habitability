"""A circular orbit fitted to radial velocities, and the arithmetic around it.

The model is deliberately the simplest one that recovers the published answers for the systems
offered in notebook 4: a circular orbit whose period and transit time are fixed from the transit
data, plus one offset and one jitter per telescope.

    velocity(t) = -K sin(2 pi (t - T0) / P) + offset[telescope]

At the transit time T0 the planet crosses in front of its star, and the star's velocity along
your line of sight passes through zero. Jitter is extra scatter added in quadrature to each
point's error bar, to stand for everything the model leaves out, the star's own activity included.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import emcee
import numpy as np

from .constants import DAY, M_EARTH, M_SUN, G

K_MAX = 2000.0  # m/s: the fit only looks for wobbles smaller than this
LOG_JITTER_MIN, LOG_JITTER_MAX = -5, 5  # jitter between e^-5 and e^5 m/s
START_JITTER = 2.0  # m/s, where every telescope's jitter starts
START_SPREAD = 1e-3  # how far the walkers start from the least-squares answer
AUTOCORR_TIMES = 50  # a chain counts as settled when it is longer than this many autocorrelation times
PERIODOGRAM_POINTS = 5000


def circular(t, K, P, T0):
    """The star's velocity (m/s) at times t for a circular orbit."""
    return -K * np.sin(2 * np.pi * (np.asarray(t) - T0) / P)


def wobble(planet_mass_earth: float, period_days: float, star_mass_sun: float, e: float = 0.0) -> float:
    """The semi-amplitude K in m/s that a planet should cause (planet mass much smaller than the star)."""
    m = planet_mass_earth * M_EARTH
    return (2 * np.pi * G / (period_days * DAY)) ** (1 / 3) * m / (star_mass_sun * M_SUN) ** (2 / 3) / np.sqrt(1 - e**2)


def mass_from_wobble(K: float, period_days: float, star_mass_sun: float, e: float = 0.0) -> float:
    """The planet's mass in Earth masses from K (for a transiting planet, sin i is close to 1)."""
    return (
        K
        * np.sqrt(1 - e**2)
        * (period_days * DAY / (2 * np.pi * G)) ** (1 / 3)
        * (star_mass_sun * M_SUN) ** (2 / 3)
        / M_EARTH
    )


def fake_velocities(
    K: float,
    P: float,
    n: int = 40,
    span_days: float = 1000.0,
    error: float = 2.0,
    jitter: float = 2.0,
    T0: float = 0.0,
    seed: int = 1,
):
    """Velocities of a planet you choose, observed at random times by one telescope.

    Returns (t, v, err): times in days, velocities in m/s, each point's error bar in m/s. The
    scatter is the error bar and the jitter added together, as a real spectrograph's would be.
    """
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, span_days, n))
    err = np.full(n, float(error))
    v = circular(t, K, P, T0) + rng.normal(0, np.hypot(error, jitter), n)
    return t, v, err


def telescope_index(telescopes) -> tuple[list[str], np.ndarray]:
    """The telescope names in alphabetical order, and for each point the position of its telescope there."""
    names = sorted(set(np.asarray(telescopes).tolist()))
    position = {name: i for i, name in enumerate(names)}
    return names, np.array([position[name] for name in np.asarray(telescopes).tolist()])


def longest_autocorr(sampler, ignore_nan: bool = False) -> float:
    """The longest autocorrelation time of any parameter, in steps, or nan when it cannot be estimated."""
    try:
        times = sampler.get_autocorr_time(tol=0)
    except Exception:  # noqa: BLE001 - an estimate that fails just means "unknown"
        return float("nan")
    return float(np.nanmax(times) if ignore_nan else np.max(times))


def settled(autocorr_steps: float, steps: int) -> bool:
    """True when the chain ran for more than AUTOCORR_TIMES autocorrelation times."""
    return bool(np.isfinite(autocorr_steps) and steps > AUTOCORR_TIMES * autocorr_steps)


@dataclass
class Fit:
    """What a fit gives back. `chain` holds the samples: one column per entry in `names`."""

    names: list
    chain: np.ndarray
    telescopes: list
    converged: bool
    autocorr_steps: float
    steps: int
    summary: dict = field(default_factory=dict)

    def value(self, name: str):
        """(median, minus, plus) for one parameter: the middle of the posterior and its 68% range."""
        i = self.names.index(name)
        lo, mid, hi = np.percentile(self.chain[:, i], [16, 50, 84])
        return mid, mid - lo, hi - mid


def _least_squares_start(phase, v, idx, n_telescopes: int) -> np.ndarray:
    """A straight least-squares answer for K and the offsets, so the walkers do not have to search."""
    columns = [np.sin(phase), np.cos(phase)] + [(idx == k).astype(float) for k in range(n_telescopes)]
    coef = np.linalg.lstsq(np.column_stack(columns), v, rcond=None)[0]
    K, offsets = max(abs(coef[0]), 0.5), coef[2:]
    return np.concatenate([[K], offsets, np.log(np.full(n_telescopes, START_JITTER))])


def fit_circular(
    t, v, err, P: float, T0: float, telescopes=None, steps: int = 4000, walkers: int = 32, seed: int = 1
) -> Fit:
    """Fit K, one offset and one jitter per telescope with emcee. P and T0 stay fixed.

    `telescopes` is a list naming the telescope of each point; leave it out for one telescope.
    The parameters are K, then one offset per telescope, then one log jitter per telescope.
    """
    t, v, err = (np.asarray(x, float) for x in (t, v, err))
    names_t, idx = telescope_index(telescopes if telescopes is not None else ["telescope"] * len(t))
    nt = len(names_t)
    phase = 2 * np.pi * (t - T0) / P

    def log_prob(theta):
        K, offsets, log_jitter = theta[0], theta[1 : 1 + nt], theta[1 + nt :]
        if not (0 < K < K_MAX) or np.any(log_jitter < LOG_JITTER_MIN) or np.any(log_jitter > LOG_JITTER_MAX):
            return -np.inf
        variance = err**2 + np.exp(2 * log_jitter[idx])
        residual = v - (-K * np.sin(phase) + offsets[idx])
        return -0.5 * np.sum(residual**2 / variance + np.log(2 * np.pi * variance))

    start = _least_squares_start(phase, v, idx, nt)
    rng = np.random.default_rng(seed)
    p0 = start + START_SPREAD * rng.standard_normal((walkers, len(start)))
    sampler = emcee.EnsembleSampler(walkers, len(start), log_prob)
    sampler.run_mcmc(p0, steps, progress=False)
    tau = longest_autocorr(sampler)
    chain = sampler.get_chain(discard=steps // 4, flat=True).copy()
    chain[:, 1 + nt :] = np.exp(chain[:, 1 + nt :])  # report jitter in m/s, not its log
    names = ["K"] + [f"offset {x}" for x in names_t] + [f"jitter {x}" for x in names_t]
    return Fit(
        names=names, chain=chain, telescopes=names_t, converged=settled(tau, steps), autocorr_steps=tau, steps=steps
    )


def hours_from_transit(t, P: float, T0: float):
    """Hours from the nearest mid-transit, for each time t (days)."""
    return ((((np.asarray(t, float) - T0) / P) + 0.5) % 1 - 0.5) * P * 24


def in_transit(t, P: float, T0: float, duration_hours: float, margin_hours: float = 0.25):
    """True for the times that fall inside a transit (plus a small margin either side)."""
    if not np.isfinite(duration_hours):
        return np.zeros(len(np.asarray(t)), bool)
    return np.abs(hours_from_transit(t, P, T0)) < duration_hours / 2 + margin_hours


def _frequencies(shortest: float, longest: float, n: int = PERIODOGRAM_POINTS):
    return np.linspace(1 / longest, 1 / shortest, n)


def periodogram(t, y, shortest: float, longest: float, n: int = PERIODOGRAM_POINTS):
    """Lomb-Scargle power at periods from `shortest` to `longest` (same unit as t)."""
    from astropy.timeseries import LombScargle

    freq = _frequencies(shortest, longest, n)
    power = LombScargle(np.asarray(t, float), np.asarray(y, float)).power(freq)
    return 1 / freq, power


def false_alarm(t, y, shortest: float, longest: float) -> float:
    """The chance that noise alone would give a periodogram peak as tall as the tallest one here."""
    from astropy.timeseries import LombScargle

    ls = LombScargle(np.asarray(t, float), np.asarray(y, float))
    tallest = ls.power(_frequencies(shortest, longest)).max()
    return float(ls.false_alarm_probability(tallest, minimum_frequency=1 / longest, maximum_frequency=1 / shortest))


def rocky_radius(mass_earth, core_fraction: float = 0.325):
    """The radius (Earth radii) of a planet made of rock and iron, for its mass (Earth masses).

    Zeng, Sasselov and Jacobsen 2016, ApJ 819, 127: R = (1.07 - 0.21 x core fraction) x M^(1/3.7), fitted
    for about 1 to 8 Earth masses. Earth's core fraction is about 0.325; a core fraction of 0 (no iron at
    all) gives the biggest a rocky planet of that mass can be. Above 8 Earth masses it is a rough guide.
    """
    return (1.07 - 0.21 * core_fraction) * np.asarray(mass_earth, float) ** (1 / 3.7)


def sigma_apart(a: float, ea: float, b: float, eb: float) -> float:
    """How many combined standard deviations apart two measurements are."""
    return abs(a - b) / np.hypot(ea, eb)
