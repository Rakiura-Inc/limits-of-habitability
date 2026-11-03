"""One model fitted to a planet's transits and its star's velocities at once.

The transit side uses batman (a transit light-curve model with limb darkening); the velocity side
is the circular orbit from the bootcamp. Both share one orbit: the same period and transit time.
The star's mass and radius enter as priors from published values, which is how the transit
duration becomes a size for the orbit and the depth becomes a radius in Earth units.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from .constants import BTJD, DAY, M_SUN, R_EARTH, R_SUN, TWO_MINUTES, G
from .fit import in_transit, longest_autocorr, mass_from_wobble, settled, telescope_index
from .lightcurve import FLATTEN_WINDOW

# The parameters every fit has, in order. After them come one velocity offset and then one
# log jitter per telescope.
SHARED = ("t0", "P", "ratio", "impact", "Mstar", "Rstar", "q1", "q2", "flux offset", "log jitter lc", "K")
N_SHARED = len(SHARED)

# What the fit allows. Outside these limits an orbit has zero probability.
MAX_RATIO = 0.3  # planet radius / star radius
MAX_K = 500.0  # m/s
MIN_STAR = 0.1  # smallest star mass (solar masses) and radius (solar radii)
T0_WINDOW = 0.1  # days the transit time may move
P_WINDOW = 0.01  # days the period may move
LOG_JITTER_LC = (-10, 3)
MAX_ABS_LOG_JITTER_RV = 5

TRANSIT_SUPERSAMPLE = 3  # model points per 2-minute exposure
NEAR_TRANSIT = 1.5  # keep light-curve points within this many transit durations of mid-transit


@dataclass
class Prepared:
    """Everything the joint fit needs for one planet, in BTJD (BJD minus 2457000) throughout."""

    planet: str
    P: float
    T0: float
    t_lc: np.ndarray  # light-curve times near transit (BTJD)
    flux: np.ndarray  # brightness minus 1, in parts per thousand
    flux_err: np.ndarray
    t_rv: np.ndarray  # velocity times (BTJD)
    rv: np.ndarray
    rv_err: np.ndarray
    telescope: np.ndarray
    sectors: list = field(default_factory=list)
    rv_in_transit: int = 0  # velocities left out because they were taken during a transit


def _unmasked(x) -> np.ndarray:
    """A plain float array of a light curve column's values, masked or not."""
    return np.array(getattr(x, "unmasked", x), dtype=float)


def _phase_days(t, T0: float, P: float):
    """Days from the nearest mid-transit."""
    return (t - T0 + 0.5 * P) % P - 0.5 * P


def prepare(planet: str, rvs, orbit: dict, duration_hours: float, max_sectors: int | None = 8) -> Prepared:
    """Download TESS data and the survey's velocities for one planet, ready for the joint fit.

    Takes up to `max_sectors` TESS 2-minute sectors (None for all), flattens each with its transits hidden,
    and keeps the data near transit. Collects the survey's velocities for the same star, leaving out any
    taken during a transit: the star's spin distorts those (the swing in notebook 4's Extra), and this
    model has no term for it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        import lightkurve as lk
    P, T0 = orbit["P"], orbit["T0"] - BTJD  # the survey's T0 is full BJD; TESS uses BTJD
    dur_days = duration_hours / 24
    found = lk.search_lightcurve(planet.split()[0], mission="TESS", author="SPOC", exptime=120)
    if max_sectors:
        found = found[:max_sectors]
    ts, fs, es, sectors = [], [], [], []
    for lc in found.download_all(flux_column="pdcsap_flux"):
        clean = lc.remove_nans().normalize()
        mask = clean.create_transit_mask(period=P, transit_time=T0, duration=2 * dur_days)
        flat = clean.flatten(window_length=FLATTEN_WINDOW, mask=mask).remove_outliers(sigma_upper=4, sigma_lower=20)
        t, f, e = _unmasked(flat.time.value), _unmasked(flat.flux.value), _unmasked(flat.flux_err.value)
        keep = (np.abs(_phase_days(t, T0, P)) < NEAR_TRANSIT * dur_days) & np.isfinite(f) & np.isfinite(e)
        ts.append(t[keep])
        fs.append((f[keep] - 1) * 1e3)
        es.append(e[keep] * 1e3)
        sectors.append(int(clean.meta.get("SECTOR", -1)))
    d = rvs[rvs.TKS == orbit["TKS"]]
    during = in_transit(d.BJD.to_numpy(), orbit["P"], orbit["T0"], duration_hours)
    d = d[~during]
    return Prepared(
        planet,
        P,
        T0,
        np.concatenate(ts),
        np.concatenate(fs),
        np.concatenate(es),
        d.BJD.to_numpy() - BTJD,
        d.RV.to_numpy(),
        d.eRV.to_numpy(),
        d.Inst.to_numpy(),
        sectors,
        int(during.sum()),
    )


@dataclass
class JointResult:
    """What the joint fit gives back.

    Each tuple is (median, plus, minus): the middle of the posterior and how far its 68% range
    reaches above and below.
    """

    radius: tuple  # in Earth radii
    mass: tuple  # in Earth masses
    K: tuple  # m/s
    impact: tuple  # impact parameter b: 0 crosses the middle of the star, 1 grazes its edge
    ratio: tuple  # planet radius / star radius
    autocorr_steps: float
    steps: int
    seconds: float
    chain: np.ndarray
    names: list
    converged: bool = False
    median: np.ndarray | None = None
    telescopes: list | None = None
    Tref: float = 0.0


def _summary(x):
    lo, mid, hi = np.percentile(x, [16, 50, 84])
    return (mid, hi - mid, mid - lo)


def _orbit_size(P, M, R):
    """The orbit's size (semi-major axis) in star radii, by Kepler's third law.

    P in days, star mass M and radius R in solar units.
    """
    return (G * M * M_SUN * (P * DAY) ** 2 / (4 * np.pi**2)) ** (1 / 3) / (R * R_SUN)


def _set_transit(params, t0, P, k, a, b, q1, q2) -> None:
    """Put one orbit into batman's parameters.

    q1 and q2 are the limb-darkening numbers of Kipping 2013, turned into batman's quadratic ones.
    """
    params.t0, params.per, params.rp, params.a = t0, P, k, a
    params.inc = np.degrees(np.arccos(min(b / a, 1.0)))
    params.u = [2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)]


def _transit_params(t0: float, P: float):
    """batman's parameters for a circular orbit. The size, inclination and limb darkening are placeholders."""
    import batman

    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a, params.inc = t0, P, 0.03, 10.0, 89.0
    params.ecc, params.w, params.limb_dark, params.u = 0.0, 90.0, "quadratic", [0.4, 0.2]
    return params


def _transit_model(params, t):
    """batman's model of the transit at times t, averaged over each 2-minute exposure."""
    import batman

    return batman.TransitModel(params, t, supersample_factor=TRANSIT_SUPERSAMPLE, exp_time=TWO_MINUTES)


def _start(Tref: float, P: float, star_mass: tuple, star_radius: tuple, n_telescopes: int):
    """Where the walkers start, and how widely they spread around it, for every parameter in order."""
    (Ms, eMs), (Rs, eRs) = star_mass, star_radius
    shared = {  # name: (start, spread)
        "t0": (Tref, 1e-3),
        "P": (P, 1e-6),
        "ratio": (0.03, 1e-3),  # a typical planet-to-star size ratio for these targets
        "impact": (0.3, 0.05),
        "Mstar": (Ms, eMs / 3),
        "Rstar": (Rs, eRs / 3),
        "q1": (0.3, 0.05),
        "q2": (0.3, 0.05),
        "flux offset": (0.0, 0.01),
        "log jitter lc": (-2.0, 0.1),
        "K": (5.0, 0.5),
    }
    per_telescope = [(0.0, 0.5)] * n_telescopes + [(1.0, 0.1)] * n_telescopes  # offsets (m/s), then log jitters
    value, spread = zip(*([shared[name] for name in SHARED] + per_telescope), strict=True)
    return np.array(value), np.array(spread)


def fit(
    prep: Prepared, star_mass: tuple, star_radius: tuple, steps: int = 20000, walkers: int = 48, seed: int = 1
) -> JointResult:
    """Fit transit and velocities together. star_mass and star_radius are (value, uncertainty), in solar units."""
    import emcee

    Ms, eMs = star_mass
    Rs, eRs = star_radius
    started = time.time()
    x, y, yerr = prep.t_lc, prep.flux, prep.flux_err
    names_t, idx = telescope_index(prep.telescope)
    nt = len(names_t)
    # A transit time in the middle of the TESS data, so the period and transit time are not tangled together.
    Tref = prep.T0 + np.round((np.median(x) - prep.T0) / prep.P) * prep.P
    params = _transit_params(Tref, prep.P)
    model = _transit_model(params, x)
    names = list(SHARED) + [f"offset {n}" for n in names_t] + [f"log jitter {n}" for n in names_t]

    def predict(p):
        t0, P, k, b, M, R, q1, q2, mean, _, K = p[:N_SHARED]
        _set_transit(params, t0, P, k, _orbit_size(P, M, R), b, q1, q2)
        lc = (model.light_curve(params) - 1) * 1e3 + mean
        rv = -K * np.sin(2 * np.pi * (prep.t_rv - t0) / P) + p[N_SHARED : N_SHARED + nt][idx]
        return lc, rv

    def allowed(p) -> bool:
        t0, P, k, b, M, R, q1, q2, _, log_jitter_lc, K = p[:N_SHARED]
        return bool(
            0 < k < MAX_RATIO
            and 0 <= b < 1 + k
            and 0 < q1 < 1
            and 0 < q2 < 1
            and 0 <= K < MAX_K
            and M > MIN_STAR
            and R > MIN_STAR
            and abs(t0 - Tref) < T0_WINDOW
            and abs(P - prep.P) < P_WINDOW
            and LOG_JITTER_LC[0] < log_jitter_lc < LOG_JITTER_LC[1]
            and np.all(np.abs(p[N_SHARED + nt :]) < MAX_ABS_LOG_JITTER_RV)
        )

    def log_prob(p):
        if not allowed(p):
            return -np.inf
        M, R, log_jitter_lc = p[4], p[5], p[9]
        lc, rv = predict(p)
        s2 = yerr**2 + np.exp(2 * log_jitter_lc)
        s2r = prep.rv_err**2 + np.exp(2 * p[N_SHARED + nt :][idx])
        transit = -0.5 * np.sum((y - lc) ** 2 / s2 + np.log(s2))
        velocities = -0.5 * np.sum((prep.rv - rv) ** 2 / s2r + np.log(s2r))
        mass_prior = -0.5 * ((M - Ms) / eMs) ** 2
        radius_prior = -0.5 * ((R - Rs) / eRs) ** 2
        return transit + velocities + mass_prior + radius_prior

    p0, scale = _start(Tref, prep.P, star_mass, star_radius, nt)
    rng = np.random.default_rng(seed)
    start = p0 + scale * rng.standard_normal((walkers, len(p0)))
    start[:, 3] = np.abs(start[:, 3])  # the impact parameter cannot be negative
    sampler = emcee.EnsembleSampler(walkers, len(p0), log_prob)
    # The parameters live on very different scales (a transit time near 2000 days, a period known to a
    # millionth of a day), which trips emcee's start check even though the walkers are independent.
    sampler.run_mcmc(start, steps, progress=False, skip_initial_state_check=True)
    tau = longest_autocorr(sampler, ignore_nan=True)
    ch = sampler.get_chain(discard=steps // 2, thin=10, flat=True)
    P, k, b, M, R, K = ch[:, 1], ch[:, 2], ch[:, 3], ch[:, 4], ch[:, 5], ch[:, 10]
    sini = np.sqrt(1 - np.clip(b / _orbit_size(P, M, R), 0, 1) ** 2)
    radius = k * R * R_SUN / R_EARTH
    mass = mass_from_wobble(K, P, M) / sini
    return JointResult(
        _summary(radius),
        _summary(mass),
        _summary(K),
        _summary(b),
        _summary(k),
        tau,
        steps,
        time.time() - started,
        ch,
        names,
        converged=settled(tau, steps),
        median=np.median(ch, axis=0),
        telescopes=names_t,
        Tref=Tref,
    )


def curves(prep: Prepared, res: JointResult, n: int = 600):
    """The fitted transit and velocity curves at the middle of the posterior, for plotting.

    Returns (hours, flux) for the transit and (phase, rv) for one orbit, plus the data shifted the same way.
    """
    p = res.median
    t0, P, k, b, M, R, q1, q2, mean, _, K = p[:N_SHARED]
    span = np.abs(_phase_days(prep.t_lc, t0, P)).max()
    tt = np.linspace(-span, span, n)
    params = _transit_params(0.0, P)
    _set_transit(params, 0.0, P, k, _orbit_size(P, M, R), b, q1, q2)
    flux_model = (_transit_model(params, tt).light_curve(params) - 1) * 1e3 + mean
    offsets = p[N_SHARED : N_SHARED + len(res.telescopes)]
    position = {name: i for i, name in enumerate(res.telescopes)}
    idx = np.array([position[name] for name in prep.telescope])
    ph = np.linspace(-0.5, 0.5, n)
    return {
        "hours": tt * 24,
        "flux": flux_model,
        "data_hours": _phase_days(prep.t_lc, t0, P) * 24,
        "data_flux": prep.flux,
        "phase": ph,
        "rv": -K * np.sin(2 * np.pi * ph),
        "data_phase": ((prep.t_rv - t0) / P + 0.5) % 1 - 0.5,
        "data_rv": prep.rv - offsets[idx],
        "data_rv_err": prep.rv_err,
        "telescope": prep.telescope,
    }
