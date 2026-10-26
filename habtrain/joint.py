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

from .fit import in_transit

G, M_SUN, R_SUN, M_EARTH, R_EARTH = 6.674e-11, 1.989e30, 6.957e8, 5.972e24, 6.371e6


@dataclass
class Prepared:
    """Everything the joint fit needs for one planet, in BTJD (BJD minus 2457000) throughout."""
    planet: str
    P: float
    T0: float
    t_lc: np.ndarray        # light-curve times near transit (BTJD)
    flux: np.ndarray        # brightness minus 1, in parts per thousand
    flux_err: np.ndarray
    t_rv: np.ndarray        # velocity times (BTJD)
    rv: np.ndarray
    rv_err: np.ndarray
    telescope: np.ndarray
    sectors: list = field(default_factory=list)
    rv_in_transit: int = 0   # velocities left out because they were taken during a transit


def prepare(planet: str, rvs, orbit: dict, duration_hours: float, max_sectors: int | None = 8) -> Prepared:
    """Download up to `max_sectors` TESS 2-minute sectors (None for all), flatten each with its transits hidden,
    keep the data near transit, and collect the survey's velocities for the same star, leaving out any taken
    during a transit: the star's spin distorts those (the swing in notebook 4's Extra), and this model has no term for it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        import lightkurve as lk
    P, T0 = orbit["P"], orbit["T0"] - 2457000.0          # the survey's T0 is full BJD; TESS uses BTJD
    dur_days = duration_hours / 24
    found = lk.search_lightcurve(planet.split()[0], mission="TESS", author="SPOC", exptime=120)
    if max_sectors:
        found = found[:max_sectors]
    ts, fs, es, sectors = [], [], [], []
    for lc in found.download_all(flux_column="pdcsap_flux"):
        lc = lc.remove_nans().normalize()
        mask = lc.create_transit_mask(period=P, transit_time=T0, duration=2 * dur_days)
        flat = lc.flatten(window_length=721, mask=mask).remove_outliers(sigma_upper=4, sigma_lower=20)
        t = np.array(getattr(flat.time.value, "unmasked", flat.time.value), dtype=float)
        f = np.array(getattr(flat.flux.value, "unmasked", flat.flux.value), dtype=float)
        e = np.array(getattr(flat.flux_err.value, "unmasked", flat.flux_err.value), dtype=float)
        phase = (t - T0 + 0.5 * P) % P - 0.5 * P
        keep = (np.abs(phase) < 1.5 * dur_days) & np.isfinite(f) & np.isfinite(e)
        ts.append(t[keep]); fs.append((f[keep] - 1) * 1e3); es.append(e[keep] * 1e3)
        sectors.append(int(lc.meta.get("SECTOR", -1)))
    d = rvs[rvs.TKS == orbit["TKS"]]
    during = in_transit(d.BJD.values, orbit["P"], orbit["T0"], duration_hours)
    d = d[~during]
    return Prepared(planet, P, T0, np.concatenate(ts), np.concatenate(fs), np.concatenate(es),
                    d.BJD.values - 2457000.0, d.RV.values, d.eRV.values, d.Inst.values, sectors, int(during.sum()))


@dataclass
class JointResult:
    """What the joint fit gives back. Each tuple is (median, plus, minus): the middle of the posterior and
    how far its 68% range reaches above and below."""
    radius: tuple            # in Earth radii
    mass: tuple              # in Earth masses
    K: tuple                 # m/s
    impact: tuple            # impact parameter b: 0 crosses the middle of the star, 1 grazes its edge
    ratio: tuple             # planet radius / star radius
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


def fit(prep: Prepared, star_mass: tuple, star_radius: tuple, steps: int = 20000, walkers: int = 48,
        seed: int = 1) -> JointResult:
    """Fit transit and velocities together. star_mass and star_radius are (value, uncertainty), in solar units."""
    import batman
    import emcee
    Ms, eMs = star_mass
    Rs, eRs = star_radius
    t0_start = time.time()
    x, y, yerr = prep.t_lc, prep.flux, prep.flux_err
    tel = np.asarray(prep.telescope)
    names_t = sorted(set(tel.tolist()))
    idx = np.array([names_t.index(i) for i in tel])
    nt = len(names_t)
    # A transit time in the middle of the TESS data, so the period and transit time are not tangled together.
    Tref = prep.T0 + np.round((np.median(x) - prep.T0) / prep.P) * prep.P

    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a, params.inc = Tref, prep.P, 0.03, 10.0, 89.0
    params.ecc, params.w, params.limb_dark, params.u = 0.0, 90.0, "quadratic", [0.4, 0.2]
    model = batman.TransitModel(params, x, supersample_factor=3, exp_time=120 / 86400)

    # Parameters: transit time, period, radius ratio, impact parameter, star mass and radius,
    # two limb-darkening numbers (q1, q2), a flux offset, light-curve jitter, K, then per telescope an offset and a jitter.
    names = (["t0", "P", "ratio", "impact", "Mstar", "Rstar", "q1", "q2", "flux offset", "log jitter lc", "K"]
             + [f"offset {n}" for n in names_t] + [f"log jitter {n}" for n in names_t])

    def predict(p):
        t0, P, k, b, M, R, q1, q2, mean, _, K = p[:11]
        a = (G * M * M_SUN * (P * 86400) ** 2 / (4 * np.pi ** 2)) ** (1 / 3) / (R * R_SUN)   # orbit size in star radii
        params.t0, params.per, params.rp, params.a = t0, P, k, a
        params.inc = np.degrees(np.arccos(min(b / a, 1.0)))
        params.u = [2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)]
        lc = (model.light_curve(params) - 1) * 1e3 + mean
        rv = -K * np.sin(2 * np.pi * (prep.t_rv - t0) / P) + p[11:11 + nt][idx]
        return lc, rv

    def log_prob(p):
        t0, P, k, b, M, R, q1, q2, mean, ljl, K = p[:11]
        lj = p[11 + nt:]
        if not (0 < k < 0.3 and 0 <= b < 1 + k and 0 < q1 < 1 and 0 < q2 < 1 and 0 <= K < 500 and M > 0.1 and R > 0.1
                and abs(t0 - Tref) < 0.1 and abs(P - prep.P) < 0.01 and -10 < ljl < 3 and np.all(np.abs(lj) < 5)):
            return -np.inf
        lc, rv = predict(p)
        s2 = yerr ** 2 + np.exp(2 * ljl)
        s2r = prep.rv_err ** 2 + np.exp(2 * lj[idx])
        return (-0.5 * np.sum((y - lc) ** 2 / s2 + np.log(s2)) - 0.5 * np.sum((prep.rv - rv) ** 2 / s2r + np.log(s2r))
                - 0.5 * ((M - Ms) / eMs) ** 2 - 0.5 * ((R - Rs) / eRs) ** 2)

    # Starting guesses; the walkers spread out from here. 0.03 is a typical planet-to-star size ratio for these targets.
    p0 = np.array([Tref, prep.P, 0.03, 0.3, Ms, Rs, 0.3, 0.3, 0.0, -2.0, 5.0] + [0.0] * nt + [1.0] * nt)
    scale = np.array([1e-3, 1e-6, 1e-3, 0.05, eMs / 3, eRs / 3, 0.05, 0.05, 0.01, 0.1, 0.5] + [0.5] * nt + [0.1] * nt)
    rng = np.random.default_rng(seed)
    start = p0 + scale * rng.standard_normal((walkers, len(p0)))
    start[:, 3] = np.abs(start[:, 3])
    sampler = emcee.EnsembleSampler(walkers, len(p0), log_prob)
    # The parameters live on very different scales (a transit time near 2000 days, a period known to a
    # millionth of a day), which trips emcee's start check even though the walkers are independent.
    sampler.run_mcmc(start, steps, progress=False, skip_initial_state_check=True)
    try:
        tau = float(np.nanmax(sampler.get_autocorr_time(tol=0)))
    except Exception:
        tau = float("nan")
    ch = sampler.get_chain(discard=steps // 2, thin=10, flat=True)
    P, k, b, M, R, K = ch[:, 1], ch[:, 2], ch[:, 3], ch[:, 4], ch[:, 5], ch[:, 10]
    a = (G * M * M_SUN * (P * 86400) ** 2 / (4 * np.pi ** 2)) ** (1 / 3) / (R * R_SUN)
    sini = np.sqrt(1 - np.clip(b / a, 0, 1) ** 2)
    radius = k * R * R_SUN / R_EARTH
    mass = K * (P * 86400 / (2 * np.pi * G)) ** (1 / 3) * (M * M_SUN) ** (2 / 3) / sini / M_EARTH
    return JointResult(_summary(radius), _summary(mass), _summary(K), _summary(b), _summary(k), tau, steps,
                       time.time() - t0_start, ch, names, converged=bool(np.isfinite(tau) and steps > 50 * tau),
                       median=np.median(ch, axis=0), telescopes=names_t, Tref=Tref)


def curves(prep: Prepared, res: JointResult, n: int = 600):
    """The fitted transit and velocity curves at the middle of the posterior, for plotting.

    Returns (hours, flux) for the transit and (phase, rv) for one orbit, plus the data shifted the same way.
    """
    import batman
    p = res.median
    t0, P, k, b, M, R, q1, q2, mean, _, K = p[:11]
    nt = len(res.telescopes)
    a = (G * M * M_SUN * (P * 86400) ** 2 / (4 * np.pi ** 2)) ** (1 / 3) / (R * R_SUN)
    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a = 0.0, P, k, a
    params.inc = np.degrees(np.arccos(min(b / a, 1.0)))
    params.ecc, params.w, params.limb_dark = 0.0, 90.0, "quadratic"
    params.u = [2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)]
    span = np.abs((prep.t_lc - t0 + 0.5 * P) % P - 0.5 * P).max()
    tt = np.linspace(-span, span, n)
    flux_model = (batman.TransitModel(params, tt, supersample_factor=3, exp_time=120 / 86400).light_curve(params) - 1) * 1e3 + mean
    data_hours = ((prep.t_lc - t0 + 0.5 * P) % P - 0.5 * P) * 24
    offsets = p[11:11 + nt]
    idx = np.array([res.telescopes.index(i) for i in prep.telescope])
    phase = ((prep.t_rv - t0) / P + 0.5) % 1 - 0.5
    ph = np.linspace(-0.5, 0.5, n)
    return dict(hours=tt * 24, flux=flux_model, data_hours=data_hours, data_flux=prep.flux,
                phase=ph, rv=-K * np.sin(2 * np.pi * ph), data_phase=phase, data_rv=prep.rv - offsets[idx],
                data_rv_err=prep.rv_err, telescope=prep.telescope)
