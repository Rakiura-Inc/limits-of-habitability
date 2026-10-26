"""Plumbing for the Gaia and radial velocity project: the catalog of accelerations, iota Draconis's
public velocities, and a two-stage best fit that gives orbitize!'s sampler a sensible start.

orbitize! (Blunt et al. 2020, AJ 159, 89) does the orbit fitting. It is an optional install:
pip install "habtrain[orbits]". Everything here imports it only when needed.

    Catalog of accelerations   Brandt 2021, ApJS 254, 42 (HGCA_vEDR3.fits, kept in orbitize!'s data folder)
    iota Dra, APF velocities   Hill et al. 2021, AJ 162, 211          VizieR J/AJ/162/211
    iota Dra, Lick velocities  Butler et al. 2006, ApJ 646, 505       VizieR J/ApJ/646/505
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .data import SHA256, fetch

SHA256.update({
    "J/AJ/162/211/table1.dat": "df2ae92f0ed6c442aa2e28c32420ba7fc69b21d2fe98291ba690b41d02591e18",
    "J/ApJ/646/505/table1.dat": "4ccf041207e4490e10ad8c0c85ab81b4c4aae55c420f63041dea36d43e2a65d6",
})

MJUP_IN_MSUN = 9.547919e-4     # Jupiter's mass in solar masses (IAU 2015)

# iota Draconis (HIP 75458). Star mass from asteroseismology and parallax from Gaia EDR3, both as
# Hill et al. 2021 and the catalog of accelerations give them.
IOTA_DRA = dict(hip=75458, star_mass=1.54, star_mass_err=0.09, parallax=32.5239, parallax_err=0.131)

# iota Dra b from velocities alone: Hill et al. 2021, table 3 (RadVel fit), value, minus, plus.
IOTA_DRA_B_PUBLISHED = {
    "semi-major axis (AU)": (1.448, 0.029, 0.028),
    "eccentricity": (0.7008, 0.0018, 0.0018),
    "minimum mass (Jupiter masses)": (11.67, 0.46, 0.45),
    "APF jitter (m/s)": (10.67, 0.42, 0.44),
    "Lick jitter (m/s)": (13.85, 0.84, 0.92),
}

# Where stage 1 holds b: its period, periastron time and the STAR's argument of periastron
# (radians), from the same table. orbitize! wants the companion's angle, which is this plus pi.
_B_PERIOD_DAYS = 510.855
_B_PERIASTRON_JD = 2452014.19
_B_OMEGA_STAR = 1.5696


def iota_dra_velocities(path: str = "iota_dra_rv.csv") -> pd.DataFrame:
    """Every public velocity of iota Dra, in the units orbitize! reads, written to `path`.

    Columns: epoch (MJD), object (0, the star), rv and rv_err (km/s), instrument ("apf" or "lick").
    """
    rows = []
    # Lick, 2000 to 2005: fixed-width table of many stars (CDS ReadMe J/ApJ/646/505); keep iota Dra.
    for line in fetch("J/ApJ/646/505/table1.dat").decode().splitlines():
        if line[0:11].strip() == "iota Dra":
            jd, rv, err = float(line[12:26]), float(line[27:34]), float(line[35:40])
            rows.append((jd - 2400000.5, rv / 1000, err / 1000, "lick"))
    # APF, 2018 to 2021: JD, velocity and error in m/s, whitespace separated.
    for line in fetch("J/AJ/162/211/table1.dat").decode().splitlines():
        p = line.split()
        if len(p) >= 3:
            rows.append((float(p[0]) - 2400000.5, float(p[1]) / 1000, float(p[2]) / 1000, "apf"))
    df = pd.DataFrame(rows, columns=["epoch", "rv", "rv_err", "instrument"]).sort_values("epoch")
    df.insert(1, "object", 0)
    df.to_csv(path, index=False, float_format="%.6f")
    return df.reset_index(drop=True)


def hgca_row(hip: int) -> dict | None:
    """One star's entry in the catalog of accelerations (EDR3 edition), or None if it is not there."""
    from astropy.io import fits
    from orbitize import DATADIR
    path = os.path.join(DATADIR, "HGCA_vEDR3.fits")
    if not os.path.exists(path):             # orbitize! fetches the catalog on first use; do the same here
        import urllib.request
        print("Downloading the catalog of accelerations (about 22 MB), once...")
        urllib.request.urlretrieve("https://cdsarc.cds.unistra.fr/ftp/J/ApJS/254/42/HGCA_vEDR3.fits", path)
    with fits.open(path, ignore_missing_simple=True) as h:
        d = h[1].data
        r = d[d["hip_id"] == hip]
        if len(r) == 0:
            return None
        return {name: (r[0][name].item() if hasattr(r[0][name], "item") else r[0][name]) for name in d.columns.names}


def pull_sigma(chi2: float, dof: int = 2) -> float:
    """Turn the catalog's chi-square (2 degrees of freedom) into a significance in sigma.

    The chance of a chi-square this big from noise alone, expressed as the number of sigma that
    leaves the same chance in both tails of a bell curve.
    """
    from scipy import stats
    p = stats.chi2.sf(chi2, dof)
    return float(stats.norm.isf(p / 2))


def plot_proper_motions(row: dict, name: str):
    """A star's motion across the sky, three ways: Hipparcos (1991), Gaia (2016), and the average
    over the 25 years between them. If they disagree, something is pulling the star."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=110)
    for ax, comp, label in zip(axes, ("pmra", "pmdec"), ("east-west", "north-south")):
        vals = [(row[f"{comp}_hip"], row[f"{comp}_hip_error"], "Hipparcos, 1991"),
                (row[f"{comp}_hg"], row[f"{comp}_hg_error"], "average, 1991 to 2016"),
                (row[f"{comp}_gaia"], row[f"{comp}_gaia_error"], "Gaia, 2016")]
        ref = vals[1][0]                      # show differences from the 25-year average
        for i, (v, e, lab) in enumerate(vals):
            ax.errorbar([i], [v - ref], yerr=[e], fmt="o", ms=7, capsize=4, color=["#3b6ea5", "#555", "#d1495b"][i])
        ax.axhline(0, color="#999", lw=0.8, ls="--")
        ax.set_xticks(range(3), [v[2] for v in vals], fontsize=8.5)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylabel("Motion minus the 25-year average (mas per year)")
        ax.set_title(f"{name}: {label} motion", fontsize=10.5)
    fig.text(0.01, 0.005, "Data: Brandt 2021, Hipparcos-Gaia Catalog of Accelerations (EDR3), ApJS 254, 42.",
             fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def iota_dra_system(rv_path: str = "iota_dra_rv.csv"):
    """orbitize! system for iota Dra with two companions (b, then c) and velocities only."""
    from orbitize import read_input, system
    data = read_input.read_file(rv_path)
    s = IOTA_DRA
    return system.System(2, data, s["star_mass"], s["parallax"], mass_err=s["star_mass_err"],
                         plx_err=s["parallax_err"], fit_secondary_mass=True)


def b_start(sys_) -> np.ndarray:
    """A starting point with b at its published orbit, seen edge on, and c far out and unknown.

    Edge on (inclination 90 degrees) means b's fitted mass is its minimum mass.
    """
    from orbitize import basis
    idx, p = sys_.param_idx, np.zeros(len(sys_.labels))
    tau_b = basis.tp_to_tau(_B_PERIASTRON_JD - 2400000.5, sys_.tau_ref_epoch, _B_PERIOD_DAYS)
    start = dict(sma1=1.448, ecc1=0.7008, inc1=np.pi / 2, aop1=_B_OMEGA_STAR + np.pi, pan1=1.0, tau1=tau_b,
                 inc2=np.pi / 2, pan2=1.0, plx=IOTA_DRA["parallax"], m0=IOTA_DRA["star_mass"],
                 m1=11.67 * MJUP_IN_MSUN, sigma_apf=0.011, sigma_lick=0.014)
    for k, v in start.items():
        p[idx[k]] = v
    return p


def best_fit(sys_, start: np.ndarray, free: list[str], x0: list[float]) -> tuple[float, np.ndarray]:
    """The most likely orbit with only the parameters in `free` allowed to move from `start`.

    Returns (log likelihood, full parameter array). Uses the same likelihood the sampler uses,
    and refuses orbits that cannot be real (negative masses, eccentricity of 1 or more).
    """
    from orbitize import sampler
    from scipy.optimize import minimize
    idx = sys_.param_idx
    like = sampler.MCMC(sys_, 1, 50, 1)._logl       # orbitize!'s own log likelihood

    def place(x):
        q = start.copy()
        for k, v in zip(free, x):
            q[idx[k]] = v
        return q

    def cost(x):
        q = place(x)
        if not (0 <= q[idx["ecc1"]] < 0.95 and 0 <= q[idx["ecc2"]] < 0.95 and q[idx["m1"]] > 0
                and q[idx["m2"]] > 0 and 3 < q[idx["sma2"]] < 100):
            return 1e12
        return -float(like(q))

    r = minimize(cost, x0, method="Nelder-Mead",
                 options=dict(maxiter=20000, xatol=1e-8, fatol=1e-5, adaptive=True))
    return -float(r.fun), place(r.x)


def describe(sys_, q: np.ndarray) -> dict:
    """b's orbit and the two jitters from a parameter array, in the units the paper reports."""
    idx = sys_.param_idx
    return {
        "semi-major axis (AU)": round(float(q[idx["sma1"]]), 4),
        "eccentricity": round(float(q[idx["ecc1"]]), 4),
        "minimum mass (Jupiter masses)": round(float(q[idx["m1"]] / MJUP_IN_MSUN), 2),
        "APF jitter (m/s)": round(float(q[idx["sigma_apf"]]) * 1000, 1),
        "Lick jitter (m/s)": round(float(q[idx["sigma_lick"]]) * 1000, 1),
    }
