"""The data for tess_02's joint fit: TESS light curves near transit and the survey's velocities.

The model, the likelihood and the emcee run are written out in the notebook. This module only
downloads and cleans the data, with notebook 3's lightkurve calls and two changes for a full fit:
a wider mask around each transit, and outliers clipped. It prints every setting it used.
Times are BTJD (BJD minus 2457000) throughout, as TESS gives them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import BTJD
from .fit import in_transit
from .lightcurve import lightkurve

FLATTEN_WINDOW_POINTS = 721  # the smoothing window: about a day of 2-minute data, as in notebook 3
# Hide this many transit lengths, centred on each transit, from the smoothing. Notebook 3 hides 1.5;
# the joint fit models the transit's sloping edges, so it keeps more room either side.
MASK_TRANSIT_LENGTHS = 2
NEAR_TRANSIT_LENGTHS = 1.5  # keep light-curve points within this many transit lengths of mid-transit
# Outliers are clipped hard above the light curve (cosmic rays, scattered light) and barely below,
# so no point inside a transit is ever clipped.
OUTLIER_SIGMA_ABOVE = 4
OUTLIER_SIGMA_BELOW = 20
MAX_SECTORS = 8


@dataclass
class Prepared:
    """One planet's data for the joint fit, in BTJD throughout."""

    planet: str
    period_days: float
    transit_btjd: float  # the published time of one mid-transit
    times_btjd: np.ndarray  # light-curve times near transit
    flux_ppt: np.ndarray  # brightness minus 1, in parts per thousand
    flux_err_ppt: np.ndarray
    rv_times_btjd: np.ndarray
    rv_ms: np.ndarray
    rv_err_ms: np.ndarray
    telescope: np.ndarray  # which spectrograph took each velocity
    sectors: list = field(default_factory=list)
    rv_in_transit: int = 0  # velocities left out because they were taken during a transit


def _values(column) -> np.ndarray:
    """A light-curve column as a plain float array, whether or not lightkurve masked it."""
    return np.array(getattr(column, "unmasked", column), dtype=float)


def _days_from_transit(times, transit_time: float, period_days: float):
    return (times - transit_time + 0.5 * period_days) % period_days - 0.5 * period_days


def _sector_near_transit(light_curve, period_days: float, transit_btjd: float, transit_days: float):
    """One sector flattened with its transits hidden, cut to the points near transit, as plain arrays."""
    clean = light_curve.remove_nans().normalize()
    mask = clean.create_transit_mask(
        period=period_days, transit_time=transit_btjd, duration=MASK_TRANSIT_LENGTHS * transit_days
    )
    flat = clean.flatten(window_length=FLATTEN_WINDOW_POINTS, mask=mask)
    flat = flat.remove_outliers(sigma_upper=OUTLIER_SIGMA_ABOVE, sigma_lower=OUTLIER_SIGMA_BELOW)
    times, flux, flux_err = _values(flat.time.value), _values(flat.flux.value), _values(flat.flux_err.value)
    near = np.abs(_days_from_transit(times, transit_btjd, period_days)) < NEAR_TRANSIT_LENGTHS * transit_days
    keep = near & np.isfinite(flux) & np.isfinite(flux_err)
    sector = int(clean.meta.get("SECTOR", -1))
    return times[keep], flux[keep], flux_err[keep], sector


def prepare(planet: str, rvs, orbit: dict, transit_hours: float) -> Prepared:
    """Download up to 8 TESS sectors and collect the survey's velocities for one planet.

    Each sector is normalised, flattened with its transits hidden and cut to the points near transit.
    Velocities taken during a transit are left out: the star's spin distorts those (the swing in
    notebook 4's Extra), and the joint model has no term for it. Prints what it did.
    """
    lk = lightkurve()
    period_days = orbit["P"]
    transit_btjd = orbit["T0"] - BTJD  # the survey's T0 is full BJD; TESS uses BTJD
    transit_days = transit_hours / 24
    found = lk.search_lightcurve(planet.split()[0], mission="TESS", author="SPOC", exptime=120)
    times, flux, flux_err, sectors = [], [], [], []
    for light_curve in found[:MAX_SECTORS].download_all(flux_column="pdcsap_flux"):
        sector_times, sector_flux, sector_err, sector = _sector_near_transit(
            light_curve, period_days, transit_btjd, transit_days
        )
        times.append(sector_times)
        flux.append((sector_flux - 1) * 1e3)  # parts per thousand
        flux_err.append(sector_err * 1e3)
        sectors.append(sector)
    print(
        f"Up to {MAX_SECTORS} sectors. Each one: flattened over {FLATTEN_WINDOW_POINTS} points (about a day) "
        f"with {MASK_TRANSIT_LENGTHS} transit lengths around each transit hidden, points over "
        f"{OUTLIER_SIGMA_ABOVE} sigma above or {OUTLIER_SIGMA_BELOW} sigma below removed, and cut to "
        f"{NEAR_TRANSIT_LENGTHS} transit lengths either side of mid-transit. "
        "The survey's times are BJD; every time here is BTJD, BJD minus 2457000, as TESS counts."
    )
    star_rvs = rvs[rvs.TKS == orbit["TKS"]]
    during = in_transit(star_rvs.BJD.to_numpy(), orbit["P"], orbit["T0"], transit_hours)
    kept = star_rvs[~during]
    return Prepared(
        planet=planet,
        period_days=period_days,
        transit_btjd=transit_btjd,
        times_btjd=np.concatenate(times),
        flux_ppt=np.concatenate(flux),
        flux_err_ppt=np.concatenate(flux_err),
        rv_times_btjd=kept.BJD.to_numpy() - BTJD,
        rv_ms=kept.RV.to_numpy(),
        rv_err_ms=kept.eRV.to_numpy(),
        telescope=kept.Inst.to_numpy(),
        sectors=sectors,
        rv_in_transit=int(during.sum()),
    )
