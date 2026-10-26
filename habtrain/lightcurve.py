"""TESS light curves: download, clean, fold. The member measures the dip.

Light curves are the Science Processing Operations Center's 2-minute data (SPOC, PDCSAP flux),
fetched from the Mikulski Archive for Space Telescopes (MAST) through lightkurve.
"""
from __future__ import annotations

import numpy as np


def download(target: str, sectors: int = 2):
    """Up to `sectors` TESS sectors of 2-minute data for a star such as "TOI-669", joined into one curve.

    Each sector's brightness is divided by its own median, so 1.0 means the star's usual brightness.
    """
    import warnings
    with warnings.catch_warnings():        # lightkurve warns about an optional add-on these notebooks never use
        warnings.simplefilter("ignore", UserWarning)
        import lightkurve as lk
    found = lk.search_lightcurve(target, mission="TESS", author="SPOC", exptime=120)
    if len(found) == 0:
        raise RuntimeError(f"No 2-minute TESS light curve found for {target}.")
    lcs = found[:sectors].download_all(flux_column="pdcsap_flux")
    return lcs.stitch().remove_nans()


def flatten(lc, P: float, T0: float, duration_hours: float):
    """Take out the star's slow changes in brightness, leaving the transits alone.

    The transits (and a margin either side) are hidden from the smoothing so it cannot eat into them.
    """
    mask = lc.create_transit_mask(period=P, transit_time=T0 - 2457000, duration=duration_hours / 24 * 1.5)
    return lc.flatten(window_length=721, mask=mask)


def _plain(x):
    """A plain float array from a possibly masked one, with masked points set to NaN."""
    data = getattr(x, "unmasked", x)
    a = np.array(data, dtype=float)
    mask = getattr(x, "mask", None)
    if mask is not None:
        a[np.asarray(mask, dtype=bool)] = np.nan
    return a


def fold(flat, P: float, T0: float):
    """Fold on the period. Returns (hours from mid-transit, brightness) as plain arrays, gaps dropped."""
    f = flat.fold(period=P, epoch_time=T0 - 2457000)
    hours, flux = _plain(f.time.value) * 24, _plain(f.flux.value)
    keep = np.isfinite(hours) & np.isfinite(flux)
    return hours[keep], flux[keep]


def binned(hours, flux, width_minutes: float = 15.0):
    """Average the folded points in bins, so the dip stands out from the scatter."""
    w = width_minutes / 60
    edges = np.arange(hours.min(), hours.max() + w, w)
    idx = np.digitize(hours, edges)
    centres, means = [], []
    for i in np.unique(idx):
        sel = idx == i
        if sel.sum() >= 5:
            centres.append(hours[sel].mean())
            means.append(np.median(flux[sel]))
    return np.array(centres), np.array(means)
