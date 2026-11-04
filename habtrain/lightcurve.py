"""Plumbing for notebook 3's light curves. The notebook calls lightkurve itself to download, flatten and fold.

Light curves are the Science Processing Operations Center's 2-minute data (SPOC, PDCSAP flux),
fetched from the Mikulski Archive for Space Telescopes (MAST) through lightkurve.
"""

from __future__ import annotations

import warnings

import numpy as np

MIN_POINTS_PER_BIN = 5


def lightkurve():
    """The lightkurve module, imported without its warning about an optional add-on nobody here uses."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        import lightkurve as lk
    return lk


def _plain(column):
    """A plain float array from a possibly masked one, with masked points set to NaN."""
    values = np.array(getattr(column, "unmasked", column), dtype=float)
    mask = getattr(column, "mask", None)
    if mask is not None:
        values[np.asarray(mask, dtype=bool)] = np.nan
    return values


def folded_arrays(folded):
    """(hours from mid-transit, brightness) from a light curve lightkurve has folded, gaps dropped.

    lightkurve keeps the gaps between sectors as masked points; this drops them, so the arrays
    work with numpy's median and the plots.
    """
    hours = _plain(folded.time.value) * 24
    flux = _plain(folded.flux.value)
    keep = np.isfinite(hours) & np.isfinite(flux)
    return hours[keep], flux[keep]


def binned(hours, flux, width_minutes: float = 15.0):
    """Average the folded points in bins, so the dip stands out from the scatter.

    Returns (bin centres, median brightness in each bin), leaving out bins with fewer than
    MIN_POINTS_PER_BIN points. A plain loop over about a hundred bins, slower than np.bincount and
    easier to read.
    """
    width_hours = width_minutes / 60
    edges = np.arange(hours.min(), hours.max() + width_hours, width_hours)
    bin_of = np.digitize(hours, edges)
    centres, middles = [], []
    for number in np.unique(bin_of):
        inside = bin_of == number
        if inside.sum() >= MIN_POINTS_PER_BIN:
            centres.append(hours[inside].mean())
            middles.append(np.median(flux[inside]))
    return np.array(centres), np.array(middles)
