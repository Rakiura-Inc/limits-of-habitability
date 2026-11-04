"""Light curves: smoothing keeps the dip, folding lines the transits up, binning averages them."""

import numpy as np
import pytest
from astropy.time import Time

import habtrain as ht

lk = pytest.importorskip("lightkurve")

P, T0 = 3.0, 2459000.5  # days; T0 as a full Julian date, as the survey gives it
DEPTH, DURATION_HOURS = 0.01, 3.0


def transiting_star(rng, days=27.0):
    """Two-minute data of a star that brightens slowly and dims by DEPTH during each transit."""
    t = np.arange(T0 - ht.constants.BTJD - 1.0, T0 - ht.constants.BTJD - 1.0 + days, 2 / 60 / 24)
    hours = ht.fit.hours_from_transit(t, P, T0 - ht.constants.BTJD)
    flux = 1 + 0.003 * np.sin(2 * np.pi * t / 9.0) - DEPTH * (np.abs(hours) < DURATION_HOURS / 2)
    flux = flux + rng.normal(0, 5e-4, t.size)
    return lk.LightCurve(time=Time(t, format="btjd", scale="tdb"), flux=flux, flux_err=np.full(t.size, 5e-4))


def test_flatten_then_fold_keeps_the_dip_and_drops_the_slow_change():
    lc = transiting_star(np.random.default_rng(1))
    flat = ht.lightcurve.flatten(lc, P, T0, DURATION_HOURS)
    hours, flux = ht.lightcurve.fold(flat, P, T0)
    assert np.all(np.abs(hours) <= P * 12)
    inside = np.median(flux[np.abs(hours) < 1])
    outside = np.median(flux[np.abs(hours) > 3])
    assert outside == pytest.approx(1.0, abs=5e-4)
    assert outside - inside == pytest.approx(DEPTH, rel=0.1)


def test_binned_takes_the_median_of_each_bin_and_skips_thin_bins():
    hours = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.5])
    flux = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 7.0])
    centres, middles = ht.lightcurve.binned(hours, flux, width_minutes=6.0)
    assert centres == pytest.approx([0.035])
    assert middles == pytest.approx([3.5])


def test_plain_turns_masked_points_into_nan():
    data = np.ma.masked_array([1.0, 2.0, 3.0], mask=[False, True, False])
    assert np.isnan(ht.lightcurve._plain(data)).tolist() == [False, True, False]
    assert ht.lightcurve._plain([4, 5]).tolist() == [4.0, 5.0]


def test_download_says_when_there_is_no_light_curve(monkeypatch):
    monkeypatch.setattr(lk, "search_lightcurve", lambda *args, **kwargs: [])
    with pytest.raises(RuntimeError, match="No 2-minute TESS light curve found for TOI-0"):
        ht.lightcurve.download("TOI-0")
