"""habtrain.lightcurve: the plain arrays from a folded light curve, and the 15-minute averages."""

import numpy as np
import pytest
from astropy.time import Time

import habtrain as ht

lk = pytest.importorskip("lightkurve")


def test_folded_arrays_give_hours_from_mid_transit_and_drop_gaps():
    times = np.arange(0.0, 9.0, 0.01)
    flux = np.ones(times.size)
    flux[10] = np.nan  # a gap
    light_curve = lk.LightCurve(time=Time(times, format="btjd", scale="tdb"), flux=flux)
    hours, folded_flux = ht.lightcurve.folded_arrays(light_curve.fold(period=3.0, epoch_time=0.0))
    assert len(hours) == len(folded_flux) == times.size - 1
    assert np.abs(hours).max() <= 1.5 * 24
    assert np.all(np.isfinite(folded_flux))


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
