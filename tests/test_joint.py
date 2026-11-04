"""tess_02's data preparation (habtrain.joint), and the notebook's own joint model on a planet we made."""

import numpy as np
import pandas as pd
import pytest
from astropy.time import Time

import habtrain as ht

from .notebook_code import definitions

pytest.importorskip("batman")
lk = pytest.importorskip("lightkurve")

P, K, RATIO, IMPACT = 3.0, 10.0, 0.05, 0.3
STAR = (ht.data.Measured(1.0, 0.05), ht.data.Measured(1.0, 0.05))  # mass and radius, solar units


@pytest.fixture(scope="module")
def tess02():
    return definitions("tess_02_radius_and_mass.ipynb")


def model_flux(tess02, times, t0):
    """Our planet's transit in parts per thousand, from the notebook's own batman calls."""
    params = tess02["batman_parameters"](t0, P)
    tess02["set_orbit"](params, [t0, P, RATIO, IMPACT, 1.0, 1.0, 0.3, 0.3])  # in SHARED_PARAMETERS order
    return (tess02["transit_model"](params, times).light_curve(params) - 1) * 1e3


def test_orbit_size_of_the_earth_is_215_sun_radii(tess02):
    assert tess02["orbit_size"](365.25, 1.0, 1.0) == pytest.approx(215.0, rel=0.01)


@pytest.fixture(scope="module")
def made(tess02):
    rng = np.random.default_rng(4)
    t0 = 2000.0
    times = np.concatenate([np.arange(-0.2, 0.2, 2 / 60 / 24) + t0 + n * P for n in range(4)])
    flux = model_flux(tess02, times, t0) + rng.normal(0, 0.3, times.size)
    rv_times = np.sort(rng.uniform(t0, t0 + 200, 40))
    telescope = np.where(np.arange(40) % 3, "HIRES", "APF")
    offsets = np.where(telescope == "APF", 5.0, -3.0)
    rv = -K * np.sin(2 * np.pi * (rv_times - t0) / P) + offsets + rng.normal(0, 1.5, 40)
    return ht.joint.Prepared(
        planet="TOI-0 b",
        period_days=P,
        transit_btjd=t0,
        times_btjd=times,
        flux_ppt=flux,
        flux_err_ppt=np.full(times.size, 0.3),
        rv_times_btjd=rv_times,
        rv_ms=rv,
        rv_err_ms=np.full(40, 1.5),
        telescope=telescope,
    )


@pytest.fixture(scope="module")
def results(tess02, made):
    setup = tess02["set_up_fit"](made, *STAR)
    sampler = tess02["run_joint_fit"](setup, 2.5, 3000)
    return setup, tess02["joint_results"](sampler, setup)


def test_the_start_lists_every_parameter_once_and_starts_near_the_dip(tess02, made):
    setup = tess02["set_up_fit"](made, *STAR)
    values, spreads = tess02["starting_point"](setup, 2.5)
    assert len(values) == len(spreads) == len(tess02["SHARED_PARAMETERS"]) + 4
    at = tess02["POSITION"]
    assert values[at["ratio"]] == pytest.approx(RATIO, rel=0.2)  # from the dip's depth
    assert values[at["K"]] == pytest.approx(K, rel=0.2)  # from the least-squares fit
    assert setup["telescopes"] == ["APF", "HIRES"]


def test_the_limits_refuse_an_impossible_orbit(tess02, made):
    setup = tess02["set_up_fit"](made, *STAR)
    values, _ = tess02["starting_point"](setup, 2.5)
    assert np.isfinite(tess02["log_probability"](values, setup))
    too_big = values.copy()
    too_big[tess02["POSITION"]["ratio"]] = 0.5  # a planet half the size of its star
    assert tess02["log_probability"](too_big, setup) == -np.inf


def test_joint_fit_recovers_the_planet_we_made(tess02, results):
    _, fitted = results
    assert fitted["K"].median == pytest.approx(K, rel=0.15)
    assert fitted["radius"].median == pytest.approx(RATIO * ht.constants.R_SUN / ht.constants.R_EARTH, rel=0.1)
    assert fitted["mass"].median == pytest.approx(tess02["mass_from_wobble"](K, P, 1.0), rel=0.2)
    for name in ("radius", "mass", "K"):
        assert fitted[name].minus > 0
        assert fitted[name].plus > 0


def test_fitted_curves_shift_each_telescope_by_its_offset(tess02, results):
    setup, fitted = results
    curves = tess02["fitted_curves"](setup, fitted)
    assert len(curves["hours"]) == len(curves["flux"]) == len(curves["phase"]) == 600
    assert np.nanmin(curves["flux"]) == pytest.approx(-(RATIO**2) * 1e3, rel=0.3)
    left = curves["data_rv"] - (-K * np.sin(2 * np.pi * curves["data_phase"]))
    assert np.mean(left) == pytest.approx(0.0, abs=1.5)


class Found(list):
    """What lightkurve's search gives back: sliceable, and downloads every sector it holds."""

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Found(super().__getitem__(index))
        return super().__getitem__(index)

    def download_all(self, flux_column):
        return list(self)


def test_prepare_keeps_data_near_transit_and_leaves_out_velocities_taken_during_one(tess02, monkeypatch, capsys):
    t0_btjd = 2000.0
    times = np.arange(t0_btjd - 1.0, t0_btjd + 12.0, 2 / 60 / 24)
    flux = 1 + model_flux(tess02, times, t0_btjd) / 1e3
    sector = lk.LightCurve(time=Time(times, format="btjd", scale="tdb"), flux=flux, flux_err=np.full(times.size, 3e-4))
    sector.meta["SECTOR"] = 42
    monkeypatch.setattr(lk, "search_lightcurve", lambda *args, **kwargs: Found([sector]))
    orbit = {"TKS": "T000001", "P": P, "T0": t0_btjd + ht.constants.BTJD}
    full = t0_btjd + ht.constants.BTJD
    rvs = pd.DataFrame({
        "TKS": ["T000001", "T000001", "T000001", "T000002"],
        "BJD": [full + 0.01, full + 1.0, full + 3 * P, full + 1.0],
        "RV": [1.0, 2.0, 3.0, 4.0],
        "eRV": [1.0, 1.0, 1.0, 1.0],
        "Inst": ["HIRES", "HIRES", "APF", "HIRES"],
    })
    prepared = ht.joint.prepare("TOI-1 b", rvs, orbit, 2.0)
    assert prepared.sectors == [42]
    assert prepared.rv_in_transit == 2
    assert prepared.rv_ms.tolist() == [2.0]
    assert prepared.rv_times_btjd.tolist() == pytest.approx([t0_btjd + 1.0])
    hours = ht.fit.hours_from_transit(prepared.times_btjd, P, t0_btjd)
    assert np.abs(hours).max() < 1.5 * 2.0 + 0.01  # only data within 1.5 transit lengths
    assert prepared.flux_ppt.min() == pytest.approx(-(RATIO**2) * 1e3, rel=0.3)  # parts per thousand
    assert "flattened over 721 points (about a day) with 2 transit lengths" in capsys.readouterr().out
