"""The joint transit and velocity fit, on a planet we made, so the answer is known."""

import numpy as np
import pandas as pd
import pytest
from astropy.time import Time

import habtrain as ht

pytest.importorskip("batman")
lk = pytest.importorskip("lightkurve")

P, K, RATIO, IMPACT = 3.0, 10.0, 0.05, 0.3
STAR = ((1.0, 0.05), (1.0, 0.05))  # (mass, error) and (radius, error), solar units


def test_orbit_size_of_the_earth_is_215_sun_radii():
    assert ht.joint._orbit_size(365.25, 1.0, 1.0) == pytest.approx(215.0, rel=0.01)


def test_start_lists_every_parameter_once():
    value, spread = ht.joint._start(2000.0, P, *STAR, n_telescopes=2)
    assert len(value) == len(spread) == ht.joint.N_SHARED + 4
    assert value[:2].tolist() == [2000.0, P]
    assert value[ht.joint.SHARED.index("Mstar")] == 1.0
    assert spread[ht.joint.SHARED.index("Rstar")] == pytest.approx(0.05 / 3)


def model_flux(t, t0):
    """The transit of our planet in parts per thousand, from batman."""
    params = ht.joint._transit_params(t0, P)
    a = ht.joint._orbit_size(P, 1.0, 1.0)
    ht.joint._set_transit(params, t0, P, RATIO, a, IMPACT, 0.3, 0.3)
    return (ht.joint._transit_model(params, t).light_curve(params) - 1) * 1e3


@pytest.fixture(scope="module")
def prepared():
    rng = np.random.default_rng(4)
    t0 = 2000.0
    t_lc = np.concatenate([np.arange(-0.2, 0.2, 2 / 60 / 24) + t0 + n * P for n in range(4)])
    flux = model_flux(t_lc, t0) + rng.normal(0, 0.3, t_lc.size)
    t_rv = np.sort(rng.uniform(t0, t0 + 200, 40))
    telescope = np.where(np.arange(40) % 3, "HIRES", "APF")
    rv = -K * np.sin(2 * np.pi * (t_rv - t0) / P) + np.where(telescope == "APF", 5.0, -3.0) + rng.normal(0, 1.5, 40)
    return ht.joint.Prepared(
        "TOI-0 b", P, t0, t_lc, flux, np.full(t_lc.size, 0.3), t_rv, rv, np.full(40, 1.5), telescope
    )


@pytest.fixture(scope="module")
def result(prepared):
    return ht.joint.fit(prepared, *STAR, steps=3000, walkers=48, seed=2)


def test_joint_fit_recovers_the_planet_we_made(result):
    assert result.ratio[0] == pytest.approx(RATIO, rel=0.1)
    assert result.K[0] == pytest.approx(K, rel=0.15)
    assert result.radius[0] == pytest.approx(RATIO * ht.constants.R_SUN / ht.constants.R_EARTH, rel=0.1)
    assert result.mass[0] == pytest.approx(ht.fit.mass_from_wobble(K, P, 1.0), rel=0.2)
    assert result.telescopes == ["APF", "HIRES"]
    assert result.names[-4:] == ["offset APF", "offset HIRES", "log jitter APF", "log jitter HIRES"]
    assert all(plus > 0 and minus > 0 for _, plus, minus in (result.radius, result.mass, result.K))


def test_curves_shift_each_telescope_by_its_offset(prepared, result):
    c = ht.joint.curves(prepared, result, n=100)
    assert len(c["hours"]) == len(c["flux"]) == len(c["phase"]) == 100
    assert np.nanmin(c["flux"]) == pytest.approx(-(RATIO**2) * 1e3, rel=0.3)
    assert np.mean(c["data_rv"] - (-K * np.sin(2 * np.pi * c["data_phase"]))) == pytest.approx(0.0, abs=1.5)


def test_prepare_keeps_data_near_transit_and_leaves_out_velocities_taken_during_one(monkeypatch):
    t0_btjd = 2000.0
    t = np.arange(t0_btjd - 1.0, t0_btjd + 12.0, 2 / 60 / 24)
    flux = 1 + model_flux(t, t0_btjd) / 1e3
    sector = lk.LightCurve(time=Time(t, format="btjd", scale="tdb"), flux=flux, flux_err=np.full(t.size, 3e-4))
    sector.meta["SECTOR"] = 42

    class Found(list):
        """What lightkurve's search gives back: sliceable, and downloads every sector it holds."""

        def __getitem__(self, i):
            return Found(super().__getitem__(i)) if isinstance(i, slice) else super().__getitem__(i)

        def download_all(self, flux_column):
            return list(self)

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
    prep = ht.joint.prepare("TOI-1 b", rvs, orbit, duration_hours=2.0)
    assert prep.sectors == [42]
    assert prep.rv_in_transit == 2
    assert prep.rv.tolist() == [2.0]
    assert prep.t_rv.tolist() == pytest.approx([t0_btjd + 1.0])
    hours = ht.fit.hours_from_transit(prep.t_lc, P, t0_btjd)
    assert np.abs(hours).max() < 1.5 * 2.0 + 0.01  # only data within 1.5 transit durations
    assert prep.flux.min() == pytest.approx(-(RATIO**2) * 1e3, rel=0.3)  # parts per thousand


def test_joint_fit_runs_the_steps_asked_for_unless_ci_shortens_them(monkeypatch, capsys):
    monkeypatch.delenv(ht.joint.CI_STEPS_VARIABLE, raising=False)
    assert ht.joint._steps_to_run(20000) == 20000
    assert capsys.readouterr().out == ""
    monkeypatch.setenv(ht.joint.CI_STEPS_VARIABLE, "2000")
    assert ht.joint._steps_to_run(20000) == 2000
    assert ht.joint._steps_to_run(500) == 500  # never more steps than asked for
    assert "runs 2000 steps instead of 20000" in capsys.readouterr().out
