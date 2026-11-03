"""The Gaia project's plumbing: velocities in orbitize!'s units, the catalog row, and the staged best fit."""

import numpy as np
import pytest

import habtrain as ht

from .conftest import place


def lick_line(name, jd, rv, err):
    return place({(0, 11): name.ljust(11), (12, 26): jd, (27, 34): rv, (35, 40): err})


@pytest.fixture
def velocities_csv(monkeypatch, tmp_path):
    """iota Dra's velocities written from small copies of the two tables, as the notebook writes them."""
    tables = {
        "J/ApJ/646/505/table1.dat": "\n".join([
            lick_line("iota Dra", "2452000.5000", "-150.0", "5.0"),
            lick_line("HD 1", "2452001.5000", "1.0", "5.0"),
            lick_line("iota Dra", "2451900.5000", "120.0", "6.0"),
        ]),
        "J/AJ/162/211/table1.dat": "2458500.5 33.5 2.0\n2458600.5 -10.0 2.5\nbad\n",
    }
    monkeypatch.setattr(ht.orbits, "fetch", lambda path: tables[path].encode())
    return tmp_path / "iota_dra_rv.csv"


def test_iota_dra_velocities_are_in_mjd_and_km_per_s_sorted_by_date(velocities_csv):
    df = ht.orbits.iota_dra_velocities(str(velocities_csv))
    assert df.columns.tolist() == ["epoch", "object", "rv", "rv_err", "instrument"]
    assert df.instrument.tolist() == ["lick", "lick", "apf", "apf"]  # HD 1 left out, in date order
    assert df.epoch[0] == pytest.approx(51900.0)
    assert df.rv[0] == pytest.approx(0.12)
    assert df.rv_err[2] == pytest.approx(0.002)
    assert (df.object == 0).all()
    assert velocities_csv.read_text().splitlines()[0] == "epoch,object,rv,rv_err,instrument"


def test_pull_sigma_turns_chi_square_into_sigma():
    # With 2 degrees of freedom the tail chance is exp(-chi2 / 2); 0.0027 is the two-tailed chance of 3 sigma.
    assert ht.orbits.pull_sigma(-2 * np.log(0.0027)) == pytest.approx(3.0, abs=0.01)
    assert ht.orbits.pull_sigma(0.0) == pytest.approx(0.0, abs=1e-9)


def test_plot_proper_motions_draws_both_directions_against_the_average():
    row = {}
    for component in ("pmra", "pmdec"):
        for suffix, value in (("hip", 10.0), ("hg", 11.0), ("gaia", 12.5)):
            row[f"{component}_{suffix}"] = value
            row[f"{component}_{suffix}_error"] = 0.1
    fig = ht.orbits.plot_proper_motions(row, "HD 1")
    left, right = fig.axes
    assert left.get_title() == "HD 1: east-west motion"
    assert right.get_title() == "HD 1: north-south motion"
    assert [t.get_text() for t in left.get_xticklabels()] == ["Hipparcos, 1991", "average, 1991 to 2016", "Gaia, 2016"]
    heights = [bar.lines[0].get_ydata()[0] for bar in left.containers]
    assert heights == pytest.approx([-1.0, 0.0, 1.5])


@pytest.fixture
def hgca(monkeypatch, tmp_path):
    """A two-star stand-in for the catalog of accelerations, where orbitize! keeps it."""
    fits = pytest.importorskip("astropy.io.fits")
    orbitize = pytest.importorskip("orbitize")
    columns = [
        fits.Column(name="hip_id", format="J", array=[1, 75458]),
        fits.Column(name="chi2", format="D", array=[0.5, 300.0]),
    ]
    fits.BinTableHDU.from_columns(columns).writeto(tmp_path / "HGCA_vEDR3.fits")
    monkeypatch.setattr(orbitize, "DATADIR", str(tmp_path))


@pytest.mark.usefixtures("hgca")
def test_hgca_row_finds_a_star_by_its_hipparcos_number():
    row = ht.orbits.hgca_row(75458)
    assert row == {"hip_id": 75458, "chi2": 300.0}
    assert type(row["chi2"]) is float
    assert ht.orbits.hgca_row(2) is None


@pytest.fixture
def iota_dra(velocities_csv):
    pytest.importorskip("orbitize")
    ht.orbits.iota_dra_velocities(str(velocities_csv))
    return ht.orbits.iota_dra_system(str(velocities_csv))


def test_b_start_puts_b_at_its_published_orbit(iota_dra):
    start = ht.orbits.b_start(iota_dra)
    assert ht.orbits.describe(iota_dra, start) == {
        "semi-major axis (AU)": 1.448,
        "eccentricity": 0.7008,
        "minimum mass (Jupiter masses)": 11.67,
        "APF jitter (m/s)": 11.0,
        "Lick jitter (m/s)": 14.0,
    }
    assert start[iota_dra.param_idx["inc1"]] == pytest.approx(np.pi / 2)


def test_best_fit_moves_only_the_free_parameters(iota_dra):
    start = ht.orbits.b_start(iota_dra)
    idx = iota_dra.param_idx
    start[idx["sma2"]], start[idx["ecc2"]], start[idx["m2"]] = 20.0, 0.3, 0.01
    log_like, best = ht.orbits.best_fit(iota_dra, start, ["gamma_apf"], [0.0])
    assert np.isfinite(log_like)
    moved = np.flatnonzero(best != start)
    assert moved.tolist() in ([], [idx["gamma_apf"]])


def test_best_fit_refuses_an_orbit_that_cannot_be_real(iota_dra):
    start = ht.orbits.b_start(iota_dra)  # c's mass is 0 here, which is refused
    log_like, _ = ht.orbits.best_fit(iota_dra, start, ["gamma_apf"], [0.0])
    assert log_like == -ht.orbits.REFUSED
