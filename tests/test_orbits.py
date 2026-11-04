"""The Gaia project: velocities in orbitize!'s units, the catalog row, and gaia_03's own best fit."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import habtrain as ht

from .conftest import place
from .notebook_code import definitions


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


def test_iota_dra_tables_keep_only_iota_dra_as_published(velocities_csv):
    table = ht.orbits.iota_dra_tables()
    assert table.columns.tolist() == ["jd", "rv_ms", "error_ms", "telescope"]
    assert table.telescope.tolist() == ["lick", "lick", "apf", "apf"]  # HD 1 left out
    assert table.jd.tolist() == [2452000.5, 2451900.5, 2458500.5, 2458600.5]
    assert table.rv_ms[1] == 120.0


def test_gaia_02s_pull_sigma_turns_chi_square_into_sigma():
    pull_sigma = definitions("gaia_02_who_feels_a_pull.ipynb")["pull_sigma"]
    # With 2 degrees of freedom the tail chance is exp(-chi2 / 2); 0.0027 is the two-tailed chance of 3 sigma.
    assert pull_sigma(-2 * np.log(0.0027)) == pytest.approx(3.0, abs=0.01)
    assert pull_sigma(0.0) == pytest.approx(0.0, abs=1e-9)


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


def test_published_orbit_of_b_names_its_source():
    published = ht.orbits.published_iota_dra_b()
    assert published.columns.tolist() == ["quantity", "value", "minus", "plus", "unit", "source"]
    assert published.set_index("quantity").value["eccentricity"] == 0.7008
    assert published.source.str.startswith("Hill et al. 2021").all()


@pytest.fixture
def gaia03(velocities_csv):
    """gaia_03's own functions, with the system and start the notebook builds for them."""
    orbitize = pytest.importorskip("orbitize")
    from orbitize import read_input, system

    public = ht.orbits.iota_dra_tables()
    pd.DataFrame({
        "epoch": public.jd - ht.orbits.MJD_OFFSET,
        "object": 0,
        "rv": public.rv_ms / 1000,
        "rv_err": public.error_ms / 1000,
        "instrument": public.telescope,
    }).to_csv(velocities_csv, index=False)
    star = ht.orbits.IOTA_DRA
    iota_dra = system.System(
        2,
        read_input.read_file(str(velocities_csv)),
        star["star_mass"],
        star["parallax"],
        mass_err=star["star_mass_err"],
        plx_err=star["parallax_err"],
        fit_secondary_mass=True,
    )
    table = iota_dra.data_table
    namespace = definitions(
        "gaia_03_inner_planet_from_velocities.ipynb",
        iota_dra=iota_dra,
        index=iota_dra.param_idx,
        mjup=ht.orbits.MJUP_IN_MSUN,
        measured=np.array([table["quant1"], table["quant2"]]).T,
        errors=np.array([table["quant1_err"], table["quant2_err"]]).T,
    )
    assert orbitize.__version__
    return namespace


def b_at_its_published_orbit(gaia03):
    index = gaia03["index"]
    start = np.zeros(len(gaia03["iota_dra"].labels))
    start[index["sma1"]], start[index["ecc1"]] = 1.448, 0.7008
    start[index["m1"]] = 11.67 * ht.orbits.MJUP_IN_MSUN
    start[index["sigma_apf"]], start[index["sigma_lick"]] = 0.011, 0.014
    start[index["inc1"]] = start[index["inc2"]] = np.pi / 2
    start[index["plx"]], start[index["m0"]] = ht.orbits.IOTA_DRA["parallax"], ht.orbits.IOTA_DRA["star_mass"]
    return start


def test_orbit_of_b_reports_the_papers_units(gaia03):
    assert gaia03["orbit_of_b"](b_at_its_published_orbit(gaia03)) == {
        "semi-major axis (AU)": 1.448,
        "eccentricity": 0.7008,
        "minimum mass (Jupiter masses)": 11.67,
        "APF jitter (m/s)": 11.0,
        "Lick jitter (m/s)": 14.0,
    }


def test_best_fit_moves_only_the_free_parameters(gaia03):
    index = gaia03["index"]
    start = b_at_its_published_orbit(gaia03)
    start[index["sma2"]], start[index["ecc2"]], start[index["m2"]] = 20.0, 0.3, 0.01
    log_likelihood, best = gaia03["best_fit"](start, {"gamma_apf": 0.0})
    assert np.isfinite(log_likelihood)
    assert np.flatnonzero(best != start).tolist() in ([], [index["gamma_apf"]])


def test_best_fit_refuses_an_orbit_that_cannot_be_real(gaia03):
    start = b_at_its_published_orbit(gaia03)  # c's mass is 0 here, which is refused
    log_likelihood, _ = gaia03["best_fit"](start, {"gamma_apf": 0.0})
    assert log_likelihood == -gaia03["REFUSED"]


def test_seed_orbitize_makes_the_next_run_repeat():
    pytest.importorskip("orbitize")
    from orbitize import sampler

    ht.orbits.seed_orbitize(3)
    first = np.random.random(), sampler.ptemcee.Sampler.keywords["random"].random()
    ht.orbits.seed_orbitize(3)
    second = np.random.random(), sampler.ptemcee.Sampler.keywords["random"].random()
    assert first == second


def test_gaia_01s_hipparcos_file_needs_no_catalog_lookup(monkeypatch):
    """gaia_01 reads the Hipparcos 2 record, which carries the star's solution, so VizieR is never asked."""
    hipparcos = pytest.importorskip("orbitize.hipparcos")
    from orbitize import DATADIR

    def no_lookup(*args, **kwargs):
        msg = "gaia_01 should not need VizieR"
        raise AssertionError(msg)

    monkeypatch.setattr(hipparcos, "Vizier", no_lookup)
    beta_pic = hipparcos.HipparcosLogProb(str(Path(DATADIR) / "H027321.d"), hip_num=27321, num_secondary_bodies=1)
    assert beta_pic.plx0 == pytest.approx(51.44)
    assert beta_pic.solution_type == 5  # a standard five-parameter solution
