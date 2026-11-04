"""Reading the archive tables: each column from its place in the line, and the cuts the notebooks make."""

import gzip
import hashlib

import numpy as np
import pandas as pd
import pytest

import habtrain as ht

from .conftest import place

CATALOG = "J/AJ/165/34/table1.dat"
SERIES = "J/ApJS/274/35/table3.dat.gz"
SUMMARY = "J/ApJS/274/35/table2.dat"
RVS = "J/ApJS/272/32/tksxxrvs.dat"
TABLE6 = "J/ApJS/272/32/table6.dat"


def catalog_line(planet, mass, rad, ohz, *, mass_limit="", mass_estimated="", rad_estimated="", teff="5772"):
    return place({
        (0, 18): planet.ljust(18),
        (21, 23): mass_limit,
        (24, 31): mass,
        (49, 50): mass_estimated,
        (51, 56): rad,
        (71, 72): rad_estimated,
        (73, 81): "365.25",
        (82, 93): "1.0",
        (94, 101): "0.02",
        (102, 108): "1361",
        (135, 138): "100",
        (139, 142): ohz,
        (143, 148): "",
        (149, 154): "0.09",
        (155, 159): "1.00",
        (160, 165): "1.00",
        (166, 170): teff,
        (190, 196): "4.8",
    })


def test_catalog_reads_each_column_from_its_place(cache):
    cache.put(
        CATALOG,
        "\n".join([
            catalog_line("Earthlike b", "1.00", "1.0", "100"),
            catalog_line("Puffy c", "30", "3.1", "40", mass_limit="<=", rad_estimated="1", teff="3500"),
        ]),
    )
    cat = ht.data.load_catalog()
    assert list(cat.columns) == list(ht.data.CATALOG_LAYOUT)
    earth, puffy = cat.iloc[0], cat.iloc[1]
    assert earth.Planet == "Earthlike b"
    assert (earth.Mass, earth.Rad, earth.Per, earth.S, earth.OHZ, earth.Teff) == (1.0, 1.0, 365.25, 1361, 100, 5772)
    assert (earth.f_Mass, earth.f_Rad, earth.l_Mass) == (0, 0, "")
    assert (puffy.f_Rad, puffy.l_Mass) == (1, "<=")
    assert np.isnan(earth.TSM)  # a blank cell reads as nan


def test_small_hz_planets_keeps_small_planets_always_in_the_zone():
    cat = pd.DataFrame({"Planet": ["a", "b", "c", "d"], "Rad": [1.0, 1.99, 2.0, 1.5], "OHZ": [100, 100, 100, 99]})
    assert ht.data.small_hz_planets(cat).Planet.tolist() == ["a", "b"]


def test_activity_series_turns_the_stored_date_into_a_full_julian_date(cache):
    line = place({(0, 12): "HD 1000".ljust(12), (24, 33): "13371.5", (34, 41): "0.170"})
    cache.put(SERIES, line + "\n")
    series = ht.data.load_activity_series()
    assert list(series.columns) == ["Name", "BJD", "S"]
    assert series.Name[0] == "HD 1000"
    assert series.BJD[0] == ht.data.JD_2005
    assert series.S[0] == pytest.approx(0.17)


def test_activity_summary_adds_how_much_each_star_varies(cache):
    line = place({
        (0, 12): "HD 2000".ljust(12),
        (28, 34): "0.200",
        (35, 42): "0.010",
        (43, 47): "120",
        (48, 53): "0.65",
        (54, 60): "-4.90",
    })
    cache.put(SUMMARY, line)
    star = ht.data.load_activity_summary().iloc[0]
    assert (star.Name, star.Nobs, star.BV, star.logRhk) == ("HD 2000", 120, 0.65, -4.9)
    assert star.variation == pytest.approx(5.0)


def test_nearest_stars_weighs_colour_and_activity_by_their_spread():
    summary = pd.DataFrame({
        "Name": ["colour off", "activity off", "close"],
        "BV": [0.80, 0.70, 0.71],
        "logRhk": [-4.90, -4.80, -4.91],
    })
    # 0.1 in colour counts as 1, and so does 0.05 in activity: "activity off" is 2 away, "colour off" 1.
    assert ht.data.nearest_stars(summary, 0.70, -4.90, n=3).Name.tolist() == ["close", "colour off", "activity off"]


def test_well_sampled_stars_counts_only_nights_since_2005():
    since = ht.data.JD_2005
    series = pd.DataFrame({
        "Name": ["busy"] * 45 + ["early"] * 45 + ["quiet"] * 44,
        "BJD": [since + i for i in range(45)] + [since - 1 - i for i in range(45)] + [since + i for i in range(44)],
    })
    assert ht.data.well_sampled_stars(series) == ["busy"]
    assert ht.data.well_sampled_stars(series, min_obs=44) == ["busy", "quiet"]


def test_nightly_averages_each_night_and_counts_in_years():
    start = ht.data.JD_2000  # midnight, so 0.1 and 0.2 days later is the same night
    series = pd.DataFrame({
        "Name": ["A", "A", "A", "B"],
        "BJD": [start + 0.1, start + 0.2, start + 365.25, start],
        "S": [0.1, 0.3, 0.5, 9.0],
    })
    night = ht.data.nightly(series, "A")
    assert night.columns.tolist() == ["year", "S", "n"]
    assert night.S.tolist() == pytest.approx([0.2, 0.5])
    assert night.n.tolist() == [2, 1]
    assert night.year.tolist() == pytest.approx([2000.0 + 0.15 / 365.25, 2001.0])


def rv_line(tks, bjd, rv, inst):
    return place({
        (0, 12): tks.ljust(12),
        (46, 62): bjd,
        (63, 86): rv,
        (87, 106): "1.5",
        (107, 123): "0.15",
        (131, 136): inst.ljust(5),
    })


def test_load_rvs_drops_rows_with_no_velocity(cache):
    cache.put(RVS, "\n".join([rv_line("T001", "2459000.1", "3.2", "HIRES"), rv_line("T001", "2459001.1", "", "APF")]))
    rvs = ht.data.load_rvs()
    assert rvs.columns.tolist() == ["TKS", "BJD", "RV", "eRV", "SHK", "Inst"]
    assert len(rvs) == 1
    assert (rvs.TKS[0], rvs.RV[0], rvs.Inst[0]) == ("T001", 3.2, "HIRES")


def table6_row(toi, letter, tks, period="", t0="", k="", k_err="", ecc="~", trend="", curve=""):
    cells = [""] * 21
    cells[0], cells[1], cells[2], cells[3], cells[5] = f"TOI {toi}", letter, tks, period, t0
    cells[7], cells[8], cells[9], cells[16], cells[19] = k, k_err, ecc, trend, curve
    return "|".join(cells)


@pytest.fixture
def survey(cache):
    """A small survey: two lone circular planets, one in a pair, one eccentric, one with a trend."""
    cache.put(
        TABLE6,
        "\n".join([
            table6_row("100", "b", "T000100", "3.5", "1500.25", "4.0", "2.0"),
            table6_row("200", "b", "T000200", "7.0", "1600.5", "9.0", "1.0"),
            table6_row("300", "b", "T000300", "2.0", "1700.0", "5.0", "1.0"),
            table6_row("300", "c", "T000300", "9.0", "1700.0", "5.0", "1.0"),
            table6_row("400", "b", "T000400", "4.0", "1800.0", "5.0", "1.0", ecc="0.3"),
            table6_row("500", "", "T000500", trend="0.01"),
            table6_row("500", "b", "T000500", "5.0", "1900.0", "5.0", "1.0"),
        ]),
    )
    cache.put(
        RVS,
        "\n".join([
            rv_line("T000100", "2459000.1", "1.0", "HIRES"),
            rv_line("T000100", "2459000.2", "1.0", "HIRES"),
            rv_line("T000100", "2459000.3", "1.0", "APF"),
            rv_line("T000200", "2459000.1", "1.0", "HIRES"),
        ]),
    )


@pytest.mark.usefixtures("survey")
def test_orbit_gives_the_period_and_transit_time_as_full_julian_dates():
    assert ht.data.orbit("TOI-100 b") == {"planet": "TOI-100 b", "TKS": "T000100", "P": 3.5, "T0": 2458500.25}


@pytest.mark.usefixtures("survey")
def test_orbit_of_an_unknown_planet_says_so():
    with pytest.raises(KeyError, match="TOI-999 b is not in"):
        ht.data.orbit("TOI-999 b")


@pytest.mark.usefixtures("survey")
def test_single_planet_systems_keeps_lone_circular_planets_without_trends_clearest_first():
    systems = ht.data.single_planet_systems()
    assert systems.planet.tolist() == ["TOI-200 b", "TOI-100 b"]  # K / error: 9 before 2
    assert systems.columns.tolist() == ["planet", "TKS", "P_days", "velocities", "telescopes"]
    toi100 = systems.iloc[1]
    assert (toi100.velocities, toi100.telescopes) == (3, "HIRES 2 + APF 1")


def test_planets_table_holds_what_you_know_before_measuring():
    table = ht.data.planets()
    assert {"planet", "TKS", "P_days", "st_rad", "st_mass", "pl_trandur"} <= set(table.columns)
    assert not {"pl_rade", "pl_bmasse", "K"} & set(table.columns)  # the answers are not in it
    assert table.planet.is_unique


def test_star_values_fall_back_to_the_bundled_table_offline(capsys):
    row = ht.data.planets().iloc[0]
    assert ht.data.star_radius(row.planet) == row.st_rad
    assert ht.data.transit_duration(row.planet) == row.pl_trandur
    assert ht.data.star_mass(row.planet) == row.st_mass
    assert "did not answer, so st_rad" in capsys.readouterr().out  # it says where the value came from


def test_a_planet_in_neither_the_archive_nor_the_table_says_so():
    with pytest.raises(KeyError, match=r"not in habtrain's planets\.csv either"):
        ht.data.star_radius("TOI-0 z")


def test_archive_answer_is_used_and_the_name_is_quoted(archive):
    archive["exoplanetarchive"] = b"st_rad\n1.234\n"
    assert ht.data.star_radius("O'Neill b") == 1.234
    assert "O%27%27Neill" in archive.urls[0]  # the quote is doubled, as ADQL wants


def test_star_reads_the_tic_number_and_averages_the_uncertainties(archive):
    archive["exoplanetarchive"] = (
        b"tic_id,ra,dec,st_mass,st_masserr1,st_masserr2,st_rad,st_raderr1,st_raderr2\n"
        b"TIC 12345,10.5,-20.25,1.1,0.1,-0.3,0.9,0.02,-0.04\n"
    )
    star = ht.data.star("TOI-1 b")
    assert star["tic"] == 12345
    assert (star["ra"], star["dec"]) == (10.5, -20.25)
    assert star["mass"] == pytest.approx((1.1, 0.2))
    assert star["radius"] == pytest.approx((0.9, 0.03))


def test_star_offline_says_to_try_again():
    with pytest.raises(RuntimeError, match="Wait a minute"):
        ht.data.star("TOI-1 b")


def test_fetch_downloads_unzips_caches_and_warns_about_a_changed_file(archive, cache, capsys, monkeypatch):
    text = b"a line\n"
    archive["J/X/1/t.dat.gz"] = gzip.compress(text)
    monkeypatch.setitem(ht.data.SHA256, "J/X/1/t.dat.gz", "0" * 64)
    assert ht.data.fetch("J/X/1/t.dat.gz") == text
    assert "differs from the copy" in capsys.readouterr().out
    assert (cache.path / "J_X_1_t.dat").read_bytes() == text
    archive.clear()  # the second read comes from the cache
    assert ht.data.fetch("J/X/1/t.dat.gz") == text


def test_fetch_is_quiet_when_the_checksum_matches(archive, cache, capsys, monkeypatch):
    archive["J/X/1/u.dat"] = b"same"
    monkeypatch.setitem(ht.data.SHA256, "J/X/1/u.dat", hashlib.sha256(b"same").hexdigest())
    ht.data.fetch("J/X/1/u.dat")
    assert capsys.readouterr().out == ""


@pytest.mark.usefixtures("cache")
def test_fetch_offline_says_to_check_the_connection():
    with pytest.raises(RuntimeError, match="Check your connection"):
        ht.data.fetch("J/X/1/missing.dat")


def test_fetch_uses_the_mirror_when_cds_does_not_answer_and_says_so(archive, cache, capsys, monkeypatch):
    archive["vizier.nao.ac.jp/ftp/cats/J/X/1/m.dat"] = b"same"
    monkeypatch.setitem(ht.data.SHA256, "J/X/1/m.dat", hashlib.sha256(b"same").hexdigest())
    assert ht.data.fetch("J/X/1/m.dat") == b"same"
    assert archive.urls == [ht.data.CDS + "J/X/1/m.dat", ht.data.MIRROR + "J/X/1/m.dat"]
    assert "came from its mirror in Japan" in capsys.readouterr().out
    assert (cache.path / "J_X_1_m.dat").read_bytes() == b"same"


@pytest.mark.usefixtures("cache")
def test_fetch_refuses_a_mirror_copy_that_is_not_the_recorded_file(archive, monkeypatch):
    archive["vizier.nao.ac.jp/ftp/cats/J/X/1/m.dat"] = b"changed"
    monkeypatch.setitem(ht.data.SHA256, "J/X/1/m.dat", hashlib.sha256(b"same").hexdigest())
    with pytest.raises(RuntimeError, match="is not the file"):
        ht.data.fetch("J/X/1/m.dat")


def test_every_file_the_notebooks_read_has_a_recorded_checksum():
    assert len(ht.data.SHA256) == 8
    assert all(len(checksum) == 64 for checksum in ht.data.SHA256.values())
