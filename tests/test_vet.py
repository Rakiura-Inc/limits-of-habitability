"""Vetting: validation reports in run order, the Gaia query, and the follow-up page."""

import pandas as pd
import pytest

import habtrain as ht


def test_reports_sort_by_last_sector_then_by_how_many_sectors_they_span():
    files = [
        "tess-s0014-s0060-dvr.pdf",
        "tess-s0001-s0060-dvr.pdf",
        "tess-s0014-s0014-dvr.pdf",
        "other-dvr.pdf",
    ]
    reports = [{"file": file} for file in files]
    assert [report["file"] for report in sorted(reports, key=ht.vet._run_order)] == [
        "other-dvr.pdf",
        "tess-s0014-s0014-dvr.pdf",
        "tess-s0014-s0060-dvr.pdf",
        "tess-s0001-s0060-dvr.pdf",
    ]


def test_validation_reports_keep_one_pdf_per_run_newest_last(monkeypatch):
    mast = pytest.importorskip("astroquery.mast")
    products = pd.DataFrame({
        "productFilename": ["a-s0002-s0002-dvr.pdf", "a-s0001-s0001-dvr.pdf", "a-lc.fits", "a-s0002-s0002-dvr.pdf"],
        "dataURI": ["mast:2", "mast:1", "mast:lc", "mast:2again"],
    })
    monkeypatch.setattr(mast.Observations, "query_criteria", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        mast.Observations, "get_product_list", lambda obs: products.to_records() if obs["target_name"] == "123" else []
    )
    reports = ht.vet.validation_reports(123)
    assert [r["file"] for r in reports] == ["a-s0001-s0001-dvr.pdf", "a-s0002-s0002-dvr.pdf"]
    assert reports[1]["url"] == ht.vet.MAST_DOWNLOAD + "mast:2again"


def test_gaia_neighbours_asks_for_stars_within_the_radius(monkeypatch):
    gaia = pytest.importorskip("astroquery.gaia")
    asked = []

    class Job:
        def get_results(self):
            return self

        def to_pandas(self):
            return pd.DataFrame({"source_id": [1]})

    monkeypatch.setattr(gaia.Gaia, "launch_job", lambda q: asked.append(q) or Job())
    assert ht.vet.gaia_neighbours(10.0, -5.0, radius_arcsec=36.0).source_id.tolist() == [1]
    assert "CIRCLE(10.0,-5.0, 0.01)" in asked[0]


def test_tic_star_reads_mass_and_radius_with_uncertainties(monkeypatch):
    mast = pytest.importorskip("astroquery.mast")

    class Table:
        def to_pandas(self):
            return pd.DataFrame({"mass": [0.9], "e_mass": [0.1], "rad": [0.8], "e_rad": [0.05]})

    monkeypatch.setattr(mast.Catalogs, "query_criteria", lambda **query: Table())
    star = ht.vet.tic_star(42)
    assert star == {"mass": (0.9, 0.1), "radius": (0.8, 0.05)}
    assert star["mass"].uncertainty == 0.1


def test_exofop_page_links_to_the_star():
    assert ht.vet.exofop_page(42) == "https://exofop.ipac.caltech.edu/tess/target.php?id=42"
