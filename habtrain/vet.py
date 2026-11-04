"""Vetting a TESS candidate: the NASA pipeline's validation reports, and nearby stars in Gaia."""

from __future__ import annotations

import re

MAST_DOWNLOAD = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
TWO_TESS_PIXELS = 42.0  # arcsec
GAIA_ROW_LIMIT = 50

# Report file names carry the sectors a pipeline run covered, e.g. "-s0014-s0060-".
_SECTORS = re.compile(r"-s(\d{4})-s(\d{4})-")


def _run_order(file: str) -> tuple[int, int, str]:
    """Sort key for reports: by the last sector, then by how many sectors the run spans.

    So the last report is the latest and widest run.
    """
    m = _SECTORS.search(file)
    first, last = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    return (last, last - first, file)


def validation_reports(tic: int):
    """The SPOC data validation reports (PDF) for a star, from MAST: one per pipeline run, newest last."""
    from astroquery.mast import Observations

    obs = Observations.query_criteria(target_name=str(tic), obs_collection="TESS", dataproduct_type="timeseries")
    products = Observations.get_product_list(obs)
    reports = products[[str(f).endswith("dvr.pdf") for f in products["productFilename"]]]
    by_file = {}
    for r in reports:
        file = str(r["productFilename"])
        by_file[file] = {"file": file, "url": MAST_DOWNLOAD + str(r["dataURI"])}
    return sorted(by_file.values(), key=lambda report: _run_order(report["file"]))


def gaia_neighbours(ra: float, dec: float, radius_arcsec: float = TWO_TESS_PIXELS):
    """Gaia DR3 stars within radius_arcsec (two TESS pixels is 42 arcsec), closest first, with their RUWE."""
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = GAIA_ROW_LIMIT
    # ra, dec and the radius are numbers, so the query cannot be turned into anything else.
    q = f"""SELECT source_id, phot_g_mean_mag, ruwe, DISTANCE(POINT({ra},{dec}), POINT(ra,dec))*3600 AS sep_arcsec
            FROM gaiadr3.gaia_source WHERE CONTAINS(POINT(ra,dec), CIRCLE({ra},{dec}, {radius_arcsec / 3600}))=1
            ORDER BY sep_arcsec"""  # noqa: S608
    return Gaia.launch_job(q).get_results().to_pandas()


def exofop_page(tic: int) -> str:
    """The ExoFOP page for a star: follow-up imaging, spectra and notes from the TESS follow-up community."""
    return f"https://exofop.ipac.caltech.edu/tess/target.php?id={tic}"


def tic_star(tic: int) -> dict:
    """The star's mass and radius from the TESS Input Catalog (TIC v8.2), with uncertainties, in solar units."""
    from astroquery.mast import Catalogs

    r = Catalogs.query_criteria(catalog="TIC", ID=int(tic)).to_pandas().iloc[0]
    return {"mass": (float(r["mass"]), float(r["e_mass"])), "radius": (float(r["rad"]), float(r["e_rad"]))}
