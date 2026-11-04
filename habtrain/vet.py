"""Vetting a TESS candidate: the NASA pipeline's validation reports, and nearby stars in Gaia."""

from __future__ import annotations

import io
import re
import urllib.parse
import urllib.request

import pandas as pd

from .data import DOWNLOAD_ERRORS, Measured

MAST_DOWNLOAD = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
TWO_TESS_PIXELS = 42.0  # arcsec
GAIA_ROW_LIMIT = 50
# When ESA's Gaia archive does not answer, ask GAVO's copy of Gaia DR3 (Heidelberg), which has the
# same source ids, magnitudes and RUWE (checked against ESA for one star, 2026-11-04).
GAVO_TAP = "https://dc.zah.uni-heidelberg.de/tap/sync"
GAVO_TIMEOUT_SECONDS = 60

# Report file names carry the sectors a pipeline run covered, e.g. "-s0014-s0060-".
_SECTORS = re.compile(r"-s(\d{4})-s(\d{4})-")


def _run_order(report: dict) -> tuple[int, int, str]:
    """Sort key for reports: by the last sector, then by how many sectors the run spans.

    So the last report is the latest and widest run.
    """
    sectors = _SECTORS.search(report["file"])
    first, last = 0, 0
    if sectors:
        first, last = int(sectors.group(1)), int(sectors.group(2))
    return (last, last - first, report["file"])


def validation_reports(tic: int):
    """The SPOC data validation reports (PDF) for a star, from MAST: one per pipeline run, newest last."""
    from astroquery.mast import Observations

    observations = Observations.query_criteria(
        target_name=str(tic), obs_collection="TESS", dataproduct_type="timeseries"
    )
    by_file = {}  # one entry per report file: several observations list the same one
    for product in Observations.get_product_list(observations):
        file = str(product["productFilename"])
        if file.endswith("dvr.pdf"):  # the data validation report
            by_file[file] = {"file": file, "url": MAST_DOWNLOAD + str(product["dataURI"])}
    return sorted(by_file.values(), key=_run_order)


def _neighbours_query(table: str, ra: float, dec: float, radius_arcsec: float) -> str:
    # ra, dec and the radius are numbers, so the query cannot be turned into anything else.
    return f"""SELECT source_id, phot_g_mean_mag, ruwe, DISTANCE(POINT({ra},{dec}), POINT(ra,dec))*3600 AS sep_arcsec
            FROM {table} WHERE CONTAINS(POINT(ra,dec), CIRCLE({ra},{dec}, {radius_arcsec / 3600}))=1
            ORDER BY sep_arcsec"""  # noqa: S608


def _neighbours_from_gavo(query: str) -> pd.DataFrame:
    form = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "MAXREC": GAIA_ROW_LIMIT, "QUERY": query}
    data = urllib.parse.urlencode(form).encode()
    with urllib.request.urlopen(GAVO_TAP, data=data, timeout=GAVO_TIMEOUT_SECONDS) as response:
        return pd.read_csv(io.BytesIO(response.read()))


def gaia_neighbours(ra: float, dec: float, radius_arcsec: float = TWO_TESS_PIXELS):
    """Gaia DR3 stars within radius_arcsec (two TESS pixels is 42 arcsec), closest first, with their RUWE.

    Asks ESA's Gaia archive, and GAVO's copy of Gaia DR3 if ESA does not answer (it says so).
    """
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = GAIA_ROW_LIMIT
    try:
        job = Gaia.launch_job(_neighbours_query("gaiadr3.gaia_source", ra, dec, radius_arcsec))
        return job.get_results().to_pandas()
    except DOWNLOAD_ERRORS:  # requests' errors, a 503 among them, are OSErrors
        print("ESA's Gaia archive did not answer, so the neighbours come from GAVO's copy of Gaia DR3.")
    try:
        return _neighbours_from_gavo(_neighbours_query("gaia.dr3lite", ra, dec, radius_arcsec))
    except DOWNLOAD_ERRORS as gavo_error:
        msg = (
            f"Neither ESA's Gaia archive nor GAVO's copy answered ({gavo_error}). "
            "Wait a few minutes and run the cell again."
        )
        raise RuntimeError(msg) from gavo_error


def exofop_page(tic: int) -> str:
    """The ExoFOP page for a star: follow-up imaging, spectra and notes from the TESS follow-up community."""
    return f"https://exofop.ipac.caltech.edu/tess/target.php?id={tic}"


def tic_star(tic: int) -> dict:
    """The star's mass and radius from the TESS Input Catalog (TIC v8.2), each Measured, in solar units."""
    from astroquery.mast import Catalogs

    row = Catalogs.query_criteria(catalog="TIC", ID=int(tic)).to_pandas().iloc[0]
    mass = Measured(float(row["mass"]), float(row["e_mass"]))
    radius = Measured(float(row["rad"]), float(row["e_rad"]))
    return {"mass": mass, "radius": radius}
