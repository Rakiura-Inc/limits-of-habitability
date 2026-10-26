"""Vetting a TESS candidate: the NASA pipeline's validation reports, and nearby stars in Gaia."""
from __future__ import annotations

import re


def validation_reports(tic: int):
    """The SPOC data validation reports (PDF) for a star, from MAST: one per pipeline run, newest last."""
    from astroquery.mast import Observations
    obs = Observations.query_criteria(target_name=str(tic), obs_collection="TESS", dataproduct_type="timeseries")
    prods = Observations.get_product_list(obs)
    reports = prods[[str(f).endswith("dvr.pdf") for f in prods["productFilename"]]]
    out = []
    for r in reports:
        out.append({"file": str(r["productFilename"]),
                    "url": "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + str(r["dataURI"])})
    def run_order(o):
        # File names carry the sectors a run covered, e.g. "-s0014-s0060-". Order by the last sector,
        # then by how many sectors the run spans, so the last report is the latest and widest run.
        m = re.search(r"-s(\d{4})-s(\d{4})-", o["file"])
        first, last = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        return (last, last - first, o["file"])
    return sorted({o["file"]: o for o in out}.values(), key=run_order)


def gaia_neighbours(ra: float, dec: float, radius_arcsec: float = 42.0):
    """Gaia DR3 stars within radius_arcsec (two TESS pixels is 42 arcsec), closest first, with their RUWE."""
    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = 50
    q = f"""SELECT source_id, phot_g_mean_mag, ruwe, DISTANCE(POINT({ra},{dec}), POINT(ra,dec))*3600 AS sep_arcsec
            FROM gaiadr3.gaia_source WHERE CONTAINS(POINT(ra,dec), CIRCLE({ra},{dec}, {radius_arcsec/3600}))=1
            ORDER BY sep_arcsec"""
    return Gaia.launch_job(q).get_results().to_pandas()


def exofop_page(tic: int) -> str:
    """The ExoFOP page for a star: follow-up imaging, spectra and notes from the TESS follow-up community."""
    return f"https://exofop.ipac.caltech.edu/tess/target.php?id={tic}"


def tic_star(tic: int) -> dict:
    """The star's mass and radius from the TESS Input Catalog (TIC v8.2), with uncertainties, in solar units."""
    from astroquery.mast import Catalogs
    r = Catalogs.query_criteria(catalog="TIC", ID=int(tic)).to_pandas().iloc[0]
    return {"mass": (float(r["mass"]), float(r["e_mass"])), "radius": (float(r["rad"]), float(r["e_rad"]))}
