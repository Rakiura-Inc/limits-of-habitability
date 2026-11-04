"""Plumbing for the Gaia and radial velocity project.

The catalog of accelerations, iota Draconis's public velocities and published orbit, a VizieR
server that answers, and a seed for orbitize!'s sampler. The orbit fits themselves are written out
in the notebooks with orbitize!'s own calls.

orbitize! (Blunt et al. 2020, AJ 159, 89) is an optional install: pip install "habtrain[orbits]".
Everything here imports it only when needed.

    Catalog of accelerations   Brandt 2021, ApJS 254, 42 (HGCA_vEDR3.fits, kept in orbitize!'s data folder)
    iota Dra, APF velocities   Hill et al. 2021, AJ 162, 211          VizieR J/AJ/162/211
    iota Dra, Lick velocities  Butler et al. 2006, ApJ 646, 505       VizieR J/ApJ/646/505
"""

from __future__ import annotations

import functools
import types
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DOWNLOAD_ERRORS, DOWNLOAD_TIMEOUT_SECONDS, fetch, parse_number, parse_text, read_fixed_width

MJUP_IN_MSUN = 9.547919e-4  # Jupiter's mass in solar masses (IAU 2015)
MJD_OFFSET = 2400000.5  # modified Julian date = Julian date minus this
HGCA_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/254/42/HGCA_vEDR3.fits"
PUBLISHED_B_CSV = Path(__file__).parent / "published" / "iota_dra_b.csv"

# orbitize! looks a star up in the Hipparcos catalog (VizieR I/311) through astroquery. These are
# the VizieR servers to ask, in turn: CDS, then its mirrors at Harvard and in Cape Town.
VIZIER_SERVERS = ("vizier.cds.unistra.fr", "vizier.cfa.harvard.edu", "vizier.idia.ac.za")
VIZIER_TEST_QUERY = "https://{server}/viz-bin/votable?-source=I/311/hip2&-out=HIP&-out.max=1"

# iota Draconis (HIP 75458). Star mass from asteroseismology and parallax from Gaia EDR3, both as
# Hill et al. 2021 and the catalog of accelerations give them.
IOTA_DRA = {"hip": 75458, "star_mass": 1.54, "star_mass_err": 0.09, "parallax": 32.5239, "parallax_err": 0.131}

# iota Dra b's period, periastron time and the STAR's argument of periastron (radians), from
# Hill et al. 2021, table 3. orbitize! wants the companion's angle, which is this plus pi.
B_PERIOD_DAYS = 510.855
B_PERIASTRON_JD = 2452014.19
B_OMEGA_STAR = 1.5696

# Butler et al. 2006, table 1: byte ranges from its CDS ReadMe (J/ApJ/646/505). One row per velocity.
LICK_LAYOUT = {
    "star": (0, 11, parse_text),
    "JD": (12, 26, parse_number),
    "RV": (27, 34, parse_number),  # m/s
    "error": (35, 40, parse_number),  # m/s
}
APF_COLUMNS = 3  # Hill et al. 2021, table 1: JD, velocity and its error (m/s), separated by spaces


def iota_dra_tables() -> pd.DataFrame:
    """Every public velocity of iota Dra, as the two papers publish them.

    Columns: jd (Julian date), rv_ms and error_ms (m/s), telescope ("lick" or "apf"). gaia_03
    turns them into the units orbitize! reads.
    """
    rows = []
    for line in fetch("J/ApJ/646/505/table1.dat").decode().splitlines():  # Lick, 2000 to 2005, many stars
        lick = read_fixed_width(line, LICK_LAYOUT)
        if lick["star"] == "iota Dra":
            rows.append((lick["JD"], lick["RV"], lick["error"], "lick"))
    for line in fetch("J/AJ/162/211/table1.dat").decode().splitlines():  # APF, 2018 to 2021, iota Dra only
        parts = line.split()
        if len(parts) < APF_COLUMNS:
            continue  # a blank line
        rows.append((float(parts[0]), float(parts[1]), float(parts[2]), "apf"))
    return pd.DataFrame(rows, columns=["jd", "rv_ms", "error_ms", "telescope"])


def published_iota_dra_b() -> pd.DataFrame:
    """iota Dra b from velocities alone, as Hill et al. 2021 table 3 gives it.

    Columns: quantity, value, minus, plus, unit, source.
    """
    return pd.read_csv(PUBLISHED_B_CSV)


def _plain_value(cell):
    """A FITS cell as a plain Python value where it has one."""
    if hasattr(cell, "item"):
        return cell.item()
    return cell


def hgca_row(hip: int) -> dict | None:
    """One star's entry in the catalog of accelerations (EDR3 edition), or None if it is not there."""
    from astropy.io import fits
    from orbitize import DATADIR

    path = Path(DATADIR) / "HGCA_vEDR3.fits"
    if not path.exists():  # orbitize! fetches the catalog on first use; do the same here
        print("Downloading the catalog of accelerations (about 22 MB), once...")
        urllib.request.urlretrieve(HGCA_URL, path)
    with fits.open(path, ignore_missing_simple=True) as catalog:
        table = catalog[1].data
        rows = table[table["hip_id"] == hip]
        if len(rows) == 0:
            return None
        return {name: _plain_value(rows[0][name]) for name in table.columns.names}


def _vizier_answers(server: str) -> bool:
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed https URLs
            VIZIER_TEST_QUERY.format(server=server), timeout=DOWNLOAD_TIMEOUT_SECONDS
        ):
            return True
    except DOWNLOAD_ERRORS:
        return False


def _first_answering_vizier() -> str | None:
    for server in VIZIER_SERVERS:
        if _vizier_answers(server):
            return server
    return None


def use_answering_vizier() -> str:
    """Send astroquery's VizieR queries, and so orbitize!'s, to the first server in VIZIER_SERVERS that answers.

    Returns the server's name. Prints a line when it is the mirror, not CDS.
    """
    from astroquery.vizier import VizierClass

    server = _first_answering_vizier()
    if server is None:
        msg = "Neither VizieR at CDS nor its mirrors answered. Wait a few minutes and run the cell again."
        raise RuntimeError(msg)
    if server != VIZIER_SERVERS[0]:
        print(f"VizieR at CDS did not answer, so the Hipparcos catalog comes from its mirror {server}.")
    # Each new astroquery query takes its server from this default, which astroquery fixes when it is imported.
    VizierClass.__init__.__kwdefaults__["vizier_server"] = server
    return server


# The three measurements of a star's motion, in time order: (column suffix, label, colour).
_MOTIONS = (
    ("hip", "Hipparcos, 1991", "#3b6ea5"),
    ("hg", "average, 1991 to 2016", "#555"),
    ("gaia", "Gaia, 2016", "#d1495b"),
)


def plot_proper_motions(row: dict, name: str):
    """A star's motion across the sky, three ways.

    Hipparcos (1991), Gaia (2016), and the average over the 25 years between them. If they
    disagree, something is pulling the star.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=110)
    for ax, component, direction in zip(axes, ("pmra", "pmdec"), ("east-west", "north-south"), strict=True):
        average = row[f"{component}_hg"]  # show differences from the 25-year average
        for i, (suffix, _, colour) in enumerate(_MOTIONS):
            value, error = row[f"{component}_{suffix}"], row[f"{component}_{suffix}_error"]
            ax.errorbar([i], [value - average], yerr=[error], fmt="o", ms=7, capsize=4, color=colour)
        ax.axhline(0, color="#999", lw=0.8, ls="--")
        ax.set_xticks(range(len(_MOTIONS)), [label for _, label, _ in _MOTIONS], fontsize=8.5)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylabel("Motion minus the 25-year average (mas per year)")
        ax.set_title(f"{name}: {direction} motion", fontsize=10.5)
    fig.text(
        0.01,
        0.005,
        "Data: Brandt 2021, Hipparcos-Gaia Catalog of Accelerations (EDR3), ApJS 254, 42.",
        fontsize=7,
        color="#666",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def seed_orbitize(seed: int) -> None:
    """Make the next orbitize! MCMC run give the same answer every time, so it matches the worked answer.

    orbitize! draws its walkers' starting points from numpy's shared random numbers, which
    np.random.seed fixes. With two or more temperatures it hands the run to ptemcee, which makes
    its own random numbers and has no seed option, so this gives it seeded ones. Call it just
    before sampler.MCMC(...).
    """
    import ptemcee
    from orbitize import sampler

    np.random.seed(seed)
    seeded = np.random.RandomState(seed)
    sampler.ptemcee = types.SimpleNamespace(Sampler=functools.partial(ptemcee.Sampler, random=seeded))
