"""Fetch and parse the public data releases the notebooks use.

Every file comes from the CDS archive (VizieR) the first time it is needed and is cached in
`data/` next to the notebook. When CDS does not answer, the file comes from its mirror at the
National Astronomical Observatory of Japan instead, and is used only if its checksum matches the
one recorded in `archive_files.csv`. A file from CDS whose checksum does not match still loads,
with one warning line: the archive may have reissued it.

    Habitable zone catalog    Hill et al. 2023, AJ 165, 34            VizieR J/AJ/165/34
    Activity cycles           Isaacson et al. 2024, ApJS 274, 35      VizieR J/ApJS/274/35
    TESS-Keck Survey release  Polanski et al. 2024, ApJS 272, 32      VizieR J/ApJS/272/32
    iota Dra, APF velocities  Hill et al. 2021, AJ 162, 211           VizieR J/AJ/162/211
    iota Dra, Lick velocities Butler et al. 2006, ApJ 646, 505        VizieR J/ApJ/646/505

Star properties come from the NASA Exoplanet Archive, with `planets.csv` as the offline copy.
"""

from __future__ import annotations

import gzip
import hashlib
import http.client
import io
import os
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from .constants import BTJD

CDS = "https://cdsarc.cds.unistra.fr/ftp/"
# The full mirror of the CDS archive in Japan. It serves plain http only, which is why a file from
# it must match its recorded checksum before it is used.
MIRROR = "http://vizier.nao.ac.jp/ftp/cats/"
MIRROR_NAME = "its mirror in Japan (vizier.nao.ac.jp)"
DOWNLOAD_TIMEOUT_SECONDS = 20  # how long to wait for an archive that does not answer
ARCHIVE_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
ARCHIVE_TIMEOUT_SECONDS = 30  # the NASA archive answers a query more slowly than CDS sends a file
CACHE = os.environ.get("HABTRAIN_DATA", "data")
PLANETS_CSV = Path(__file__).with_name("planets.csv")
ARCHIVE_FILES_CSV = Path(__file__).with_name("archive_files.csv")


def _recorded_checksums() -> dict[str, str]:
    """The SHA-256 of every archive file the notebooks read (gzipped files: of the unzipped bytes)."""
    table = pd.read_csv(ARCHIVE_FILES_CSV)
    return dict(zip(table.path, table.sha256, strict=True))


SHA256 = _recorded_checksums()

JD_2000 = 2451544.5  # 2000 January 1, 0h UT
JD_2005 = 2453371.5  # 2005 January 1, 0h UT
ACTIVITY_JD_OFFSET = 2440000.0  # the activity table stores BJD minus this

# How far apart two stars are on the star map: 0.1 in colour counts as much as 0.05 in activity.
COLOUR_SCALE = 0.1
ACTIVITY_SCALE = 0.05


def _cache_path(path: str) -> Path:
    return Path(CACHE) / path.replace("/", "_").removesuffix(".gz")


# What a download that fails raises: no connection, a timeout, an HTTP error, or a cut-off reply.
DOWNLOAD_ERRORS = (OSError, http.client.HTTPException)


def _read_url(url: str) -> bytes:
    """The bytes at `url`, gzip already unpacked when the file name ends in .gz."""
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:  # noqa: S310 - fixed archive URLs
        raw = response.read()
    if url.endswith(".gz"):
        raw = gzip.decompress(raw)
    return raw


def _matches_record(path: str, raw: bytes) -> bool:
    """True when the file has the checksum recorded for it, or when none is recorded."""
    expected = SHA256.get(path)
    return expected is None or hashlib.sha256(raw).hexdigest() == expected


def _from_mirror(path: str, cds_error: Exception) -> bytes:
    """The file from the CDS mirror, used only if it is exactly the recorded file."""
    try:
        raw = _read_url(MIRROR + path)
    except DOWNLOAD_ERRORS as mirror_error:
        msg = (
            f"Could not download {path} from the CDS archive ({cds_error}) or {MIRROR_NAME} ({mirror_error}). "
            "Check your connection and run the cell again."
        )
        raise RuntimeError(msg) from None
    if not _matches_record(path, raw):
        msg = (
            f"The CDS archive did not answer, and the copy of {path} on {MIRROR_NAME} is not the file this "
            "notebook was written against. Wait a few minutes and run the cell again."
        )
        raise RuntimeError(msg)
    print(f"The CDS archive did not answer, so {path} came from {MIRROR_NAME}. It is the same file.")
    return raw


def fetch(path: str) -> bytes:
    """The bytes of one archive file (gzip already unpacked), from the cache, CDS, or the CDS mirror."""
    local = _cache_path(path)
    if local.exists():
        return local.read_bytes()
    try:
        raw = _read_url(CDS + path)
    except DOWNLOAD_ERRORS as cds_error:
        raw = _from_mirror(path, cds_error)
    else:
        if not _matches_record(path, raw):
            print(f"Note: {path} differs from the copy this notebook was written against. It still loads.")
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(raw)
    return raw


def _lines(path: str) -> list[str]:
    return fetch(path).decode("latin-1").splitlines()


def parse_number(cell: str) -> float:
    """A number from a table cell, or nan when the cell is blank or not a number."""
    try:
        return float(cell.strip())
    except ValueError:
        return np.nan


def parse_text(cell: str) -> str:
    return cell.strip()


def parse_flag(cell: str) -> int:
    """1 for a cell holding the flag 1, else 0 (the catalog leaves unflagged cells blank)."""
    return 1 if cell.strip() == "1" else 0


def read_fixed_width(line: str, layout: dict) -> dict:
    """One row of a fixed-width table: layout maps each column name to (start, end, parser)."""
    return {name: parse(line[start:end]) for name, (start, end, parse) in layout.items()}


# ---------------------------------------------------------------- habitable zone catalog

# Byte ranges of the catalog's columns (CDS ReadMe for J/AJ/165/34), in the order the table shows them.
CATALOG_LAYOUT = {
    "Planet": (0, 18, parse_text),
    "Mass": (24, 31, parse_number),
    "f_Mass": (49, 50, parse_flag),
    "l_Mass": (21, 23, parse_text),
    "Rad": (51, 56, parse_number),
    "f_Rad": (71, 72, parse_flag),
    "Per": (73, 81, parse_number),
    "a": (82, 93, parse_number),
    "e": (94, 101, parse_number),
    "S": (102, 108, parse_number),
    "CHZ": (135, 138, parse_number),
    "OHZ": (139, 142, parse_number),
    "TSM": (143, 148, parse_number),
    "K": (149, 154, parse_number),
    "Mstar": (155, 159, parse_number),
    "Rstar": (160, 165, parse_number),
    "Teff": (166, 170, parse_number),
    "Vmag": (190, 196, parse_number),
}


def load_catalog() -> pd.DataFrame:
    """Every known planet that spends time in its star's habitable zone (328 rows).

    f_Mass and f_Rad are 1 where the value was estimated from a mass-radius relation
    (Chen and Kipping 2017) rather than measured. l_Mass is "<=" or ">=" where the mass is only a limit.
    """
    rows = [read_fixed_width(line, CATALOG_LAYOUT) for line in _lines("J/AJ/165/34/table1.dat")]
    return pd.DataFrame(rows)


SMALL_RADIUS_EARTH = 2  # notebook 1's "small": under 2 Earth radii
WHOLE_ORBIT_PERCENT = 100  # CHZ and OHZ are the percentage of the orbit spent in the zone


def small_hz_planets(cat: pd.DataFrame) -> pd.DataFrame:
    """Planets under 2 Earth radii that stay in the optimistic zone their whole orbit."""
    return cat[(cat.Rad < SMALL_RADIUS_EARTH) & (cat.OHZ == WHOLE_ORBIT_PERCENT)]


# ---------------------------------------------------------------- activity cycles

ACTIVITY_SERIES_LAYOUT = {"Name": (0, 12, parse_text), "BJD": (24, 33, parse_number), "S": (34, 41, parse_number)}

ACTIVITY_SUMMARY_LAYOUT = {
    "Name": (0, 12, parse_text),
    "Smed": (28, 34, parse_number),
    "SSD": (35, 42, parse_number),
    "Nobs": (43, 47, parse_number),
    "BV": (48, 53, parse_number),
    "logRhk": (54, 60, parse_number),
}


def load_activity_series() -> pd.DataFrame:
    """52,372 S-value measurements of 710 stars (Keck HIRES, 2005 to 2023; the file starts in late 2004).

    Columns: Name, BJD, S.
    """
    rows = [read_fixed_width(line, ACTIVITY_SERIES_LAYOUT) for line in _lines("J/ApJS/274/35/table3.dat.gz")]
    df = pd.DataFrame(rows, columns=list(ACTIVITY_SERIES_LAYOUT))
    df["BJD"] += ACTIVITY_JD_OFFSET
    return df


def load_activity_summary() -> pd.DataFrame:
    """One line per star: typical activity, how much it varies, colour, and how often it was observed.

    Columns: Name, Smed (typical S-value), SSD (spread of its S-values), Nobs, BV (colour: bigger
    is redder and cooler), logRhk (activity scaled so stars of different colours compare),
    variation (SSD as a percentage of Smed).
    """
    rows = [read_fixed_width(line, ACTIVITY_SUMMARY_LAYOUT) for line in _lines("J/ApJS/274/35/table2.dat")]
    df = pd.DataFrame(rows)
    df["variation"] = 100 * df.SSD / df.Smed
    return df


def nearest_stars(summary: pd.DataFrame, bv: float, logrhk: float, n: int = 5) -> pd.DataFrame:
    """The n stars closest to a spot on the star map (colour bv, activity logrhk)."""
    distance = np.hypot((summary.BV - bv) / COLOUR_SCALE, (summary.logRhk - logrhk) / ACTIVITY_SCALE)
    return summary.assign(distance=distance).nsmallest(n, "distance").drop(columns="distance")


def well_sampled_stars(series: pd.DataFrame, min_obs: int = 45) -> list[str]:
    """Stars with at least `min_obs` observations since 2005: the survey's own cut (285 stars)."""
    counts = series[series.BJD >= JD_2005].groupby("Name").size()
    return sorted(counts[counts >= min_obs].index)


# ---------------------------------------------------------------- TESS-Keck Survey release

RV_LAYOUT = {
    "TKS": (0, 12, parse_text),
    "BJD": (46, 62, parse_number),
    "RV": (63, 86, parse_number),
    "eRV": (87, 106, parse_number),
    "SHK": (107, 123, parse_number),
    "Inst": (131, 136, parse_text),
}

# Positions of the columns used from the survey's fitted table (pipe-separated, J/ApJS/272/32 table6).
# A row with no planet letter describes the whole system (its long-term trends).
TABLE6_COLUMNS = {
    "toi": 0,
    "letter": 1,
    "tks": 2,
    "period": 3,
    "t0": 5,
    "K": 7,
    "K_err": 8,
    "eccentricity": 9,  # "~" when the orbit was fitted as circular
    "linear_trend": 16,
    "curved_trend": 19,
}


def load_rvs() -> pd.DataFrame:
    """All 9,204 TESS-Keck Survey measurements, 9,000 of them with a velocity.

    Columns: TKS, BJD, RV, eRV, SHK, Inst. Rows with no velocity (an activity value only) are dropped.
    """
    rows = [read_fixed_width(line, RV_LAYOUT) for line in _lines("J/ApJS/272/32/tksxxrvs.dat")]
    return pd.DataFrame([row for row in rows if not np.isnan(row["RV"])])


def _table6() -> list[dict]:
    """The rows of the survey's fitted table, each cell stripped, keyed by TABLE6_COLUMNS."""
    rows = []
    for line in _lines("J/ApJS/272/32/table6.dat"):
        cells = line.split("|")
        row = {}
        for name, position in TABLE6_COLUMNS.items():
            row[name] = ""
            if position < len(cells):  # a short line leaves the last cells out
                row[name] = cells[position].strip()
        row["toi"] = row["toi"].replace("TOI", "").strip()
        rows.append(row)
    return rows


def _nonzero(cell: str) -> bool:
    value = parse_number(cell)
    return not np.isnan(value) and value != 0


def _has_trend(system_row: dict | None) -> bool:
    """True when the system was fitted with a linear or curved long-term trend."""
    return system_row is not None and (_nonzero(system_row["linear_trend"]) or _nonzero(system_row["curved_trend"]))


def orbit(planet: str) -> dict:
    """The published period and transit time for a planet such as "TOI-669 b".

    Returns only what a fit needs as fixed inputs (TKS star id, P in days, T0 as BJD). The
    published wobble and mass are deliberately not returned: they are the answer key.
    """
    toi, letter = planet.replace("TOI-", "").split()
    for row in _table6():
        if row["toi"] == toi and row["letter"] == letter:
            return {"planet": planet, "TKS": row["tks"], "P": float(row["period"]), "T0": float(row["t0"]) + BTJD}
    msg = f"{planet} is not in the TESS-Keck Survey fitted table."
    raise KeyError(msg)


def _telescope_counts(instruments: pd.Series) -> str:
    """How many velocities each telescope took, such as "HIRES 37 + APF 12"."""
    return " + ".join(f"{name} {count}" for name, count in instruments.value_counts().items())


def single_planet_systems() -> pd.DataFrame:
    """Planets alone in their system, on circular orbits, with no long-term trend.

    The systems you can fit with the simple model in notebook 4, with how many velocities each has.
    Sorted with the clearest signals first. Holds no published wobble or mass.
    """
    rows = _table6()
    system_rows = {row["tks"]: row for row in rows if not row["letter"]}
    planet_rows = [row for row in rows if row["letter"]]
    planets_per_star = Counter(row["tks"] for row in planet_rows)
    rvs = load_rvs()
    velocities = rvs.groupby("TKS").size()
    telescopes = rvs.groupby("TKS").Inst.agg(_telescope_counts)
    chosen = []
    for row in planet_rows:
        tks = row["tks"]
        if planets_per_star[tks] != 1 or row["eccentricity"] != "~" or _has_trend(system_rows.get(tks)):
            continue
        wobble, wobble_err = parse_number(row["K"]), parse_number(row["K_err"])
        strength = 0.0
        if wobble_err:
            strength = wobble / wobble_err
        chosen.append({
            "planet": f"TOI-{row['toi']} {row['letter']}",
            "TKS": tks,
            "P_days": float(row["period"]),
            "velocities": int(velocities.get(tks, 0)),
            "telescopes": telescopes.get(tks, ""),
            "_strength": strength,
        })
    df = pd.DataFrame(chosen).sort_values("_strength", ascending=False).drop(columns="_strength")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- NASA Exoplanet Archive


def planets() -> pd.DataFrame:
    """The planets the notebooks let you measure, with what you can know before measuring them.

    One planet per star, on a circular orbit with no long-term trend, from the TESS-Keck Survey
    release, joined with star properties from the NASA Exoplanet Archive (read 2026-10-25). Holds no
    published radius, wobble or mass: those are what you measure.
    """
    return pd.read_csv(PLANETS_CSV)


def _from_planets_csv(planet: str, column: str) -> float:
    """One value for a planet from the package's own table (planets.csv); KeyError if it is not there."""
    table = planets()
    row = table[table.planet == planet]
    if len(row) == 0:
        msg = f"{planet} is not in habtrain's planets.csv either, so there is no {column} for it. Check its name."
        raise KeyError(msg)
    return float(row[column].iloc[0])


def _archive_csv(columns: str, planet: str) -> str:
    """The NASA Exoplanet Archive's answer, as CSV text, for some columns of one planet's row."""
    name = planet.replace("'", "''")  # quote the name the way ADQL quotes strings
    query = f"select {columns} from pscomppars where pl_name='{name}'"  # noqa: S608 - a read-only public archive
    url = ARCHIVE_TAP + urllib.parse.urlencode({"query": query, "format": "csv"})
    with urllib.request.urlopen(url, timeout=ARCHIVE_TIMEOUT_SECONDS) as response:  # noqa: S310 - a fixed https URL
        return response.read().decode()


def _archive_value(planet: str, column: str) -> float:
    """One number for a planet from the archive, or from planets.csv when the archive cannot give it."""
    try:
        lines = _archive_csv(column, planet).strip().splitlines()
    except DOWNLOAD_ERRORS:
        print(f"The NASA Exoplanet Archive did not answer, so {column} for {planet} comes from planets.csv.")
        return _from_planets_csv(planet, column)
    if len(lines) < 2 or not lines[1].strip():  # noqa: PLR2004 - a header line, then the value
        print(f"The NASA Exoplanet Archive has no {column} for {planet}, so it comes from planets.csv.")
        return _from_planets_csv(planet, column)
    return float(lines[1])


def star_radius(planet: str) -> float:
    """The host star's radius in solar radii (NASA Exoplanet Archive; fallback: planets.csv)."""
    return _archive_value(planet, "st_rad")


def transit_duration(planet: str) -> float:
    """How long the planet's transit lasts, in hours (NASA Exoplanet Archive; fallback: planets.csv)."""
    return _archive_value(planet, "pl_trandur")


def star_mass(planet: str) -> float:
    """The host star's mass in solar masses, from the NASA Exoplanet Archive (fallback: planets.csv)."""
    mass = _archive_value(planet, "st_mass")
    if np.isnan(mass):
        msg = f"Neither the NASA Exoplanet Archive nor planets.csv has a star mass for {planet}."
        raise RuntimeError(msg)
    return mass


class Measured(NamedTuple):
    """A published value and its uncertainty, one number for both sides."""

    value: float
    uncertainty: float


def _symmetric(err_plus: float, err_minus: float) -> float:
    """One uncertainty from an upper and a lower one: the average of their sizes."""
    return float((abs(err_plus) + abs(err_minus)) / 2)


def star(planet: str) -> dict:
    """The host star's TIC number, position, and mass and radius with uncertainties (NASA Exoplanet Archive).

    Returns {"tic", "ra", "dec", "mass", "radius"}; mass and radius are Measured, in solar units.
    """
    columns = "tic_id,ra,dec,st_mass,st_masserr1,st_masserr2,st_rad,st_raderr1,st_raderr2"
    try:
        answer = _archive_csv(columns, planet)
    except DOWNLOAD_ERRORS as error:
        msg = (
            f"Could not reach the NASA Exoplanet Archive for {planet}'s star ({error}). "
            "Wait a minute and run the cell again."
        )
        raise RuntimeError(msg) from None
    row = pd.read_csv(io.StringIO(answer)).iloc[0]
    return {
        "tic": int(str(row.tic_id).replace("TIC", "").strip()),
        "ra": float(row.ra),
        "dec": float(row.dec),
        "mass": Measured(float(row.st_mass), _symmetric(row.st_masserr1, row.st_masserr2)),
        "radius": Measured(float(row.st_rad), _symmetric(row.st_raderr1, row.st_raderr2)),
    }
