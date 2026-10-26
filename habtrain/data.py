"""Fetch and parse the three public data releases the notebooks use.

Every file comes from the CDS archive (VizieR) the first time it is needed and is cached in
`data/` next to the notebook. A file whose checksum does not match the one recorded here still
loads, with one warning line: the archive may have reissued it.

    Habitable zone catalog    Hill et al. 2023, AJ 165, 34            VizieR J/AJ/165/34
    Activity cycles           Isaacson et al. 2024, ApJS 274, 35      VizieR J/ApJS/274/35
    TESS-Keck Survey release  Polanski et al. 2024, ApJS 272, 32      VizieR J/ApJS/272/32
"""
from __future__ import annotations

import gzip
import hashlib
import io
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

CDS = ["https://cdsarc.cds.unistra.fr/ftp/"]
CACHE = os.environ.get("HABTRAIN_DATA", "data")

# Checksums of the files as fetched on 2026-10-25 (gzipped files: of the unzipped bytes).
SHA256 = {
    "J/AJ/165/34/table1.dat": "38a21f64a660d92e10812f847bc68e50951ebcba0d3e8e4b3299f989e8fc2fbb",
    "J/ApJS/274/35/table2.dat": "ff11036825b1915ca1228f8c8cdb6c17553989bda4e96a7c278e64595ec167be",
    "J/ApJS/274/35/table3.dat.gz": "a522f68c01db21c4adae071c7c4cf103ccb32273d5ebbcdb78b9aaa1488cd2d9",
    "J/ApJS/274/35/table4.dat": "6976ae9dd0dda35d0afe0237c515c6a15465ef869978ffae86c2c86ef3f1fdf5",
    "J/ApJS/272/32/tksxxrvs.dat": "d4cc7a7c3ee23030ce7a84f4226cf29b6601ee0e2171b7b37adf3d9f62362273",
    "J/ApJS/272/32/table6.dat": "ad9fd91d095887dbdc3dfe025c0320beb2264084baa0dedf71fadddfb8e67e26",
}



SUN_TEFF = 5772        # K
EARTH_INSOLATION = 1361  # W/m2


def fetch(path: str) -> bytes:
    """The bytes of one archive file (gzip already unpacked), from the cache or the archive."""
    local = os.path.join(CACHE, path.replace("/", "_").removesuffix(".gz"))
    if os.path.exists(local):
        with open(local, "rb") as fh:
            return fh.read()
    raw, last = None, None
    for base in CDS:
        try:
            with urllib.request.urlopen(base + path, timeout=60) as r:
                raw = r.read()
            break
        except Exception as exc:  # keep the error to report it
            last = exc
    if raw is None:
        raise RuntimeError(f"Could not download {path} from the CDS archive ({last}). "
                           "Check your connection and run the cell again.")
    if path.endswith(".gz"):
        raw = gzip.decompress(raw)
    want = SHA256.get(path)
    if want and hashlib.sha256(raw).hexdigest() != want:
        print(f"Note: {path} differs from the copy this notebook was written against. It still loads.")
    os.makedirs(CACHE, exist_ok=True)
    with open(local, "wb") as fh:
        fh.write(raw)
    return raw


def _num(s: str):
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------- habitable zone catalog

def load_catalog() -> pd.DataFrame:
    """Every known planet that spends time in its star's habitable zone (328 rows).

    f_Mass and f_Rad are 1 where the value was estimated from a mass-radius relation
    (Chen and Kipping 2017) rather than measured. l_Mass is "<=" or ">=" where the mass is only a limit.
    """
    text = fetch("J/AJ/165/34/table1.dat").decode("latin-1")
    rows = []
    for line in text.splitlines():
        L = line.ljust(196)
        rows.append({
            "Planet": L[0:18].strip(),
            "Mass": _num(L[24:31]), "f_Mass": 1 if L[49:50].strip() == "1" else 0,
            "l_Mass": L[21:23].strip(),           # "<=" or ">=" when the mass is only a limit
            "Rad": _num(L[51:56]), "f_Rad": 1 if L[71:72].strip() == "1" else 0,
            "Per": _num(L[73:81]), "a": _num(L[82:93]), "e": _num(L[94:101]),
            "S": _num(L[102:108]),
            "CHZ": _num(L[135:138]), "OHZ": _num(L[139:142]),
            "TSM": _num(L[143:148]), "K": _num(L[149:154]),
            "Mstar": _num(L[155:159]), "Rstar": _num(L[160:165]), "Teff": _num(L[166:170]),
            "Vmag": _num(L[190:196]),
        })
    return pd.DataFrame(rows)


def small_hz_planets(cat: pd.DataFrame) -> pd.DataFrame:
    """Planets under 2 Earth radii that stay in the optimistic zone their whole orbit."""
    return cat[(cat.Rad < 2) & (cat.OHZ == 100)]


# ---------------------------------------------------------------- activity cycles

def load_activity_series() -> pd.DataFrame:
    """52,372 S-value measurements of 710 stars (Keck HIRES, 2005 to 2023; the file starts in late 2004). Columns: Name, BJD, S."""
    text = fetch("J/ApJS/274/35/table3.dat.gz").decode("latin-1")
    names, bjd, s = [], [], []
    for line in text.splitlines():
        names.append(line[0:12].strip())
        bjd.append(_num(line[24:33]) + 2440000.0)
        s.append(_num(line[34:41]))
    return pd.DataFrame({"Name": names, "BJD": bjd, "S": s})


def load_activity_summary() -> pd.DataFrame:
    """One line per star: typical activity, how much it varies, colour, and how often it was observed.

    Columns: Name, Smed (typical S-value), SSD (spread of its S-values), Nobs, BV (colour: bigger
    is redder and cooler), logRhk (activity scaled so stars of different colours compare),
    variation (SSD as a percentage of Smed).
    """
    text = fetch("J/ApJS/274/35/table2.dat").decode("latin-1")
    rows = []
    for L in text.splitlines():
        L = L.ljust(66)
        rows.append({"Name": L[0:12].strip(), "Smed": _num(L[28:34]), "SSD": _num(L[35:42]),
                     "Nobs": _num(L[43:47]), "BV": _num(L[48:53]), "logRhk": _num(L[54:60])})
    df = pd.DataFrame(rows)
    df["variation"] = 100 * df.SSD / df.Smed
    return df


def nearest_stars(summary: pd.DataFrame, bv: float, logrhk: float, n: int = 5) -> pd.DataFrame:
    """The n stars closest to a spot on the star map (colour bv, activity logrhk)."""
    d = np.hypot((summary.BV - bv) / 0.1, (summary.logRhk - logrhk) / 0.05)
    return summary.assign(distance=d).nsmallest(n, "distance").drop(columns="distance")


def well_sampled_stars(series: pd.DataFrame, min_obs: int = 45) -> list[str]:
    """Stars with at least `min_obs` observations since 2005: the survey's own cut (285 stars)."""
    since = series[series.BJD >= 2453371.5]
    counts = since.groupby("Name").size()
    return sorted(counts[counts >= min_obs].index)


def nightly(series: pd.DataFrame, name: str) -> pd.DataFrame:
    """One star's S-values averaged per night, with time in years. Columns: year, S, n."""
    one = series[series.Name == name]
    night = np.floor(one.BJD.values + 0.5)
    g = pd.DataFrame({"night": night, "BJD": one.BJD.values, "S": one.S.values}).groupby("night")
    out = pd.DataFrame({"BJD": g.BJD.mean(), "S": g.S.mean(), "n": g.size()}).reset_index(drop=True)
    out["year"] = 2000.0 + (out.BJD - 2451544.5) / 365.25
    return out[["year", "S", "n"]]


# ---------------------------------------------------------------- TESS-Keck Survey release

def load_rvs() -> pd.DataFrame:
    """All 9,204 TESS-Keck Survey measurements, 9,000 of them with a velocity. Columns: TKS, BJD, RV, eRV, SHK, Inst.

    Rows with no velocity (an activity value only) are dropped.
    """
    text = fetch("J/ApJS/272/32/tksxxrvs.dat").decode("latin-1")
    rows = []
    for line in text.splitlines():
        rv = _num(line[63:86])
        if np.isnan(rv):
            continue
        rows.append({"TKS": line[0:12].strip(), "BJD": _num(line[46:62]), "RV": rv,
                     "eRV": _num(line[87:106]), "SHK": _num(line[107:123]), "Inst": line[131:136].strip()})
    return pd.DataFrame(rows)


def _nonzero(x: str) -> bool:
    v = _num(x)
    return not np.isnan(v) and v != 0.0


def _table6() -> list[list[str]]:
    return [L.rstrip("\n").split("|") for L in fetch("J/ApJS/272/32/table6.dat").decode("latin-1").splitlines()]


def orbit(planet: str) -> dict:
    """The published period and transit time for a planet such as "TOI-669 b".

    Returns only what a fit needs as fixed inputs (TKS star id, P in days, T0 as BJD). The
    published wobble and mass are deliberately not returned: they are the answer key.
    """
    toi, letter = planet.replace("TOI-", "").split()
    for c in _table6():
        if c[0].replace("TOI", "").strip() == toi and c[1].strip() == letter:
            return {"planet": planet, "TKS": c[2].strip(), "P": float(c[3]), "T0": float(c[5]) + 2457000.0}
    raise KeyError(f"{planet} is not in the TESS-Keck Survey fitted table.")


def single_planet_systems() -> pd.DataFrame:
    """Planets alone in their system, on circular orbits, with no long-term trend.

    The systems you can fit with the simple model in notebook 4, with how many velocities each has.
    Sorted with the clearest signals first. Holds no published wobble or mass.
    """
    rows, sysrow, count = [], {}, {}
    t6 = _table6()
    for c in t6:
        if not c[1].strip():
            sysrow[c[2].strip()] = c
    for c in t6:
        if c[1].strip():
            count[c[2].strip()] = count.get(c[2].strip(), 0) + 1
    rvs = load_rvs()
    n = rvs.groupby("TKS").size()
    inst = rvs.groupby("TKS").Inst.agg(lambda s: " + ".join(f"{k} {v}" for k, v in s.value_counts().items()))
    for c in t6:
        tks = c[2].strip()
        if not c[1].strip() or count[tks] != 1 or c[9].strip() != "~":
            continue
        s = sysrow.get(tks)
        if s and (_nonzero(s[16]) or _nonzero(s[19])):   # a linear or curved long-term trend
            continue
        k, ek = _num(c[7]), _num(c[8])
        rows.append({"planet": f"TOI-{c[0].replace('TOI', '').strip()} {c[1].strip()}", "TKS": tks,
                     "P_days": float(c[3]), "velocities": int(n.get(tks, 0)), "telescopes": inst.get(tks, ""),
                     "_strength": k / ek if ek else 0.0})
    df = pd.DataFrame(rows).sort_values("_strength", ascending=False).drop(columns="_strength")
    return df.reset_index(drop=True)


SUN_IN_EARTH_RADII = 109.08


def _from_table(planet: str, column: str) -> float:
    """One value for a planet from the package's own table (planets.csv), or nan if it is not there."""
    t = planets()
    row = t[t.planet == planet]
    return float(row[column].iloc[0]) if len(row) and pd.notna(row[column].iloc[0]) else float("nan")


def star_radius(planet: str) -> float:
    """The host star's radius in solar radii (NASA Exoplanet Archive; fallback: planets.csv)."""
    q = f"select st_rad from pscomppars where pl_name='{planet}'"
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?" + urllib.parse.urlencode({"query": q, "format": "csv"})
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return float(r.read().decode().strip().splitlines()[1])
    except Exception:
        return _from_table(planet, "st_rad")


def transit_duration(planet: str) -> float:
    """How long the planet's transit lasts, in hours (NASA Exoplanet Archive; fallback: planets.csv)."""
    q = f"select pl_trandur from pscomppars where pl_name='{planet}'"
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?" + urllib.parse.urlencode({"query": q, "format": "csv"})
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return float(r.read().decode().strip().splitlines()[1])
    except Exception:
        return _from_table(planet, "pl_trandur")


def planets() -> pd.DataFrame:
    """The planets the notebooks let you measure, with what you can know before measuring them.

    One planet per star, on a circular orbit with no long-term trend, from the TESS-Keck Survey
    release, joined with star properties from the NASA Exoplanet Archive (read 2026-10-25). Holds no
    published radius, wobble or mass: those are what you measure.
    """
    here = os.path.dirname(__file__)
    return pd.read_csv(os.path.join(here, "planets.csv"))


def star(planet: str) -> dict:
    """The host star's TIC number, position, and mass and radius with uncertainties (NASA Exoplanet Archive).

    Returns {"tic", "ra", "dec", "mass": (value, uncertainty), "radius": (value, uncertainty)} in solar units.
    """
    q = ("select tic_id,ra,dec,st_mass,st_masserr1,st_masserr2,st_rad,st_raderr1,st_raderr2 "
         f"from pscomppars where pl_name='{planet}'")
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?" + urllib.parse.urlencode({"query": q, "format": "csv"})
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            row = pd.read_csv(io.StringIO(r.read().decode())).iloc[0]
    except Exception as exc:
        raise RuntimeError(f"Could not reach the NASA Exoplanet Archive for {planet}'s star ({exc}). "
                           "Wait a minute and run the cell again.") from None
    tic = int(str(row.tic_id).replace("TIC", "").strip())
    return {"tic": tic, "ra": float(row.ra), "dec": float(row.dec),
            "mass": (float(row.st_mass), float((abs(row.st_masserr1) + abs(row.st_masserr2)) / 2)),
            "radius": (float(row.st_rad), float((abs(row.st_raderr1) + abs(row.st_raderr2)) / 2))}


def star_mass(planet: str) -> float:
    """The host star's mass in solar masses, from the NASA Exoplanet Archive (fallback: planets.csv)."""
    q = f"select st_mass from pscomppars where pl_name='{planet}'"
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?" + urllib.parse.urlencode({"query": q, "format": "csv"})
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return float(r.read().decode().strip().splitlines()[1])
    except Exception:
        mass = _from_table(planet, "st_mass")
        if not np.isnan(mass):
            return mass
        raise RuntimeError(f"Could not reach the NASA Exoplanet Archive for {planet}'s star mass.")
