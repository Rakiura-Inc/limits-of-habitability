"""The plots the notebooks draw. Each returns the figure so it can go into the hand-in."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .data import EARTH_INSOLATION

LIMIT = 0.8          # Earth radii: the lab's size limit for a sun-like star (Hill et al. 2026)
CREDIT_CATALOG = "Data: Hill et al. 2023, A Catalog of Habitable Zone Exoplanets, VizieR J/AJ/165/34."


def _plain_ticks(ax):
    """Label log axes as plain numbers (0.3, 1, 10), not powers of ten, and drop the minor-tick labels."""
    for a in (ax.xaxis, ax.yaxis):
        if a.get_scale() == "log":
            a.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            a.set_minor_formatter(matplotlib.ticker.NullFormatter())


def catalog(df, color_by: str | None = None, highlight: list | None = None, title: str | None = None,
            numbers: bool = False):
    """Every planet's radius against the starlight it gets, with the 0.8 Earth radii line.

    Grey: radius estimated from the mass. Blue: radius measured. Red ring: mass and radius both
    measured. With color_by="Teff" the points are coloured by their star's temperature instead.
    """
    fig, ax = plt.subplots(figsize=(9, 6), dpi=110)
    x = df.S / EARTH_INSOLATION
    if color_by:
        sc = ax.scatter(x, df.Rad, c=df[color_by], cmap="coolwarm_r", s=18, zorder=2)
        fig.colorbar(sc, ax=ax, label="Star temperature (K)" if color_by == "Teff" else color_by)
    else:
        est, meas = df.f_Rad == 1, df.f_Rad == 0
        # "Measured" mass: from data (f_Mass 0) and not just a limit (l_Mass empty), as the notebook counts it.
        mass_ok = (df.f_Mass == 0) & (df.l_Mass.fillna("").str.strip() == "") if "l_Mass" in df else (df.f_Mass == 0)
        both = meas & mass_ok
        ax.scatter(x[est], df.Rad[est], s=14, c="#c9c9c9", label=f"radius estimated from mass ({est.sum()})")
        ax.scatter(x[meas], df.Rad[meas], s=18, c="#3b6ea5", label=f"radius measured ({meas.sum()})")
        ax.scatter(x[both], df.Rad[both], s=80, facecolors="none", edgecolors="#d1495b", linewidths=1.6,
                   label=f"mass and radius both measured ({both.sum()})")
        ax.legend(loc="center right", fontsize=8.5, framealpha=0.95)
    ax.axhline(LIMIT, color="#d1495b", ls="--", lw=1.2, zorder=0)
    ax.text(0.98, 0.03, "0.8 Earth radii: the lab's limit, modelled for a sun-like star",
            transform=ax.transAxes, ha="right", color="#d1495b", fontsize=9)
    if numbers:                               # label each point with its row number, counting from 1
        for i, (sx, ry) in enumerate(zip(x, df.Rad), start=1):
            ax.annotate(str(i), (sx, ry), xytext=(3, 3), textcoords="offset points", fontsize=7.5)
    for name in highlight or []:
        r = df[df.Planet == name]
        if len(r):
            ax.annotate(name, (float(r.S.iloc[0]) / EARTH_INSOLATION, float(r.Rad.iloc[0])),
                        xytext=(-70, 14), textcoords="offset points", fontsize=8.5,
                        arrowprops=dict(arrowstyle="-", color="#333", lw=0.8))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel("Starlight reaching the planet (Earth = 1; more starlight to the left)")
    ax.set_ylabel("Planet radius (Earth radii)")
    ax.set_yticks([0.5, 0.8, 1, 2, 4, 8, 14])
    _plain_ticks(ax)
    ax.set_ylim(0.6, 16)
    ax.set_title(title or f"Every known planet that spends time in its star's habitable zone ({len(df)})", fontsize=11)
    fig.text(0.01, 0.005, CREDIT_CATALOG, fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def wobbles(df, floor: float, mine: str | None = None):
    """Each planet's predicted wobble, sorted, against an instrument's floor."""
    d = df.sort_values("K").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=110)
    ax.bar(range(len(d)), d.K, color=["#d1495b" if p == mine else "#3b6ea5" for p in d.Planet])
    ax.axhline(floor, color="#333", ls="--", lw=1)
    ax.text(0, floor * 1.08, f"{floor * 100:.0f} cm/s", fontsize=9)
    ax.set_xticks(range(len(d)), d.Planet, rotation=75, fontsize=7.5)
    ax.set_yscale("log"); ax.set_ylabel("Predicted wobble K (m/s)")
    _plain_ticks(ax)
    ax.set_title("Predicted wobble of each planet", fontsize=11)
    fig.text(0.01, 0.005, CREDIT_CATALOG, fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def activity(night, name: str):
    """One star's nightly S-values over the years."""
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=110)
    ax.plot(night.year, night.S, ".", color="#3b6ea5", ms=5)
    ax.set_xlabel("Year"); ax.set_ylabel("S-value (activity)")
    ax.set_title(f"{name}: magnetic activity, one point per night", fontsize=11)
    fig.text(0.01, 0.005, "Data: Isaacson et al. 2024, California Legacy Survey V, VizieR J/ApJS/274/35.",
             fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def periodogram(periods, power, unit: str, title: str, mark: float | None = None, credit: str | None = None):
    """Power against period. A tall peak is a repeating signal at that period. Pass credit for real data."""
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=110)
    ax.plot(periods, power, color="#333", lw=1)
    if mark:
        ax.axvline(mark, color="#d1495b", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlabel(f"Period ({unit})"); ax.set_ylabel("Power")
    _plain_ticks(ax)
    ax.set_title(title, fontsize=11)
    if credit:
        fig.text(0.01, 0.005, credit, fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.04 if credit else 0, 1, 1))
    return fig


def velocities(t, v, err, telescopes=None, P: float | None = None, T0: float | None = None,
               K: float | None = None, offsets: dict | None = None, title: str = "", credit: str | None = None):
    """Velocities against time, and folded on the period when P and T0 are given.

    With K and offsets from a fit, the folded panel shows the fitted orbit and each telescope's
    points shifted by its own offset.
    """
    t, v, err = (np.asarray(x, float) for x in (t, v, err))
    tel = np.asarray(telescopes if telescopes is not None else ["telescope"] * len(t))
    names = sorted(set(tel.tolist()))
    colors = dict(zip(names, ["#3b6ea5", "#d1495b", "#2a9d8f"]))
    ncol = 2 if P else 1
    fig, axes = plt.subplots(1, ncol, figsize=(11 if P else 9, 3.8), dpi=110, squeeze=False)
    ax = axes[0][0]
    for n in names:
        m = tel == n
        ax.errorbar(t[m] - 2457000, v[m], err[m], fmt=".", color=colors[n], label=n, alpha=0.8)
    ax.set_xlabel("Time (BJD - 2457000)"); ax.set_ylabel("Velocity (m/s)"); ax.legend(fontsize=8)
    ax.set_title(title or "Velocities over time", fontsize=10)
    if P:
        ax = axes[0][1]
        phase = ((t - T0) / P + 0.5) % 1 - 0.5
        for n in names:
            m = tel == n
            shift = (offsets or {}).get(n, 0.0)
            ax.errorbar(phase[m], v[m] - shift, err[m], fmt=".", color=colors[n], alpha=0.8)
        if K is not None:
            ph = np.linspace(-0.5, 0.5, 300)
            ax.plot(ph, -K * np.sin(2 * np.pi * ph), color="#333", lw=1.5)
        ax.set_xlabel("Orbital phase (0 = transit)"); ax.set_ylabel("Velocity minus offset (m/s)")
        ax.set_title(f"Folded on P = {P:.4f} d", fontsize=10)
    if credit:
        fig.text(0.01, 0.005, credit, fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.05 if credit else 0, 1, 1))
    return fig


def corner(fit, truths: dict | None = None, credit: str | None = None):
    """Every fitted parameter against every other. Truths, if given, are drawn as lines. Pass credit for real data."""
    import corner as _corner
    labels = fit.names
    t = [truths.get(n) for n in labels] if truths else None
    fig = _corner.corner(fit.chain, labels=labels, truths=t, quantiles=[0.16, 0.5, 0.84],
                         show_titles=True, title_fmt=".2f")
    if credit:
        fig.text(0.01, 0.005, credit, fontsize=7, color="#666")
    return fig


def star_map(summary, mark: dict | None = None, title: str | None = None):
    """Each star placed by colour (across) and average activity (up), coloured by how much it varies.

    mark: {"name": label} stars to circle and label.
    """
    fig, ax = plt.subplots(figsize=(9, 6), dpi=110)
    sc = ax.scatter(summary.BV, summary.logRhk, c=summary.variation.clip(upper=15), cmap="viridis", s=22)
    fig.colorbar(sc, ax=ax, label="How much its activity varies (% of its typical value)")
    for name, label in (mark or {}).items():
        r = summary[summary.Name == name]
        if len(r):
            x, y = float(r.BV.iloc[0]), float(r.logRhk.iloc[0])
            ax.scatter([x], [y], s=140, facecolors="none", edgecolors="#d1495b", linewidths=1.8)
            ax.annotate(label, (x, y), xytext=(10, 8), textcoords="offset points", fontsize=9, color="#d1495b")
    ax.set_xlabel("Colour, B-V (smaller is bluer and hotter, bigger is redder and cooler)")
    ax.set_ylabel("Average activity, log R'HK (higher is more active)")
    ax.set_title(title or f"The {len(summary)} stars you can pick from", fontsize=11)
    fig.text(0.01, 0.005, "Data: Isaacson et al. 2024, California Legacy Survey V, VizieR J/ApJS/274/35.",
             fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def star_map_interactive(summary):
    """The same map, interactive: hover over a point to see the star's name. Needs plotly (Colab has it)."""
    try:
        import plotly.express as px
    except ImportError:
        print("Plotly is not installed here, so use the map above and ht.data.nearest_stars(...) instead.")
        return None
    fig = px.scatter(summary, x="BV", y="logRhk", color=summary.variation.clip(upper=15), hover_name="Name",
                     hover_data={"Nobs": True, "BV": ":.2f", "logRhk": ":.2f"},
                     labels={"BV": "Colour, B-V", "logRhk": "Average activity, log R'HK", "color": "Variation (%)"},
                     title="Hover over a star to see its name")
    fig.add_annotation(text="Data: Isaacson et al. 2024, California Legacy Survey V, VizieR J/ApJS/274/35.",
                       xref="paper", yref="paper", x=0, y=-0.12, showarrow=False, font=dict(size=9, color="#666"))
    return fig


def light_curve(lc, title: str):
    """Brightness against time, one point every 2 minutes."""
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=110)
    ax.plot(lc.time.value, lc.flux.value, ".", ms=1, color="#3b6ea5")
    ax.set_xlabel("Time (BJD - 2457000)"); ax.set_ylabel("Brightness (1 = usual)")
    ax.set_title(title, fontsize=11)
    fig.text(0.01, 0.005, "Data: NASA TESS, SPOC 2-minute light curves, via MAST.", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def transit(hours, flux, bins=None, duration_hours: float | None = None, title: str = ""):
    """The folded transit: every point in grey, the binned average in blue, the transit shaded."""
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=110)
    window = (np.abs(hours) < 3 * duration_hours) if duration_hours else np.ones(len(hours), bool)
    ax.plot(hours[window], flux[window], ".", ms=1.5, color="#c9c9c9", label="every point")
    if bins is not None:
        bx, by = bins
        keep = (np.abs(bx) < 3 * duration_hours) if duration_hours else np.ones(len(bx), bool)
        ax.plot(bx[keep], by[keep], "o", ms=4, color="#3b6ea5", label="15-minute averages")
    if duration_hours:
        ax.axvspan(-duration_hours / 2, duration_hours / 2, color="#f2e8cf", zorder=0, label="the transit")
    ax.set_xlabel("Hours from mid-transit"); ax.set_ylabel("Brightness (1 = usual)")
    ax.set_title(title, fontsize=11); ax.legend(fontsize=8, loc="lower right")
    if bins is not None and len(bins[1]):
        lo, hi = np.nanmin(bins[1]), np.nanmax(bins[1]); pad = (hi - lo) * 0.6 + 1e-4
        ax.set_ylim(lo - pad, hi + pad)
    fig.text(0.01, 0.005, "Data: NASA TESS, SPOC 2-minute light curves, via MAST.", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def planet_map(planets, mark: str | None = None):
    """The planets you can measure, each labelled with its number.

    Across: orbital period. Up: how bright the star looks to TESS (smaller numbers are brighter, so
    brighter stars sit higher). Size: how many velocities the survey took. Colour: one telescope or both.
    """
    p = planets.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=110)
    both = p.telescopes.str.contains("APF") & p.telescopes.str.contains("HIRES")
    size = 15 + 1.2 * p.velocities
    ax.scatter(p.P_days[~both], p.sy_tmag[~both], s=size[~both], c="#3b6ea5", alpha=0.75, label="HIRES only")
    ax.scatter(p.P_days[both], p.sy_tmag[both], s=size[both], c="#d1495b", alpha=0.75, label="HIRES and APF")
    for i, r in p.iterrows():
        ax.annotate(str(i + 1), (r.P_days, r.sy_tmag), xytext=(5, 4), textcoords="offset points", fontsize=8.5,
                    fontweight="bold" if r.planet == mark else "normal")
    if mark is not None and (p.planet == mark).any():
        r = p[p.planet == mark].iloc[0]
        ax.scatter([r.P_days], [r.sy_tmag], s=size[p.planet == mark].iloc[0] * 2.2, facecolors="none",
                   edgecolors="#333", linewidths=1.8)
    ax.set_xscale("log"); ax.invert_yaxis()
    ax.set_xlabel("Orbital period (days)"); ax.set_ylabel("Star brightness to TESS (magnitude; up is brighter)")
    _plain_ticks(ax)
    ax.set_title("The planets you can measure. Bigger points have more velocities.", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower left")
    fig.text(0.01, 0.005, "Data: TESS-Keck Survey release (Polanski et al. 2024) and NASA Exoplanet Archive.",
             fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def joint(c: dict, title: str):
    """The joint fit: the folded transit with its model (left), the folded velocities with theirs (right)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=110)
    a1.plot(c["data_hours"], c["data_flux"], ".", ms=1.5, color="#c9c9c9")
    a1.plot(c["hours"], c["flux"], color="#d1495b", lw=1.6)
    a1.set_xlabel("Hours from mid-transit"); a1.set_ylabel("Brightness minus 1 (parts per thousand)")
    a1.set_title("Transit", fontsize=10)
    colors = {"HIRES": "#3b6ea5", "APF": "#2a9d8f"}
    for t in sorted(set(c["telescope"])):
        m = np.asarray(c["telescope"]) == t
        a2.errorbar(c["data_phase"][m], c["data_rv"][m], c["data_rv_err"][m], fmt=".", color=colors.get(t, "#333"), label=t, alpha=0.8)
    a2.plot(c["phase"], c["rv"], color="#d1495b", lw=1.6)
    a2.set_xlabel("Orbital phase (0 = transit)"); a2.set_ylabel("Velocity minus offset (m/s)")
    a2.set_title("Velocities", fontsize=10); a2.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.text(0.01, 0.005, "Data: NASA TESS (SPOC, via MAST) and the TESS-Keck Survey release (Polanski et al. 2024).",
             fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig


# Earth and Neptune, from the NASA planetary fact sheet: Neptune is 17.1 Earth masses and 3.88 Earth radii.
NEPTUNE = (17.1, 3.88)


def mass_radius(mass, radius, label: str, mass_err=None, radius_err=None):
    """Your planet's mass and radius against what rock could make: the biggest a rocky planet can be
    (no iron core) and an Earth-like mix, from Zeng et al. 2016, with Earth and Neptune for reference."""
    from .fit import rocky_radius
    top_m = max(30, 1.5 * float(mass))          # widen the plot for giant planets
    top_r = max(5, 1.3 * float(radius))
    m = np.logspace(np.log10(0.5), np.log10(top_m), 200)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=110)
    ax.plot(m, rocky_radius(m, 0.0), color="#8d6e63", lw=1.6, label="biggest a rocky planet can be (no iron)")
    ax.plot(m, rocky_radius(m, 0.325), color="#8d6e63", lw=1.2, ls="--", label="rock and iron like Earth")
    ax.axvspan(8, top_m, color="#eeeeee", zorder=0)
    ax.text(8.3, 0.62, "beyond the fitted range", fontsize=8, color="#777")
    ax.scatter([1, NEPTUNE[0]], [1, NEPTUNE[1]], color="#555", s=25, zorder=3)
    ax.annotate("Earth", (1, 1), xytext=(6, -12), textcoords="offset points", fontsize=8.5)
    ax.annotate("Neptune", NEPTUNE, xytext=(6, 4), textcoords="offset points", fontsize=8.5)
    ax.errorbar([mass], [radius], xerr=None if mass_err is None else [[mass_err]], yerr=None if radius_err is None else [[radius_err]],
                fmt="o", color="#d1495b", ms=8, zorder=4, label=label)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.5, top_m); ax.set_ylim(0.6, top_r)
    ax.set_xticks([t for t in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500) if t <= top_m])
    ax.set_yticks([t for t in (0.6, 1, 2, 3, 4, 5, 8, 12, 16, 20) if t <= top_r])
    _plain_ticks(ax)
    ax.set_xlabel("Mass (Earth masses)"); ax.set_ylabel("Radius (Earth radii)")
    ax.set_title("Above the solid line, rock alone cannot make a planet this big", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    fig.text(0.01, 0.005, "Rocky lines: Zeng, Sasselov and Jacobsen 2016, ApJ 819, 127. Earth and Neptune: NASA planetary fact sheet.",
             fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig
