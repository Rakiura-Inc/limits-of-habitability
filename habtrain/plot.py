"""The plots the notebooks draw. Each returns the figure so it can go into the hand-in."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from .constants import BTJD, EARTH_INSOLATION_W_M2
from .fit import EARTH_CORE_FRACTION, rocky_radius

LIMIT = 0.8  # Earth radii: the lab's size limit for a sun-like star (Hill et al. 2026)
ZENG_MAX_MASS_EARTH = 8  # Zeng et al. 2016 fitted their rocky radii up to about 8 Earth masses

# Earth and Neptune, from the NASA planetary fact sheet: Neptune is 17.1 Earth masses and 3.88 Earth radii.
NEPTUNE = (17.1, 3.88)

BLUE, RED, GREEN, BROWN = "#3b6ea5", "#d1495b", "#2a9d8f", "#8d6e63"
LIGHT_GREY, GREY, DARK = "#c9c9c9", "#666", "#333"
SAND = "#f2e8cf"

CREDIT_CATALOG = "Data: Hill et al. 2023, A Catalog of Habitable Zone Exoplanets, VizieR J/AJ/165/34."
CREDIT_ACTIVITY = "Data: Isaacson et al. 2024, California Legacy Survey V, VizieR J/ApJS/274/35."
CREDIT_TESS = "Data: NASA TESS, SPOC 2-minute light curves, via MAST."
CREDIT_SURVEY = "Data: TESS-Keck Survey release (Polanski et al. 2024) and NASA Exoplanet Archive."
CREDIT_JOINT = "Data: NASA TESS (SPOC, via MAST) and the TESS-Keck Survey release (Polanski et al. 2024)."
CREDIT_ROCK = (
    "Rocky lines: Zeng, Sasselov and Jacobsen 2016, ApJ 819, 127. Earth and Neptune: NASA planetary fact sheet."
)


def _credit(fig, text: str) -> None:
    """Write where the data came from in small grey type at the bottom left."""
    fig.text(0.01, 0.005, text, fontsize=7, color=GREY)


def _plain_number(value, _position) -> str:
    return f"{value:g}"


def _plain_ticks(ax):
    """Label log axes as plain numbers (0.3, 1, 10), not powers of ten, and drop the minor-tick labels."""
    for axis in (ax.xaxis, ax.yaxis):
        if axis.get_scale() == "log":
            axis.set_major_formatter(ticker.FuncFormatter(_plain_number))
            axis.set_minor_formatter(ticker.NullFormatter())


def _mass_measured(df):
    """True where a planet's mass is measured: from data (f_Mass 0) and not just a limit (l_Mass empty).

    This is how the notebook counts it.
    """
    if "l_Mass" not in df:
        return df.f_Mass == 0
    return (df.f_Mass == 0) & (df.l_Mass.fillna("").str.strip() == "")


def _catalog_points(fig, ax, df) -> None:
    """Grey: radius estimated from the mass. Blue: radius measured. Red ring: mass and radius both measured."""
    starlight = df.S / EARTH_INSOLATION_W_M2
    estimated, measured = df.f_Rad == 1, df.f_Rad == 0
    both = measured & _mass_measured(df)
    ax.scatter(
        starlight[estimated],
        df.Rad[estimated],
        s=14,
        c=LIGHT_GREY,
        label=f"radius estimated from mass ({estimated.sum()})",
    )
    ax.scatter(starlight[measured], df.Rad[measured], s=18, c=BLUE, label=f"radius measured ({measured.sum()})")
    ax.scatter(
        starlight[both],
        df.Rad[both],
        s=80,
        facecolors="none",
        edgecolors=RED,
        linewidths=1.6,
        label=f"mass and radius both measured ({both.sum()})",
    )
    ax.legend(loc="center right", fontsize=8.5, framealpha=0.95)


def _catalog_limit_line(ax) -> None:
    ax.axhline(LIMIT, color=RED, ls="--", lw=1.2, zorder=0)
    ax.text(
        0.98,
        0.03,
        "0.8 Earth radii: the lab's limit, modelled for a sun-like star",
        transform=ax.transAxes,
        ha="right",
        color=RED,
        fontsize=9,
    )


def _point_numbers(ax, df) -> None:
    """Label each planet with its row number, counting from 1."""
    starlight = df.S / EARTH_INSOLATION_W_M2
    for number, (x, y) in enumerate(zip(starlight, df.Rad, strict=True), start=1):
        ax.annotate(str(number), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7.5)


def _label_planet(ax, df, name: str) -> None:
    row = df[df.Planet == name]
    if len(row) == 0:
        return  # a planet from another table: nothing to label here
    position = (float(row.S.iloc[0]) / EARTH_INSOLATION_W_M2, float(row.Rad.iloc[0]))
    ax.annotate(
        name,
        position,
        xytext=(-70, 14),
        textcoords="offset points",
        fontsize=8.5,
        arrowprops={"arrowstyle": "-", "color": DARK, "lw": 0.8},
    )


def _catalog_axes(fig, ax, df, title: str | None) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Starlight reaching the planet (Earth = 1; more starlight to the left)")
    ax.set_ylabel("Planet radius (Earth radii)")
    ax.set_yticks([0.5, 0.8, 1, 2, 4, 8, 14])
    _plain_ticks(ax)
    ax.set_ylim(0.6, 16)
    ax.set_title(title or f"Every known planet that spends time in its star's habitable zone ({len(df)})", fontsize=11)
    _credit(fig, CREDIT_CATALOG)
    fig.tight_layout(rect=(0, 0.02, 1, 1))


def catalog(df, highlight: str | None = None):
    """Every planet's radius against the starlight it gets, with the 0.8 Earth radii line.

    Grey: radius estimated from the mass. Blue: radius measured. Red ring: mass and radius both
    measured. `highlight` names one planet to label.
    """
    fig, ax = plt.subplots(figsize=(9, 6), dpi=110)
    _catalog_points(fig, ax, df)
    _catalog_limit_line(ax)
    if highlight is not None:
        _label_planet(ax, df, highlight)
    _catalog_axes(fig, ax, df, None)
    return fig


def catalog_by_temperature(df, title: str):
    """The catalog plot with each planet coloured by its star's surface temperature."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=110)
    points = ax.scatter(df.S / EARTH_INSOLATION_W_M2, df.Rad, c=df.Teff, cmap="coolwarm_r", s=18, zorder=2)
    fig.colorbar(points, ax=ax, label="Star temperature (K)")
    _catalog_limit_line(ax)
    _catalog_axes(fig, ax, df, title)
    return fig


def numbered_catalog(df, title: str, highlight: str | None = None):
    """The catalog plot with each planet numbered from 1, in table order, as the menu numbers them."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=110)
    _catalog_points(fig, ax, df)
    _catalog_limit_line(ax)
    _point_numbers(ax, df)
    if highlight is not None:
        _label_planet(ax, df, highlight)
    _catalog_axes(fig, ax, df, title)
    return fig


def wobbles(df, floor: float, mine: str | None = None):
    """Each planet's predicted wobble, sorted, against an instrument's floor."""
    by_wobble = df.sort_values("K").reset_index(drop=True)
    colors = []
    for planet in by_wobble.Planet:
        if planet == mine:
            colors.append(RED)
        else:
            colors.append(BLUE)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=110)
    ax.bar(range(len(by_wobble)), by_wobble.K, color=colors)
    ax.axhline(floor, color=DARK, ls="--", lw=1)
    ax.text(0, floor * 1.08, f"{floor * 100:.0f} cm/s", fontsize=9)
    ax.set_xticks(range(len(by_wobble)), by_wobble.Planet, rotation=75, fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylabel("Predicted wobble K (m/s)")
    _plain_ticks(ax)
    ax.set_title("Predicted wobble of each planet", fontsize=11)
    _credit(fig, CREDIT_CATALOG)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def periodogram(periods, power, *, unit: str, title: str, mark: float | None = None, credit: str | None = None):
    """Power against period. A tall peak is a repeating signal at that period. Pass credit for real data."""
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=110)
    ax.plot(periods, power, color=DARK, lw=1)
    if mark:
        ax.axvline(mark, color=RED, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(f"Period ({unit})")
    ax.set_ylabel("Power")
    _plain_ticks(ax)
    ax.set_title(title, fontsize=11)
    if credit:
        _credit(fig, credit)
        fig.tight_layout(rect=(0, 0.04, 1, 1))  # room for the credit line
    else:
        fig.tight_layout(rect=(0, 0, 1, 1))
    return fig


def velocities(
    table,
    *,
    orbit: dict | None = None,
    K: float | None = None,
    offsets: dict | None = None,
    title: str = "",
    credit: str | None = None,
):
    """Velocities against time, and folded on the orbit's period when `orbit` ({"P", "T0"}) is given.

    `table` has the survey's columns: BJD, RV and eRV (m/s), Inst (the telescope). With K and the
    telescopes' offsets from a fit, the folded panel shows the fitted orbit and each telescope's
    points shifted by its own offset.
    """
    times, velocity, error = (table[column].to_numpy(float) for column in ("BJD", "RV", "eRV"))
    telescope = table.Inst.to_numpy()
    names = sorted(set(telescope.tolist()))
    colors = dict(zip(names, [BLUE, RED, GREEN], strict=False))  # up to three telescopes
    if orbit is None:
        fig, axes = plt.subplots(1, 1, figsize=(9, 3.8), dpi=110, squeeze=False)
    else:  # a second panel for the folded velocities
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=110, squeeze=False)
    ax = axes[0][0]
    for name in names:
        mine = telescope == name
        ax.errorbar(times[mine] - BTJD, velocity[mine], error[mine], fmt=".", color=colors[name], label=name, alpha=0.8)
    ax.set_xlabel("Time (BJD - 2457000)")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend(fontsize=8)
    ax.set_title(title or "Velocities over time", fontsize=10)
    if orbit is not None:
        _folded_velocities(axes[0][1], table, orbit, K=K, offsets=offsets or {}, colors=colors)
    if credit:
        _credit(fig, credit)
        fig.tight_layout(rect=(0, 0.05, 1, 1))  # room for the credit line
    else:
        fig.tight_layout(rect=(0, 0, 1, 1))
    return fig


def _folded_velocities(ax, table, orbit: dict, *, K: float | None, offsets: dict, colors: dict) -> None:
    period = orbit["P"]
    phase = ((table.BJD.to_numpy(float) - orbit["T0"]) / period + 0.5) % 1 - 0.5
    telescope = table.Inst.to_numpy()
    for name in sorted(set(telescope.tolist())):
        mine = telescope == name
        shifted = table.RV.to_numpy(float)[mine] - offsets.get(name, 0.0)
        ax.errorbar(phase[mine], shifted, table.eRV.to_numpy(float)[mine], fmt=".", color=colors[name], alpha=0.8)
    if K is not None:
        model_phase = np.linspace(-0.5, 0.5, 300)
        ax.plot(model_phase, -K * np.sin(2 * np.pi * model_phase), color=DARK, lw=1.5)
    ax.set_xlabel("Orbital phase (0 = transit)")
    ax.set_ylabel("Velocity minus offset (m/s)")
    ax.set_title(f"Folded on P = {period:.4f} d", fontsize=10)


def corner(samples, truths: dict | None = None, credit: str | None = None):
    """Every fitted parameter against every other, one column of `samples` each.

    `truths` ({column: value}) draws the values you made as lines. Pass credit for real data.
    """
    import corner as corner_plot

    truth_values = None
    if truths:
        truth_values = [truths.get(name) for name in samples.columns]
    fig = corner_plot.corner(
        samples.to_numpy(),
        labels=list(samples.columns),
        truths=truth_values,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".2f",
    )
    if credit:
        _credit(fig, credit)
    return fig


def star_map(summary, mark: dict | None = None, title: str | None = None):
    """Each star placed by colour (across) and average activity (up), coloured by how much it varies.

    mark: {"name": label} stars to circle and label.
    """
    fig, ax = plt.subplots(figsize=(9, 6), dpi=110)
    variation = summary.variation.clip(upper=15)  # a few stars vary far more; keep the colour scale useful
    points = ax.scatter(summary.BV, summary.logRhk, c=variation, cmap="viridis", s=22)
    fig.colorbar(points, ax=ax, label="How much its activity varies (% of its typical value)")
    for name, label in (mark or {}).items():
        star = summary[summary.Name == name]
        if len(star) == 0:
            continue  # a star not on this map
        colour, activity = float(star.BV.iloc[0]), float(star.logRhk.iloc[0])
        ax.scatter([colour], [activity], s=140, facecolors="none", edgecolors=RED, linewidths=1.8)
        ax.annotate(label, (colour, activity), xytext=(10, 8), textcoords="offset points", fontsize=9, color=RED)
    ax.set_xlabel("Colour, B-V (smaller is bluer and hotter, bigger is redder and cooler)")
    ax.set_ylabel("Average activity, log R'HK (higher is more active)")
    ax.set_title(title or f"The {len(summary)} stars you can pick from", fontsize=11)
    _credit(fig, CREDIT_ACTIVITY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def star_map_interactive(summary):
    """The same map, interactive: hover over a point to see the star's name. Needs plotly (Colab has it)."""
    try:
        import plotly.express as px
    except ImportError:
        print("Plotly is not installed here, so use the map above and ht.data.nearest_stars(...) instead.")
        return None
    fig = px.scatter(
        summary,
        x="BV",
        y="logRhk",
        color=summary.variation.clip(upper=15),
        hover_name="Name",
        hover_data={"Nobs": True, "BV": ":.2f", "logRhk": ":.2f"},
        labels={"BV": "Colour, B-V", "logRhk": "Average activity, log R'HK", "color": "Variation (%)"},
        title="Hover over a star to see its name",
    )
    fig.add_annotation(
        text=CREDIT_ACTIVITY,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.12,
        showarrow=False,
        font={"size": 9, "color": GREY},
    )
    return fig


def _near_transit(hours, duration_hours: float | None):
    """True for points within three transit durations of mid-transit; every point when there is no duration."""
    if not duration_hours:
        return np.ones(len(hours), bool)
    return np.abs(hours) < 3 * duration_hours


def transit(hours, flux, *, bins=None, duration_hours: float | None = None, title: str = ""):
    """The folded transit: every point in grey, the binned average in blue, the transit shaded."""
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=110)
    window = _near_transit(hours, duration_hours)
    ax.plot(hours[window], flux[window], ".", ms=1.5, color=LIGHT_GREY, label="every point")
    if bins is not None:
        bx, by = bins
        keep = _near_transit(bx, duration_hours)
        ax.plot(bx[keep], by[keep], "o", ms=4, color=BLUE, label="15-minute averages")
    if duration_hours:
        ax.axvspan(-duration_hours / 2, duration_hours / 2, color=SAND, zorder=0, label="the transit")
    ax.set_xlabel("Hours from mid-transit")
    ax.set_ylabel("Brightness (1 = usual)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    if bins is not None and len(bins[1]):
        lo, hi = np.nanmin(bins[1]), np.nanmax(bins[1])
        pad = (hi - lo) * 0.6 + 1e-4
        ax.set_ylim(lo - pad, hi + pad)
    _credit(fig, CREDIT_TESS)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def planet_map(planets, mark: str | None = None):
    """The planets you can measure, each labelled with its number.

    Across: orbital period. Up: how bright the star looks to TESS (smaller numbers are brighter, so
    brighter stars sit higher). Size: how many velocities the survey took. Colour: one telescope or both.
    """
    table = planets.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=110)
    both = table.telescopes.str.contains("APF") & table.telescopes.str.contains("HIRES")
    size = 15 + 1.2 * table.velocities  # point area grows with the number of velocities
    ax.scatter(table.P_days[~both], table.sy_tmag[~both], s=size[~both], c=BLUE, alpha=0.75, label="HIRES only")
    ax.scatter(table.P_days[both], table.sy_tmag[both], s=size[both], c=RED, alpha=0.75, label="HIRES and APF")
    for number, planet in enumerate(table.itertuples(), start=1):
        weight = "normal"
        if planet.planet == mark:
            weight = "bold"
        ax.annotate(
            str(number),
            (planet.P_days, planet.sy_tmag),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8.5,
            fontweight=weight,
        )
    chosen = table.planet == mark
    if chosen.any():
        planet = table[chosen].iloc[0]
        ring_size = size[chosen].iloc[0] * 2.2
        ax.scatter([planet.P_days], [planet.sy_tmag], s=ring_size, facecolors="none", edgecolors=DARK, linewidths=1.8)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Orbital period (days)")
    ax.set_ylabel("Star brightness to TESS (magnitude; up is brighter)")
    _plain_ticks(ax)
    ax.set_title("The planets you can measure. Bigger points have more velocities.", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower left")
    _credit(fig, CREDIT_SURVEY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def joint(curves: dict, title: str):
    """The joint fit: the folded transit with its model (left), the folded velocities with theirs (right).

    `curves` holds the model curves and the data, as tess_02's fitted_curves() returns them.
    """
    fig, (transit_ax, velocity_ax) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=110)
    transit_ax.plot(curves["data_hours"], curves["data_flux"], ".", ms=1.5, color=LIGHT_GREY)
    transit_ax.plot(curves["hours"], curves["flux"], color=RED, lw=1.6)
    transit_ax.set_xlabel("Hours from mid-transit")
    transit_ax.set_ylabel("Brightness minus 1 (parts per thousand)")
    transit_ax.set_title("Transit", fontsize=10)
    colors = {"HIRES": BLUE, "APF": GREEN}
    telescope = np.asarray(curves["telescope"])
    for name in sorted(set(telescope)):
        mine = telescope == name
        velocity_ax.errorbar(
            curves["data_phase"][mine],
            curves["data_rv"][mine],
            curves["data_rv_err"][mine],
            fmt=".",
            color=colors.get(name, DARK),
            label=name,
            alpha=0.8,
        )
    velocity_ax.plot(curves["phase"], curves["rv"], color=RED, lw=1.6)
    velocity_ax.set_xlabel("Orbital phase (0 = transit)")
    velocity_ax.set_ylabel("Velocity minus offset (m/s)")
    velocity_ax.set_title("Velocities", fontsize=10)
    velocity_ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    _credit(fig, CREDIT_JOINT)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig


def mass_radius(mass, radius, *, label: str, mass_err=None):
    """Your planet's mass and radius against what rock could make.

    The biggest a rocky planet can be (no iron core) and an Earth-like mix, from Zeng et al. 2016,
    with Earth and Neptune for reference.
    """
    top_mass = max(30, 1.5 * float(mass))  # widen the plot for giant planets
    top_radius = max(5, 1.3 * float(radius))
    masses = np.logspace(np.log10(0.5), np.log10(top_mass), 200)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=110)
    ax.plot(masses, rocky_radius(masses, 0.0), color=BROWN, lw=1.6, label="biggest a rocky planet can be (no iron)")
    earth_like = rocky_radius(masses, EARTH_CORE_FRACTION)
    ax.plot(masses, earth_like, color=BROWN, lw=1.2, ls="--", label="rock and iron like Earth")
    ax.axvspan(ZENG_MAX_MASS_EARTH, top_mass, color="#eeeeee", zorder=0)
    ax.text(8.3, 0.62, "beyond the fitted range", fontsize=8, color="#777")
    ax.scatter([1, NEPTUNE[0]], [1, NEPTUNE[1]], color="#555", s=25, zorder=3)
    ax.annotate("Earth", (1, 1), xytext=(6, -12), textcoords="offset points", fontsize=8.5)
    ax.annotate("Neptune", NEPTUNE, xytext=(6, 4), textcoords="offset points", fontsize=8.5)
    mass_bar = None
    if mass_err is not None:
        mass_bar = [[mass_err]]
    ax.errorbar(
        [mass],
        [radius],
        xerr=mass_bar,
        yerr=None,
        fmt="o",
        color=RED,
        ms=8,
        zorder=4,
        label=label,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, top_mass)
    ax.set_ylim(0.6, top_radius)
    ax.set_xticks([tick for tick in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500) if tick <= top_mass])
    ax.set_yticks([tick for tick in (0.6, 1, 2, 3, 4, 5, 8, 12, 16, 20) if tick <= top_radius])
    _plain_ticks(ax)
    ax.set_xlabel("Mass (Earth masses)")
    ax.set_ylabel("Radius (Earth radii)")
    ax.set_title("Above the solid line, rock alone cannot make a planet this big", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    _credit(fig, CREDIT_ROCK)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig
