"""Every plot draws what it says: the right panels, labels, counts and credit line."""

import sys

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import habtrain as ht


def texts(fig):
    return [t.get_text() for t in fig.texts]


@pytest.fixture
def catalog():
    return pd.DataFrame({
        "Planet": ["a b", "c d", "e f"],
        "S": [1361.0, 2722.0, 680.5],
        "Rad": [1.0, 0.7, 3.0],
        "f_Rad": [0, 0, 1],
        "f_Mass": [0, 1, 0],
        "l_Mass": ["", "", "<="],
        "K": [0.1, 2.0, 0.5],
        "Teff": [5772, 3500, 4500],
    })


def test_catalog_counts_each_kind_of_planet_in_the_legend(catalog):
    fig = ht.plot.catalog(catalog, highlight=["c d"], numbers=True)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == [
        "radius estimated from mass (1)",
        "radius measured (2)",
        "mass and radius both measured (1)",
    ]
    assert fig.axes[0].get_title().endswith("(3)")
    assert ht.plot.CREDIT_CATALOG in texts(fig)
    annotations = [t.get_text() for t in fig.axes[0].texts]
    assert {"1", "2", "3", "c d"} <= set(annotations)


def test_catalog_coloured_by_temperature_has_a_colour_bar(catalog):
    fig = ht.plot.catalog(catalog, color_by="Teff", title="Mine")
    assert fig.axes[0].get_title() == "Mine"
    assert fig.axes[1].get_ylabel() == "Star temperature (K)"


def test_wobbles_colours_your_planet_and_draws_the_floor(catalog):
    fig = ht.plot.wobbles(catalog, floor=0.3, mine="c d")
    bars = fig.axes[0].patches
    assert [b.get_height() for b in bars] == [0.1, 0.5, 2.0]  # sorted
    assert len({b.get_facecolor() for b in bars}) == 2
    assert "30 cm/s" in [t.get_text() for t in fig.axes[0].texts]


def test_activity_and_periodogram_are_one_panel_each():
    night = pd.DataFrame({"year": [2005.0, 2006.0], "S": [0.2, 0.3]})
    assert len(ht.plot.activity(night, "HD 1").axes) == 1
    fig = ht.plot.periodogram([1, 2, 3], [0.1, 0.9, 0.2], "years", "Peaks", mark=2, credit="Data: me")
    assert fig.axes[0].get_xlabel() == "Period (years)"
    assert texts(fig) == ["Data: me"]
    assert texts(ht.plot.periodogram([1, 2], [0.1, 0.2], "days", "No credit")) == []


def test_velocities_adds_a_folded_panel_when_given_a_period():
    t, v, err = ht.fit.fake_velocities(5.0, 3.0, n=20)
    assert len(ht.plot.velocities(t, v, err).axes) == 1
    fig = ht.plot.velocities(t, v, err, ["A"] * 10 + ["B"] * 10, P=3.0, T0=0.0, K=5.0, offsets={"A": 1.0})
    assert len(fig.axes) == 2
    assert fig.axes[1].get_title() == "Folded on P = 3.0000 d"
    assert len(fig.axes[1].lines) >= 1  # the fitted orbit


def test_corner_draws_every_pair_of_parameters():
    rng = np.random.default_rng(0)
    fit = ht.fit.Fit(["K", "jitter"], rng.normal(size=(500, 2)), ["A"], True, 1.0, 1)
    fig = ht.plot.corner(fit, truths={"K": 0.0}, credit="Data: me")
    assert len(fig.axes) == 4
    assert "Data: me" in texts(fig)


def test_star_map_circles_and_labels_the_marked_stars():
    summary = pd.DataFrame({
        "Name": ["HD 1", "HD 2"],
        "BV": [0.6, 0.9],
        "logRhk": [-4.9, -4.5],
        "variation": [2.0, 30.0],
    })
    fig = ht.plot.star_map(summary, mark={"HD 2": "yours"})
    assert [t.get_text() for t in fig.axes[0].texts] == ["yours"]
    assert fig.axes[0].get_title() == "The 2 stars you can pick from"


def test_star_map_interactive_says_so_without_plotly(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.express", None)
    assert ht.plot.star_map_interactive(pd.DataFrame()) is None
    assert "Plotly is not installed" in capsys.readouterr().out


def test_transit_shades_the_transit_and_zooms_on_the_averages():
    hours = np.linspace(-10, 10, 400)
    flux = 1 - 0.01 * (np.abs(hours) < 1)
    bins = ht.lightcurve.binned(hours, flux, 30)
    fig = ht.plot.transit(hours, flux, bins=bins, duration_hours=2.0, title="Dip")
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["every point", "15-minute averages", "the transit"]
    low, high = ax.get_ylim()
    assert low < 0.99 < 1.0 < high


def test_planet_map_rings_your_planet():
    planets = pd.DataFrame({
        "planet": ["TOI-1 b", "TOI-2 b"],
        "P_days": [3.0, 10.0],
        "sy_tmag": [9.0, 10.0],
        "velocities": [30, 60],
        "telescopes": ["HIRES 30", "HIRES 40 + APF 20"],
    })
    fig = ht.plot.planet_map(planets, mark="TOI-2 b")
    ax = fig.axes[0]
    assert len(ax.collections) == 3  # HIRES only, both telescopes, and the ring
    bold = [t.get_text() for t in ax.texts if t.get_fontweight() == "bold"]
    assert bold == ["2"]


def test_joint_plot_shows_transit_and_velocities():
    c = {
        "data_hours": np.array([-1.0, 0.0, 1.0]),
        "data_flux": np.array([0.0, -1.0, 0.0]),
        "hours": np.array([-1.0, 1.0]),
        "flux": np.array([0.0, 0.0]),
        "telescope": np.array(["APF", "HIRES", "HIRES"]),
        "data_phase": np.array([0.1, 0.2, 0.3]),
        "data_rv": np.array([1.0, 2.0, 3.0]),
        "data_rv_err": np.array([0.5, 0.5, 0.5]),
        "phase": np.array([-0.5, 0.5]),
        "rv": np.array([0.0, 0.0]),
    }
    fig = ht.plot.joint(c, "TOI-1 b")
    assert [ax.get_title() for ax in fig.axes] == ["Transit", "Velocities"]
    assert ht.plot.CREDIT_JOINT in texts(fig)


def test_mass_radius_widens_for_a_giant_planet():
    small = ht.plot.mass_radius(5.0, 1.5, "mine", mass_err=1.0, radius_err=0.1)
    assert small.axes[0].get_xlim() == pytest.approx((0.5, 30))
    giant = ht.plot.mass_radius(300.0, 11.0, "giant")
    assert giant.axes[0].get_xlim() == pytest.approx((0.5, 450))
    assert giant.axes[0].get_ylim() == pytest.approx((0.6, 14.3))


def test_light_curve_plots_time_against_brightness():
    class Column:
        def __init__(self, value):
            self.value = value

    class LightCurve:
        time = Column(np.arange(5.0))
        flux = Column(np.ones(5))

    fig = ht.plot.light_curve(LightCurve(), "Star")
    assert isinstance(fig, Figure)
    assert fig.axes[0].lines[0].get_xdata().tolist() == [0, 1, 2, 3, 4]
