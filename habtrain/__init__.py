"""habtrain: the plumbing behind the Limits of Habitability bootcamp notebooks.

Library for plumbing, notebook for judgment: this package fetches the public catalogs, parses
them, fits orbits and draws the plots. Choosing what to filter, reading the plots and deciding
what they mean stay in the notebooks.
"""

from . import choose, constants, data, fit, joint, lightcurve, orbits, plot, portal, record, vet

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "choose",
    "constants",
    "data",
    "fit",
    "joint",
    "lightcurve",
    "orbits",
    "plot",
    "portal",
    "record",
    "vet",
]
