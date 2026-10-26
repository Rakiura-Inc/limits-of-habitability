"""habtrain: the plumbing behind the Limits of Habitability bootcamp notebooks.

Library for plumbing, notebook for judgment: this package fetches the public catalogs, parses
them, fits orbits and draws the plots. Choosing what to filter, reading the plots and deciding
what they mean stay in the notebooks.
"""

__version__ = "0.1.0"

from . import choose, data, fit, joint, lightcurve, orbits, plot, portal, record, vet  # noqa: E402,F401
