"""habtrain: the plumbing behind the Limits of Habitability bootcamp notebooks.

Library for plumbing, notebook for method and judgment. This package downloads and parses the
public catalogs, runs emcee once a notebook has shown how, draws the fiddly plots, records answers
and connects to Rakiura. The models, likelihoods and the library calls they teach (lightkurve,
astropy, emcee, batman, orbitize!) are written out in the notebooks, where members read them.
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
