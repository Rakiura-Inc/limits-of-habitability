# Limits of Habitability notebooks

Notebooks for the Limits of Habitability lab on [Rakiura](https://labs.rakiura.co). Each runs in Google Colab
with no setup: open it, run the first cell, and it installs what it needs.

## The bootcamp

| Notebook | What you do | Open |
|---|---|---|
| [01_find_your_test_planet](notebooks/01_find_your_test_planet.ipynb) | Find a real planet that could test the lab's size limit, then work out whether it can be weighed | [Colab](https://colab.research.google.com/github/Rakiura-Inc/limits-of-habitability/blob/main/notebooks/01_find_your_test_planet.ipynb) |
| [02_planet_or_star](notebooks/02_planet_or_star.ipynb) | Find a star's magnetic activity cycle in twenty years of data, and decide what it means for a planet search | [Colab](https://colab.research.google.com/github/Rakiura-Inc/limits-of-habitability/blob/main/notebooks/02_planet_or_star.ipynb) |
| [03_find_the_dip](notebooks/03_find_the_dip.ipynb) | Measure a planet's size from the dip in its star's TESS light curve | [Colab](https://colab.research.google.com/github/Rakiura-Inc/limits-of-habitability/blob/main/notebooks/03_find_the_dip.ipynb) |
| [04_weigh_a_planet](notebooks/04_weigh_a_planet.ipynb) | Fit an orbit to radial velocities, weigh a real planet, and decide whether it is rock or gas | [Colab](https://colab.research.google.com/github/Rakiura-Inc/limits-of-habitability/blob/main/notebooks/04_weigh_a_planet.ipynb) |

## TESS-Keck Survey Data Validation (research project)

| Notebook | What you do | Open |
|---|---|---|
| [tess_01_vet_a_candidate](notebooks/tess_01_vet_a_candidate.ipynb) | Read NASA's validation tests for a TESS-Keck Survey planet, check for nearby stars, and fill in a vetting sheet | [Colab](https://colab.research.google.com/github/Rakiura-Inc/limits-of-habitability/blob/main/notebooks/tess_01_vet_a_candidate.ipynb) |
| [tess_02_radius_and_mass](notebooks/tess_02_radius_and_mass.ipynb) | Fit a planet's TESS transits and Keck velocities together for its radius and mass, then test what the answer depends on | [Colab](https://colab.research.google.com/github/Rakiura-Inc/limits-of-habitability/blob/main/notebooks/tess_02_radius_and_mass.ipynb) |

## Joint Fitting of Gaia and Radial Velocity Data (research project)

| Notebook | What you do | Open |
|---|---|---|
| [gaia_01_first_orbitize_fit](notebooks/gaia_01_first_orbitize_fit.ipynb) | Install orbitize!, run its Hipparcos and Gaia example, run it longer, and decide whether it has settled | [Colab](https://colab.research.google.com/github/Rakiura-Inc/limits-of-habitability/blob/main/notebooks/gaia_01_first_orbitize_fit.ipynb) |
| [gaia_02_who_feels_a_pull](notebooks/gaia_02_who_feels_a_pull.ipynb) | Read the Hipparcos-Gaia Catalog of Accelerations for stars from the lab's papers, and decide which feel a pull | [Colab](https://colab.research.google.com/github/Rakiura-Inc/limits-of-habitability/blob/main/notebooks/gaia_02_who_feels_a_pull.ipynb) |
| [gaia_03_inner_planet_from_velocities](notebooks/gaia_03_inner_planet_from_velocities.ipynb) | Recover iota Draconis b's eccentric orbit from public velocities, and check it against the lab's paper | [Colab](https://colab.research.google.com/github/Rakiura-Inc/limits-of-habitability/blob/main/notebooks/gaia_03_inner_planet_from_velocities.ipynb) |

## Data

Everything is public and downloaded by the notebooks when first needed:

- A Catalog of Habitable Zone Exoplanets, Hill et al. 2023, AJ 165, 34: VizieR J/AJ/165/34
- The California Legacy Survey V, Isaacson et al. 2024, ApJS 274, 35: VizieR J/ApJS/274/35
- The TESS-Keck Survey XX, Polanski et al. 2024, ApJS 272, 32: VizieR J/ApJS/272/32
- TESS 2-minute light curves and data validation reports (SPOC) from the Mikulski Archive for Space Telescopes (MAST)
- Star properties from the NASA Exoplanet Archive. `habtrain/planets.csv` holds a small copy, read on 2026-10-25
- Gaia DR3 positions and RUWE, from the ESA Gaia archive; follow-up listings on ExoFOP
- The Hipparcos-Gaia Catalog of Accelerations, EDR3 edition, Brandt 2021, ApJS 254, 42 (ships inside orbitize!)
- iota Draconis velocities: Butler et al. 2006, ApJ 646, 505 (VizieR J/ApJ/646/505) and Hill et al. 2021, AJ 162, 211 (VizieR J/AJ/162/211)

### Acknowledgements

This work has made use of the NASA Exoplanet Archive, which is operated by the California Institute of
Technology, under contract with the National Aeronautics and Space Administration under the Exoplanet
Exploration Program. It includes data collected by the TESS mission, obtained from the MAST data archive at
the Space Telescope Science Institute (STScI); funding for the TESS mission is provided by the NASA Explorer
Program. It has made use of the VizieR catalogue access tool, CDS, Strasbourg, France. It has made use of
data from the European Space Agency (ESA) mission Gaia, processed by the Gaia Data Processing and Analysis
Consortium (DPAC). It has made use of the Exoplanet Follow-up Observation Program (ExoFOP) website, which is
operated by the California Institute of Technology, under contract with NASA. Cite the papers above if you
use the data.

## Running it yourself

```
make local   # a virtual environment with the package, then Jupyter Lab on the notebooks
make test    # the tests
make run     # run every notebook with blank answers into notebooks/executed/
```

`habtrain/` holds the plumbing: downloads, fits, plots, recording answers. The notebooks hold the thinking.
The TESS project notebooks also need `pip install "habtrain[tess]"`, which adds batman, astroquery and
setuptools; the Gaia project notebooks need `pip install "habtrain[orbits]"`, which adds orbitize!. Their first cells do this. Answers to the exercises are not in this repository: they open on
each step in Rakiura after you hand it in.

Code: MIT licence (see LICENSE). batman-package, used only by the TESS project notebooks, is GPL-3.0 and is
installed separately, not included here.
