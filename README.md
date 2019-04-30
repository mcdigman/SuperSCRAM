# README #
Version 1.0 of SuperSCRAM: Super-Sample Covariance Reduction and Mitigation
Described in arXiv:1904.12071

Author contact: Matthew C. Digman digman.12@osu.edu 

Dependencies are:
numpy
scipy
camb
spherical\_geometry
matplotlib
astropy
dill (optional, makes viewing saved states from runs possible)
pytest (for testing)
basemap (optional)

the following creates a conda environment with the required dependencies:

`conda create -n ssc_build_test -c conda-forge python=2.7.15 basemap scipy matplotlib=2.1.2 pytest numpy astropy cython mpmath future healpy spherical-geometry future
pip install --upgrade numpy
pip install spherical_geometry camb future
pip install dill`


To run a version of the demonstration case in the paper (arXiv:1904.12071),
`python wfirst_embed_demo.py`

The results of the highly converged run in the paper can be accessed using 
`python wfirst_demo_reconstitute`

To run the unit tests,
`python bundled_test.py`


Develop repository for the super sample covariance project. 
