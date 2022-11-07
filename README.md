# aia-flare-catalog

Code base for generating an AIA-based flare catalog using a peak detection algorithm on AIA timeseries data.

A conda environment can be initialized from the solar.yml file.

The Jupyter notebooks provide a starting point for manual verification of the AIA catalog.

The final AIA-based flare catalog is located in 'aia_flares_catalog_verified.csv'. This contains C, M and X flares from 2010-2017 as automatically detected by the peak finding routine in 'aia_sharps_find_peaks.py'. The M and X flares were manually verified and corrected. 
