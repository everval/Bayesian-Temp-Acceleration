# Bayesian Temperature Acceleration

This repository accompanies the paper **“Spatial emergence of acceleration in global warming.”** It contains the analysis code and, following publication, the paper and supplementary material.

## Repository contents

```text
Bayesian-Temp-Acceleration/
├── README.md
├── data_grid.py
├── grf.py
├── paper.pdf
├── supplement.pdf
├── figures/
├── data/
└── results/
```

The main files and directories are:

* `data_grid.py`: preprocessing of the gridded climate datasets, estimation of the Bayesian quadratic temperature models, posterior sampling, and diagnostic plots.
* `grf.py`: spatial interpolation of missing posterior estimates using a Gaussian random field, generation of spatial figures, and calculation of threshold-exceedance summaries.
* `paper.pdf`: the published paper, added when available.
* `supplement.pdf`: the supplementary material, added when available.
* `figures/`: selected final figures from the paper.
* `data/`: local input data. This directory is excluded from version control.
* `results/`: generated intermediate results and figures. This directory is excluded from version control.

## Selected results

Selected figures from the study are included in the `figures/` directory.

For example:

```markdown
![Timing of detectable temperature acceleration in HadCRUT5 at the 50% and 90% posterior probability thresholds.](figures/exceeded_combined_HC.png)
```

The full set of results and their interpretation are provided in the paper and supplementary material.

## Data availability

The climate datasets used in the analysis are not included in this repository. They must be obtained directly from their original providers and may be subject to separate terms of use.

The datasets can be accessed from the following official sources:

* [HadCRUT5 — Met Office Hadley Centre](https://www.metoffice.gov.uk/hadobs/hadcrut5/)
* [ERA5 monthly averaged data on single levels — Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means)
* [NOAAGlobalTemp — NOAA National Centers for Environmental Information](https://www.ncei.noaa.gov/products/land-based-station/noaa-global-temp)

The ERA5 dataset is also identified by DOI:

```text
10.24381/cds.f17050d7
```

Use the exact dataset versions specified in the paper and supplementary material. Dataset updates released after the analysis may produce results that differ from those reported in the paper.

After downloading and preparing the datasets, place the required NetCDF files in the local `data/` directory using the following filenames:

```text
data/
├── data_grid_hadcrut.nc
├── data_grid_noaa.nc
└── data_grid_era5.nc
```

The `data/` directory is excluded from version control. Dataset versions, access dates, citations, and preprocessing procedures are documented in the paper and supplementary material.

## Reproducibility

The repository contains the scripts used to perform the analysis and generate the results reported in the paper.

Exact reproduction requires:

* the dataset versions specified in the paper and supplementary material;
* the required Python dependencies;
* the analysis settings described in the methods;
* the same preprocessing procedures and truncation periods.

Generated intermediate files are not stored in the repository but can be recreated by running the supplied scripts in the following order:

1. Run `data_grid.py` to preprocess the data and generate posterior summaries.
2. Run `grf.py` to interpolate missing spatial estimates and generate the final spatial outputs.
