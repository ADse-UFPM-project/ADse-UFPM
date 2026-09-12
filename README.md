# ADse-UFPM

This repository contains the Python implementation and case-study inputs for the ADse-UFPM model described in the manuscript *Integrating park typology, service targets, climate conditions, and social-ecological coupling for urban forest park management*.

ADse-UFPM is a park-scale, annual-step social-ecological simulation model. It represents ecological condition, recreation, conservation targets, ecosystem-service provision, visitor satisfaction, management conflict, extreme disturbance, management allocation, and early-warning signals.

## Repository contents

- `ADse-UFPM.py`: core model, sensitivity analysis, Sobol analysis, early-warning calculations, and optional landscape preprocessing.
- `ADse-UFPM_run.py`: entry script for selecting and running one site and climate scenario.
- `UFPM_SMF_paras.csv`: shared model parameters.
- `Sobol_indices.csv`: reported Sobol sensitivity indices.
- `Beijing/`, `Chengdu/`, `Guangzhou/`, and `Harbin/`: site configurations, climate scenarios, landscape descriptors, and model results.

## Requirements

Python 3.10 or later is recommended. Install the required packages with:

```bash
python -m pip install -r requirements.txt
```

The geospatial packages listed in `requirements.txt` are needed only when new landscape descriptors are generated from park boundaries. They are not required when the supplied `Landscape_descriptors.csv` files are used.

## Running a simulation

Open `ADse-UFPM_run.py` and set the following values near the top of the file:

- `SITE`: site folder, such as `Beijing`.
- `SITE_CONFIG`: site configuration file within that folder.
- `SCENARIO`: climate scenario file, such as `Baseline.csv`, `RCP4.5.csv`, or `RCP8.5.csv`.
- `USE_LANDSCAPE`: use the supplied landscape descriptors when set to `True`.
- `RUN_SENSITIVITY` and `RUN_SOBOL`: enable the optional analyses when required.

Then run:

```bash
python ADse-UFPM_run.py
```

Results are written to:

```text
<site>/Results/<scenario>/
```

The standard outputs are `Simulation_results.csv`, `Annual_summary.csv`, and `Early_warning.csv`. Enabling the optional analyses also produces `Sensitivity_analysis.csv` and a site-level Sobol indices file.

## Landscape preprocessing

The supplied landscape descriptor files can be used directly. To derive descriptors for a new boundary, set `BUILD_LANDSCAPE = True` in `ADse-UFPM_run.py` and provide the boundary path, projected EPSG code, staff number, and image date window. This option retrieves Sentinel-2 and OpenStreetMap data and therefore requires internet access.

## Reproducibility

Simulation length, replicate count, random seed, management setting, and site characteristics are read from the selected site configuration. Keep the directory structure unchanged so that the runner can locate the parameter, scenario, and landscape files.
