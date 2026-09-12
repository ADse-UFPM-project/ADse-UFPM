# -*- coding: utf-8 -*-
"""Run one ADse-UFPM site and climate scenario.

Edit the settings below, then run this file. Each site folder contains its
configuration, scenario files, landscape descriptors, and generated results.
"""

import importlib.util
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# Site and scenario
SITE = "Beijing"
SITE_CONFIG = "Beijing_NSF.csv"
SCENARIO = "Baseline.csv"

# Landscape inputs
USE_LANDSCAPE = True
BUILD_LANDSCAPE = False
BOUNDARY_FILE = None  # Example: Path("Beijing") / "park_boundary.shp"
UTM_EPSG = 32650
STAFF = 320
DATE_WINDOW = "2025-07-01/2025-09-30"

# Optional analyses
RUN_SENSITIVITY = False
RUN_SOBOL = False

SENS_BUDGET_RANGE = [50, 75, 100, 125, 150]
SENS_NMANAGERS_RANGE = [1, 2, 3, 4, 5]
SENS_N_REPS = 30
SENS_N_YEARS = 100

SOBOL_N_BASE = 64
SOBOL_N_YEARS = 50
SOBOL_N_REPS = 3


def load_model():
    """Load the core module, whose filename contains a hyphen."""
    model_path = BASE_DIR / "ADse-UFPM.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"Core model not found: {model_path}")

    spec = importlib.util.spec_from_file_location("adse_ufpm", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the core model from {model_path}")

    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model


def require_file(path, description):
    """Raise a clear error before a long run starts with a missing input."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def main():
    model = load_model()

    parameter_file = BASE_DIR / "UFPM_SMF_paras.csv"
    site_dir = BASE_DIR / SITE
    config_file = site_dir / SITE_CONFIG
    scenario_file = site_dir / SCENARIO

    require_file(parameter_file, "Parameter file")
    require_file(config_file, "Site configuration")
    require_file(scenario_file, "Scenario file")

    params = model.load_params(parameter_file)
    model.apply_scenario(params, scenario_file)
    config = model.load_site_config(config_file)

    if BUILD_LANDSCAPE:
        if BOUNDARY_FILE is None:
            raise ValueError(
                "Set BOUNDARY_FILE before enabling BUILD_LANDSCAPE"
            )
        boundary_path = Path(BOUNDARY_FILE)
        if not boundary_path.is_absolute():
            boundary_path = BASE_DIR / boundary_path
        require_file(boundary_path, "Park boundary")
        model.build_landscape(
            boundary_path,
            UTM_EPSG,
            STAFF,
            site_dir,
            date_window=DATE_WINDOW,
        )

    landscape = (
        model.load_landscape(site_dir, config) if USE_LANDSCAPE else None
    )
    scenario_name = Path(SCENARIO).stem
    output_dir = site_dir / "Results" / scenario_name

    print("-" * 60)
    print(f"Site folder    : {site_dir.name}  (config: {config_file.name})")
    print(f"  Site name      : {config.get('site_name', config_file.stem)}")
    print(f"  Park type      : {config['park_type']}")
    print(f"  Service target : {config['service_target']}")
    print(f"  Climate zone   : {config['climate_zone']}")
    print(f"  Scenario       : {SCENARIO}")
    print(f"  Results folder : {output_dir}")

    model.run_site(
        site_cfg=config,
        params=params,
        output_dir=str(output_dir),
        landscape=landscape,
    )

    social_params = {
        key: config[key]
        for key in (
            'gdp_per_capita',
            'civic_participation',
            'park_visitor_density',
            'park_area',
        )
        if key in config
    } or None

    if RUN_SENSITIVITY:
        print("\n[Sensitivity] grid sweep ...")
        model.sensitivity_analysis(
            park_type=config['park_type'],
            service_target=config['service_target'],
            climate_zone=config['climate_zone'],
            params=params,
            budget_range=SENS_BUDGET_RANGE,
            n_managers_range=SENS_NMANAGERS_RANGE,
            n_reps=SENS_N_REPS,
            n_years=SENS_N_YEARS,
            seed=config.get('seed', 2024),
            output_dir=str(output_dir),
            social_params=social_params,
            management_mode=config.get('management_mode', 'single'),
            portfolio_method=config.get('portfolio_method', 'random'),
            landscape=landscape,
        )

    if RUN_SOBOL:
        print("\n[Sobol] global sensitivity ...")
        model.sobol_sensitivity(
            site_cfg=config,
            params=params,
            output_dir=str(output_dir),
            n_base=SOBOL_N_BASE,
            n_years=SOBOL_N_YEARS,
            n_reps=SOBOL_N_REPS,
            seed=config.get('seed', 2024),
            landscape=landscape,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
