# -*- coding: utf-8 -*-
"""
ADse-UFPM.py
Adaptive Socio-Ecological Urban Forest Management (ADse-UFPM) Framework

Core simulation routines for a park-scale, annual time-step coupled
social-ecological model. Provides the four functional process modules
(management, visitor, community, regulatory), the annual simulation
loop, the replicate runner, sensitivity-analysis drivers, and the
site-level orchestration used by the run script.

Climate forcing (stochastic variability, warming, and the climate-extreme
disturbance regime) is not hardcoded; it is supplied by a scenario file in the
Scenarios/ folder via load_scenario()/apply_scenario().

Created on 2026-03-19
Updated 2026-06-06
@author Xinyuan Wei
"""

import json
import os
import time
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
import geopandas as gpd
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from scipy import ndimage
from shapely.geometry import LineString
from shapely.ops import unary_union
landscape_tools = True #False
#----------------------------------------------------------------------------
# Action vocabulary
#----------------------------------------------------------------------------
action_order = ['monitoring', 'invasive_removal', 'ecological_restoration',
                'infrastructure', 'visitor_experience', 'ecosystem_service']

action_labels = {
    'monitoring':             'Monitoring',
    'invasive_removal':       'Invasive removal',
    'ecological_restoration': 'Ecological restoration',
    'infrastructure':         'Infrastructure',
    'visitor_experience':     'Visitor experience',
    'ecosystem_service':      'Ecosystem service',
}

#----------------------------------------------------------------------------
# Section 1: Input loaders
#----------------------------------------------------------------------------
def load_params(paras_path):
    """
    Load parameters from CSV with prefix-based naming scheme.

    Returns dict with keys:
        park_type_params, service_target_weights, climate_params,
        ci_weights, intervention, ecology, conflict,
        social_coupling, climate_variability,
        disturbance, optimization, early_warning.
    """
    df = pd.read_csv(paras_path, comment='#')

    result = {
        'park_type_params': {},
        'service_target_weights': {},
        'climate_params': {},
        'ci_weights': {},
        'intervention': {},
        'ecology': {},
        'conflict': {},
        'social_coupling': {},
        'climate_variability': {},
        'disturbance': {},
        'optimization': {},
        'early_warning': {},
        'landscape': {},
    }

    park_type_prefixes = ['NSF_', 'NISF_', 'NRF_', 'PF_']
    park_type_map = {'NSF_': 'NSF', 'NISF_': 'NISF', 'NRF_': 'NRF', 'PF_': 'PF'}
    service_prefixes = ['conservation_', 'tourism_', 'recreation_', 'environmental_']
    climate_prefixes = ['Tropical_', 'Subtropical_', 'Temperate_', 'Boreal_', 'Semiarid_']
    climate_zone_map = {
        'Tropical_':    'Tropical',
        'Subtropical_': 'Subtropical',
        'Temperate_':   'Temperate',
        'Boreal_':      'Boreal',
        'Semiarid_':    'Semi-arid',
    }
    other_prefix_to_group = [
        (('Ce_', 'Cr_', 'Ct_', 'Cep_'),                              'ci_weights'),
        (('invrem_', 'ecorest_', 'infra_', 'visitor_', 'ecosvc_'),   'intervention'),
        (('eco_',), 'ecology'),
        (('cfl_',), 'conflict'),
        (('soc_',), 'social_coupling'),
        (('var_',), 'climate_variability'),
        (('ext_',), 'disturbance'),
        (('opt_',), 'optimization'),
        (('ew_',),  'early_warning'),
        (('lsc_',), 'landscape'),
    ]

    for _, row in df.iterrows():
        para_name = row['para_name'].strip()
        try:
            value = float(row['value'])
        except (ValueError, TypeError):
            continue

        matched = False

        for prefix in park_type_prefixes:
            if para_name.startswith(prefix):
                park_type = park_type_map[prefix]
                key = para_name[len(prefix):]
                result['park_type_params'].setdefault(park_type, {})[key] = value
                matched = True
                break
        if matched:
            continue

        for prefix in service_prefixes:
            if para_name.startswith(prefix):
                service_type = prefix.rstrip('_')
                key = para_name[len(prefix):]
                result['service_target_weights'].setdefault(service_type, {})[key] = value
                matched = True
                break
        if matched:
            continue

        for prefix in climate_prefixes:
            if para_name.startswith(prefix):
                zone = climate_zone_map[prefix]
                key = para_name[len(prefix):]
                result['climate_params'].setdefault(zone, {})[key] = value
                matched = True
                break
        if matched:
            continue

        for prefixes, group in other_prefix_to_group:
            if any(para_name.startswith(p) for p in prefixes):
                result[group][para_name] = value
                break

    return result


def load_site_config(site_path):
    """
    Load a site configuration file (csv, columns: var, value, units, detail).

    Splits each line on the first comma only so that detail fields that
    themselves contain commas do not break the reader. Returns a dict
    with required keys park_type, service_target, climate_zone, plus the
    optional numeric and string keys present in the file. String keys such
    as management_mode and portfolio_method are retained as-is.
    """
    numeric_keys = {
        'n_years', 'n_reps', 'n_managers', 'seed', 'budget',
        'park_area', 'gdp_per_capita', 'civic_participation',
        'park_visitor_density',
        'edge_density', 'shape_index', 'edge_fraction', 'canopy_observed',
        'canopy_edge_density', 'division_index', 'largest_patch',
        'core_canopy_30', 'matrix_built', 'access_density',
        'vehicle_density', 'edge_per_staff',
    }
    int_keys = {'n_years', 'n_reps', 'n_managers', 'seed'}

    config = {}
    with open(site_path, encoding='utf-8-sig') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',', 1)
            if len(parts) < 2:
                continue
            key = parts[0].strip()
            if key == 'var':
                continue
            value_str = parts[1].split(',')[0].strip()
            if not value_str or value_str.startswith('#'):
                continue
            if key in numeric_keys:
                try:
                    config[key] = (int(float(value_str))
                                   if key in int_keys else float(value_str))
                except (ValueError, TypeError):
                    pass
            else:
                config[key] = value_str

    for req in ('park_type', 'service_target', 'climate_zone'):
        if req not in config:
            raise ValueError(f"Site config missing required key: {req}")
    return config

def load_landscape(site_dir, site_cfg=None):
    # Read Landscape_descriptors.csv from a site folder, written by
    # ADse-UFPM_Landscape.py. Values already present in the site config win,
    # so a park can be tuned by hand. Returns None when nothing is available,
    # which makes the model fall back to the original non-spatial behaviour.
    keys = ("park_area", "edge_density", "shape_index", "edge_fraction",
            "canopy_observed", "canopy_edge_density", "division_index",
            "largest_patch", "core_canopy_30", "matrix_built",
            "access_density", "vehicle_density", "edge_per_staff")
    found = {}
    path = os.path.join(str(site_dir), "Landscape_descriptors.csv")
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) < 2 or parts[0].strip() not in keys:
                    continue
                try:
                    found[parts[0].strip()] = float(parts[1])
                except (ValueError, TypeError):
                    pass
    if site_cfg:
        for key in keys:
            if key in site_cfg:
                found[key] = float(site_cfg[key])
    return found or None


def landscape_terms(lsc, params):
    # Turn the measured descriptors into the dimensionless terms the annual
    # loop uses. Every term is 1.0, or 0.0 where additive, when the landscape
    # parameters are left at their neutral defaults, so version 1 is recovered.
    ls = params["landscape"]
    if not lsc:
        return {"edge_fraction": 0.0, "kappa": 0.0, "propagule": 0.0,
                "workload": 1.0, "removal_scale": 1.0, "disturb_edge": 1.0}

    edge_ref = ls.get("lsc_canopy_edge_reference", 230.0)
    staff_ref = ls.get("lsc_edge_per_staff_reference", 440.0)
    vehicle_ref = ls.get("lsc_vehicle_reference", 65.0)

    # Edge exposure: how much canopy edge there is, and how hostile the ground
    # beyond it is. Canopy edge is used, not the park outline, because internal
    # gaps carry most of the edge in every park measured so far.
    kappa = (lsc.get("canopy_edge_density", edge_ref) / edge_ref
             * lsc.get("matrix_built", 0.3))

    # Propagule pressure rises with a hostile matrix and a broken canopy.
    propagule = (ls.get("lsc_propagule_scale", 0.0)
                 * lsc.get("matrix_built", 0.3)
                 * (1.0 - lsc.get("largest_patch", 1.0)))

    # Workload: canopy edge each staff member covers, slowed where crews
    # cannot drive to the work.
    access = max(lsc.get("vehicle_density", vehicle_ref), 1.0)
    access_penalty = (vehicle_ref / access) ** ls.get("lsc_access_exponent", 0.5)
    load = lsc.get("edge_per_staff", staff_ref) / staff_ref * access_penalty
    # Sub-linear: workload spans two orders of magnitude across these parks, so
    # a linear penalty would wipe out the largest site entirely.
    workload = load ** ls.get("lsc_workload_exponent", 0.0)

    # Removal is less durable on a broken canopy: more re-invasion sources.
    removal_scale = 1.0 - (ls.get("lsc_division_weight", 0.0)
                           * lsc.get("division_index", 0.0))

    return {"edge_fraction": lsc.get("edge_fraction", 0.0),
            "kappa": kappa,
            "propagule": propagule,
            "workload": max(workload, 1.0),
            "removal_scale": float(np.clip(removal_scale, 0.1, 1.0)),
            "disturb_edge": 1.0 + ls.get("lsc_edge_disturbance", 0.0)}


def site_folder_name(site_id):
    """
    Convert a site_id such as 'City_NSF' into the output folder name
    'City_nsf' (city token kept as-is, park-type suffix lower-cased).
    """
    if '_' not in site_id:
        return site_id[:1].upper() + site_id[1:].lower()
    head, _, tail = site_id.partition('_')
    return head + '_' + tail.lower()

def load_scenario(scenario_path):
    """
    Load a climate-scenario file and return its climate-variability (var_) and
    disturbance (ext_) parameter groups.

    A scenario file uses the same prefixed CSV format as the parameter table
    and fully defines the climate forcing of a run (stochastic anomalies,
    warming trend, and the climate-extreme disturbance regime), independently
    of the park. Switch scenarios by pointing the run script at a different
    file in the Scenarios/ folder (e.g. Baseline.csv, RCP45.csv, RCP85.csv).
    """
    sc = load_params(scenario_path)
    return {'climate_variability': sc['climate_variability'],
            'disturbance':         sc['disturbance']}

def apply_scenario(params, scenario_path):
    """
    Merge a climate scenario (see load_scenario) into a loaded parameter set,
    replacing the climate-variability and disturbance groups. Returns params.
    The model itself contains no climate-scenario values; they are supplied
    entirely by the chosen scenario file.
    """
    sc = load_scenario(scenario_path)
    params['climate_variability'].update(sc['climate_variability'])
    params['disturbance'].update(sc['disturbance'])
    return params

#----------------------------------------------------------------------------
# Section 2: Per-step process functions
#----------------------------------------------------------------------------
def compute_condition_indices(native_cov, invasive_cov, canopy, cl, params):
    """
    Compute the four condition indices from landscape state.

    ce  : Ecological condition (native quality, lack of invasion).
    cr  : Recreation condition (canopy, non-invasive space, visitor base).
    ct  : Conservation-target condition (habitat, canopy, visitor fit).
    cep : Ecosystem-service provisioning (carbon, native, structural density).

    Returns (ce, cr, ct, cep), all clipped to [0, 1].
    """
    ci_w = params['ci_weights']

    ce = (ci_w.get('Ce_w_native', 0.40)         * native_cov +
          ci_w.get('Ce_w_native_quality', 0.35) * native_cov * (1 - invasive_cov) +
          ci_w.get('Ce_w_uninvaded', 0.25)      * (1 - invasive_cov))

    visitor_quality = 0.5
    cr = (ci_w.get('Cr_w_canopy', 0.40)      * canopy +
          ci_w.get('Cr_w_noninvasive', 0.35) * (1 - invasive_cov) +
          ci_w.get('Cr_w_visitor', 0.25)     * visitor_quality)

    ct = (ci_w.get('Ct_w_habitat', 0.40) * native_cov +
          ci_w.get('Ct_w_canopy', 0.35)  * canopy +
          ci_w.get('Ct_w_visitor', 0.25) * visitor_quality)

    eco_carbon_native_advantage = params['ecology'].get('eco_carbon_native_advantage', 1.2)
    eco_carbon_reference        = params['ecology'].get('eco_carbon_reference', 100.0)
    carbon_max_tha              = cl.get('carbon_max_tha', 150.0)
    heat_urgency                = cl.get('heat_urgency', 0.1)

    carbon_norm    = np.clip(canopy * (1 + heat_urgency) /
                             (carbon_max_tha / eco_carbon_reference), 0, 1)
    carbon_density = np.clip(
        canopy * (eco_carbon_native_advantage * native_cov + (1 - native_cov)) /
        (carbon_max_tha / eco_carbon_reference), 0, 1)

    cep = (ci_w.get('Cep_w_carbon',  0.35) * carbon_norm +
           ci_w.get('Cep_w_native',  0.35) * native_cov +
           ci_w.get('Cep_w_density', 0.30) * carbon_density)

    return (np.clip(ce, 0, 1), np.clip(cr, 0, 1),
            np.clip(ct, 0, 1), np.clip(cep, 0, 1))

def select_management_action(ce, cr, ct, cep, st, invasive_cov, params):
    """
    Select a single management action for the current annual step
    (single-action management mode).

    The management priority score is mps = w_e*ce + w_r*cr + w_t*ct + w_ep*cep
    using the service-target weights. Returns (action, mps, deficit).
    """
    eco_invasive_threshold = params['ecology'].get('eco_invasive_threshold', 0.4)

    st_w = params['service_target_weights'].get(st, {})
    w_e  = st_w.get('w_e',  0.25)
    w_r  = st_w.get('w_r',  0.25)
    w_t  = st_w.get('w_t',  0.25)
    w_ep = st_w.get('w_ep', 0.25)

    mps = w_e * ce + w_r * cr + w_t * ct + w_ep * cep

    deficit = {
        'ecological':        max(0, 0.7 - ce),
        'recreation':        max(0, 0.7 - cr),
        'conservation':      max(0, 0.7 - ct),
        'ecosystem_service': max(0, 0.7 - cep),
    }

    if invasive_cov > eco_invasive_threshold:
        action = 'invasive_removal'
    elif max(deficit.values()) < 0.05:
        action = 'monitoring'
    elif deficit['ecological']   == max(deficit.values()):
        action = 'ecological_restoration'
    elif deficit['recreation']   == max(deficit.values()):
        action = 'infrastructure'
    elif deficit['conservation'] == max(deficit.values()):
        action = 'visitor_experience'
    else:
        action = 'ecosystem_service'

    return action, mps, deficit


def apply_management_intervention(action, native_cov, invasive_cov, canopy,
                                   cr, ct, cep,
                                   eff_budget, eff_months, mgmt_scale, params):
    """
    Apply the selected single management action to the landscape. Modifies
    the ecological state variables and deducts the action-specific cost from
    the effective budget. The eff_months/8 ratio normalises gains relative
    to the temperate baseline 8-month management window.

    Returns (native_cov, invasive_cov, canopy, cr, ct, cep, remaining_budget).
    """
    intv = params['intervention']
    climate_scale = eff_months / 8.0

    if action == 'invasive_removal':
        rate    = intv.get('invrem_removal_rate', 0.08)
        boost   = intv.get('invrem_native_boost', 0.30)
        cost    = intv.get('invrem_cost', 20.0)
        removed = min(rate * climate_scale * mgmt_scale, invasive_cov)
        invasive_cov -= removed
        native_cov   += removed * boost

    elif action == 'ecological_restoration':
        ng   = intv.get('ecorest_native_gain', 0.04)
        inv  = intv.get('ecorest_invasive_reduction', 0.02)
        cost = intv.get('ecorest_cost', 15.0)
        native_cov   += ng  * climate_scale * mgmt_scale
        invasive_cov -= inv * climate_scale * mgmt_scale

    elif action == 'infrastructure':
        cost = intv.get('infra_cost', 8.0)
        cr  += intv.get('infra_recreation_gain', 0.05) * mgmt_scale

    elif action == 'visitor_experience':
        cost = intv.get('visitor_cost', 10.0)
        ct  += intv.get('visitor_ct_gain', 0.04) * mgmt_scale

    elif action == 'ecosystem_service':
        cost   = intv.get('ecosvc_cost', 8.0)
        cep   += intv.get('ecosvc_cep_gain', 0.03)    * mgmt_scale
        canopy += intv.get('ecosvc_canopy_gain', 0.02) * mgmt_scale

    else:   # monitoring
        cost = 0.0

    remaining_budget = max(0.0, eff_budget - cost)

    return (np.clip(native_cov,   0, 1),
            np.clip(invasive_cov, 0, 1),
            np.clip(canopy,       0, 1),
            np.clip(cr,           0, 1),
            np.clip(ct,           0, 1),
            np.clip(cep,          0, 1),
            remaining_budget)


def simulate_ecological_processes(native_cov, invasive_cov, canopy,
                                   park_type, cl, pt, params,
                                   temp_anomaly, precip_anomaly, rng,
                                   events=None, spread_mult=1.0,
                                   disturb_mult=1.0, propagule=0.0):
    """
    Simulate natural ecological processes for one annual step:
    invasive spread (logistic-type), drought-driven canopy stress,
    passive canopy recovery, passive native recovery, and -- when the
    climate-extreme disturbance module is active -- acute damage from
    discrete extreme events (heatwave, drought, storm, flood, pest).

    The discrete-event shocks are applied after the continuous processes
    so that they represent acute losses that subsequently relax back
    toward the park-type equilibrium through passive recovery.

    Returns (native_cov, invasive_cov, canopy, disturbance_info) where
    disturbance_info is a dict with the canopy and native losses caused by
    discrete extreme events this year.
    """
    eco = params['ecology']
    var = params['climate_variability']

    base_spread        = pt.get('base_spread', 0.05)
    invasive_mult      = cl.get('invasive_mult', 1.0)
    invasive_temp_sens = var.get('var_invasive_temp_sensitivity', 0.05)
    invasive_mult_adj  = invasive_mult * (1.0 + invasive_temp_sens * temp_anomaly)

    stochastic_factor = rng.uniform(
        eco.get('eco_spread_stochastic_min', 0.5),
        eco.get('eco_spread_stochastic_max', 1.5))
    spread_rate = base_spread * invasive_mult_adj * stochastic_factor * spread_mult

    eco_invasive_threshold = eco.get('eco_invasive_threshold', 0.4)
    if invasive_cov > eco_invasive_threshold:
        spread_rate *= eco.get('eco_density_self_limit', 0.5)

    invasive_cov += spread_rate * (1.0 - invasive_cov)

    # Propagule arrival from the surrounding matrix, zero unless the
    # landscape module supplies it.
    if propagule:
        invasive_cov += propagule * (1.0 - invasive_cov)

    drought_threshold = var.get('var_drought_threshold', 0.10)
    if precip_anomaly < -drought_threshold:
        excess = abs(precip_anomaly) - drought_threshold
        canopy -= cl.get('drought_stress', 0.05) * excess * rng.uniform(0.5, 1.5)

    canopy_equil = pt.get('canopy_init', 0.6)
    canopy      += eco.get('eco_passive_canopy_recovery', 0.02) * (canopy_equil - canopy)

    if invasive_cov < eco_invasive_threshold:
        passive = (eco.get('eco_passive_recovery_NRF', 0.015)
                   if park_type == 'NRF'
                   else eco.get('eco_passive_recovery_other', 0.005))
        native_cov   += passive
        invasive_cov -= passive * 0.5

    canopy += (var.get('var_canopy_precip_sensitivity', 0.02)
               * precip_anomaly * rng.uniform(0.5, 1.5))

    # --- Climate-extreme disturbance: acute damage + post-event regrowth ---
    dist_canopy_loss = 0.0
    dist_native_loss = 0.0
    dist = params['disturbance']
    if events and dist.get('ext_enable', 1.0) >= 0.5:
        dist_canopy_loss = (
            dist.get('ext_canopy_loss_heatwave', 0.04) * events.get('heatwave', 0.0) +
            dist.get('ext_canopy_loss_drought',  0.05) * events.get('drought',  0.0) +
            dist.get('ext_canopy_loss_storm',    0.09) * events.get('storm',    0.0) +
            dist.get('ext_canopy_loss_flood',    0.02) * events.get('heavy_rain', 0.0) +
            dist.get('ext_canopy_loss_pest',     0.03) * events.get('pest',     0.0))
        dist_native_loss = (
            dist.get('ext_native_loss_pest',     0.03) * events.get('pest',    0.0) +
            dist.get('ext_native_loss_drought',  0.005) * events.get('drought', 0.0))
        dist_canopy_loss *= disturb_mult
        dist_native_loss *= disturb_mult
        canopy     -= dist_canopy_loss
        native_cov -= dist_native_loss
        # invasive species often exploit disturbance-opened gaps
        invasive_cov += dist.get('ext_invasive_gap_gain', 0.02) * (
            events.get('storm', 0.0) + events.get('pest', 0.0)) * (1.0 - invasive_cov)
        # Enhanced post-disturbance regrowth toward the park-type canopy
        # equilibrium (the disturbance-recovery rate from the spec). Applied
        # only while the disturbance module is active, so ext_enable = 0
        # reproduces the original canopy dynamics exactly.
        canopy += dist.get('ext_recovery_rate', 0.06) * max(0.0,
                  pt.get('canopy_init', 0.6) - canopy)

    return (np.clip(native_cov,   0, 1),
            np.clip(invasive_cov, 0, 1),
            np.clip(canopy,       0, 1),
            {'canopy_loss': dist_canopy_loss, 'native_loss': dist_native_loss})


def simulate_visitor_dynamics(native_cov, invasive_cov, canopy, cl, yr,
                              social_params, params, rng, events=None):
    """
    Compute visitor satisfaction and the visitor-load penalty on Cr.
    When the disturbance module is active, heatwave / drought / storm
    events depress satisfaction.
    Returns (satisfaction, visitor_load_effect).
    """
    seasonal = cl.get('visitor_seas', 0.1) * np.sin(2 * np.pi * yr)
    satisfaction = np.clip(
        0.35 * native_cov +
        0.30 * (1 - invasive_cov) +
        0.35 * canopy +
        seasonal +
        rng.normal(0, 0.05),
        0, 1)

    if events:
        dist = params['disturbance']
        sat_loss = (dist.get('ext_satisfaction_loss_heatwave', 0.10) * events.get('heatwave', 0.0) +
                    dist.get('ext_satisfaction_loss_drought',  0.06) * events.get('drought',  0.0) +
                    dist.get('ext_satisfaction_loss_storm',    0.08) * events.get('storm',    0.0))
        satisfaction = float(np.clip(satisfaction - sat_loss, 0, 1))

    visitor_load_effect = 0.0
    if social_params:
        density  = social_params.get('park_visitor_density', 0)
        baseline = params['social_coupling'].get('soc_visitor_load_baseline', 50)
        if density > baseline:
            visitor_load_effect = params['social_coupling'].get(
                'soc_visitor_cr_penalty', 0.1)
    return satisfaction, visitor_load_effect


def simulate_community_engagement(park_type, social_params, params, rng):
    """
    Compute community-engagement bonuses to native recovery and invasive
    reduction when civic participation exceeds the engagement threshold.
    Returns (community_recovery_bonus, community_invasive_reduction).
    """
    soc = params['social_coupling']
    threshold = soc.get('soc_engagement_threshold', 0.3)
    recovery_bonus = invasive_reduction = 0.0
    if social_params:
        civic = social_params.get('civic_participation', 0)
        if civic > threshold:
            recovery_bonus     = soc.get('soc_community_recovery_bonus', 0.010) * civic
            invasive_reduction = soc.get('soc_community_invasive_reduction', 0.005) * civic
    return recovery_bonus, invasive_reduction


def detect_management_conflict(ce, satisfaction, invasive_cov, remaining_budget,
                                cl, social_params, params, rng, events=None):
    """
    Determine whether a management conflict event occurs in this annual step.
    When the disturbance module is active, pest outbreaks and storms add to
    the conflict probability (acute damage strains stakeholder relations).
    Returns True/False.
    """
    eco = params['ecology']
    cfl = params['conflict']
    soc = params['social_coupling']

    type_a = (ce < cfl.get('cfl_Ce_threshold', 0.5)) and \
             (satisfaction < cfl.get('cfl_sat_threshold', 0.5))
    type_b = invasive_cov > eco.get('eco_invasive_threshold', 0.4)
    type_c = remaining_budget < 0

    condition_score = int(type_a or type_b or type_c)

    conflict_amp = cl.get('conflict_amp', 0.3)
    if social_params:
        regulatory_strength = min(
            1.0,
            social_params.get('gdp_per_capita', 25000) /
            soc.get('soc_budget_gdp_reference', 40000))
        if regulatory_strength > 0.5:
            conflict_amp += (soc.get('soc_regulatory_sensitivity', 0.15)
                             * (regulatory_strength - 0.5))

    prob = condition_score * conflict_amp + rng.uniform(
        0, cfl.get('cfl_stochastic_range', 0.10)
    )

    if events:
        dist = params['disturbance']
        prob += (dist.get('ext_conflict_pest',  0.12) * events.get('pest',  0.0) +
                 dist.get('ext_conflict_storm', 0.06) * events.get('storm', 0.0))

    return rng.uniform(0, 1) < prob

#----------------------------------------------------------------------------
# Section 2b: Climate-extreme disturbance sampling
#----------------------------------------------------------------------------
def generate_extreme_events(yr, temp_anomaly, precip_anomaly, cl, params, rng):
    """
    Sample discrete climate-extreme events for the current year.

    Five event types are considered: heatwave, drought, heavy rainfall /
    flood, storm damage, and pest / disease outbreak. For each type an
    occurrence probability is formed from a base annual frequency, a
    sensitivity to the current temperature / precipitation anomaly, a
    contribution from the climate-zone modifiers already present in the
    parameter table (heat_urgency, drought_stress, invasive_mult), and an
    optional warming-driven trend that raises frequencies over time. If an
    event occurs, an intensity in [0, 1] is drawn.

    Returns a dict with one intensity per event type (0.0 if the event did
    not occur). When ext_enable < 0.5 the function returns all-zero
    intensities WITHOUT drawing any random numbers, so the disabled module
    reproduces the original random-number stream exactly.
    """
    events = {'heatwave': 0.0, 'drought': 0.0, 'heavy_rain': 0.0,
              'storm': 0.0, 'pest': 0.0}

    dist = params['disturbance']
    if dist.get('ext_enable', 1.0) < 0.5:
        return events

    warm = dist.get('ext_warming_freq_trend', 0.002) * yr
    imin = dist.get('ext_intensity_min', 0.30)
    imax = dist.get('ext_intensity_max', 1.00)

    def _intensity(scale=1.0):
        return float(np.clip(rng.uniform(imin, imax) * scale, 0.0, 1.0))

    # Heatwave -- more likely and stronger under positive temperature anomaly
    heat_p = (dist.get('ext_heatwave_freq', 0.10) +
              dist.get('ext_heatwave_temp_sensitivity', 0.12) * max(0.0, temp_anomaly) +
              dist.get('ext_heat_zone_coupling', 0.15) * cl.get('heat_urgency', 0.10) + warm)
    if rng.uniform(0, 1) < np.clip(heat_p, 0, 1):
        events['heatwave'] = _intensity(1.0 + 0.5 * max(0.0, temp_anomaly))

    # Drought -- more likely and stronger under negative precipitation anomaly
    drought_p = (dist.get('ext_drought_freq', 0.12) +
                 dist.get('ext_drought_precip_sensitivity', 0.50) * max(0.0, -precip_anomaly) +
                 dist.get('ext_drought_zone_coupling', 0.20) * cl.get('drought_stress', 0.10) + warm)
    if rng.uniform(0, 1) < np.clip(drought_p, 0, 1):
        events['drought'] = _intensity(1.0 + 0.5 * max(0.0, -precip_anomaly))

    # Heavy rainfall / flood -- more likely under positive precipitation anomaly
    rain_p = (dist.get('ext_heavy_rain_freq', 0.12) +
              dist.get('ext_heavy_rain_precip_sensitivity', 0.50) * max(0.0, precip_anomaly) +
              warm)
    if rng.uniform(0, 1) < np.clip(rain_p, 0, 1):
        events['heavy_rain'] = _intensity(1.0 + 0.5 * max(0.0, precip_anomaly))

    # Storm damage -- weather-driven, largely climate-independent base rate
    storm_p = dist.get('ext_storm_prob', 0.08) + warm
    if rng.uniform(0, 1) < np.clip(storm_p, 0, 1):
        events['storm'] = _intensity()

    # Pest / disease outbreak -- favoured by warmth and high invasive context
    pest_p = (dist.get('ext_pest_prob', 0.07) +
              dist.get('ext_pest_temp_sensitivity', 0.05) * max(0.0, temp_anomaly) +
              dist.get('ext_pest_zone_coupling', 0.02) * cl.get('invasive_mult', 1.0) + warm)
    if rng.uniform(0, 1) < np.clip(pest_p, 0, 1):
        events['pest'] = _intensity()

    return events

def generate_climate_anomaly(yr, site_climate, params, rng, cl=None):
    """
    Generate temperature and precipitation anomalies for the current year and
    sample any discrete climate-extreme events.

    Returns dict {'temp_anomaly', 'precip_anomaly', 'events'} where 'events'
    is the per-type intensity dict from generate_extreme_events(). The two
    continuous anomaly draws happen first, so the disturbance module (which
    draws afterwards, and only when enabled) does not perturb the historical
    anomaly stream.
    """
    var = params['climate_variability']
    temp_anomaly = (rng.normal(0, var.get('var_temp_anomaly_sd', 0.5))
                    + var.get('var_warming_rate', 0.02) * yr)
    precip_anomaly = rng.normal(0, var.get('var_precip_anomaly_sd', 0.3))
    events = generate_extreme_events(yr, temp_anomaly, precip_anomaly,
                                     cl or {}, params, rng)
    return {'temp_anomaly': temp_anomaly,
            'precip_anomaly': precip_anomaly,
            'events': events}

#----------------------------------------------------------------------------
# Section 3: Management portfolio optimization module
#----------------------------------------------------------------------------
# Default action costs and per-cycle effects are read from params; the
# portfolio module reuses the same intervention parameters as the single-
# action mode, so a portfolio that places 100% weight on one action is
# numerically equivalent to selecting that action in single mode.

def default_portfolio(params):
    """
    Return the documented default budget-share portfolio
    (invasive removal 0.30, ecological restoration 0.25,
    ecosystem service 0.20, infrastructure 0.15, monitoring 0.10).
    Values can be overridden by opt_share_<action> parameters and are
    renormalised to sum to 1.
    """
    opt = params['optimization']
    raw = {
        'monitoring':             opt.get('opt_share_monitoring', 0.10),
        'invasive_removal':       opt.get('opt_share_invasive_removal', 0.30),
        'ecological_restoration': opt.get('opt_share_ecological_restoration', 0.25),
        'infrastructure':         opt.get('opt_share_infrastructure', 0.15),
        'visitor_experience':     opt.get('opt_share_visitor_experience', 0.00),
        'ecosystem_service':      opt.get('opt_share_ecosystem_service', 0.20),
    }
    total = sum(raw.values()) or 1.0
    return {a: raw[a] / total for a in action_order}

def _portfolio_bounds(params):
    """Per-action (min_share, max_share) bounds from opt_min_/opt_max_."""
    opt = params['optimization']
    default_max = {
        'monitoring': 0.40, 'invasive_removal': 0.70,
        'ecological_restoration': 0.60, 'infrastructure': 0.50,
        'visitor_experience': 0.40, 'ecosystem_service': 0.50,
    }
    bounds = {}
    for a in action_order:
        lo = opt.get(f'opt_min_{a}', 0.0)
        hi = opt.get(f'opt_max_{a}', default_max.get(a, 0.6))
        bounds[a] = (max(0.0, lo), min(1.0, max(lo, hi)))
    return bounds

def _sample_portfolio(bounds, rng):
    """
    Draw a random feasible budget-share vector. Shares are sampled within
    their per-action bounds, normalised to sum to 1, then clipped against
    the upper bounds and renormalised (soft constraint satisfaction).
    """
    los = np.array([bounds[a][0] for a in action_order])
    his = np.array([bounds[a][1] for a in action_order])
    raw = np.array([rng.uniform(bounds[a][0], bounds[a][1]) for a in action_order])
    if raw.sum() <= 0:
        raw = np.ones(len(action_order))
    shares = raw / raw.sum()
    for _ in range(4):
        over = shares > his + 1e-9
        if not over.any():
            break
        shares = np.minimum(shares, his)
        deficit = 1.0 - shares.sum()
        free = ~over
        if deficit > 0 and free.any():
            pool = shares[free].sum()
            if pool > 0:
                shares[free] += deficit * (shares[free] / pool)
            else:
                shares[free] += deficit / free.sum()
        else:
            break
    s = shares.sum()
    if s > 0:
        shares = shares / s
    return {a: float(shares[i]) for i, a in enumerate(action_order)}

def _project_to_bounds(shares, bounds):
    """
    Project an arbitrary share vector onto the feasible set defined by the
    per-action [min, max] bounds with a sum-to-one constraint, using an
    iterative clip-and-redistribute (water-filling) scheme. This guarantees
    that every evaluated portfolio -- including the structured seed
    candidates -- respects the configured opt_min_/opt_max_ shares.
    """
    los = np.array([bounds[a][0] for a in action_order])
    his = np.array([bounds[a][1] for a in action_order])
    x = np.array([max(0.0, shares.get(a, 0.0)) for a in action_order], dtype=float)
    if x.sum() <= 0:
        x = np.ones(len(action_order))
    x = x / x.sum()
    for _ in range(8):
        x = np.clip(x, los, his)
        s = x.sum()
        if abs(s - 1.0) < 1e-9:
            break
        if s > 1.0:
            room = x - los
            total = room.sum()
            if total <= 1e-12:
                break
            x = x - (s - 1.0) * (room / total)
        else:
            room = his - x
            total = room.sum()
            if total <= 1e-12:
                break
            x = x + (1.0 - s) * (room / total)
    x = np.clip(x, los, his)
    s = x.sum()
    if s > 0:
        x = x / s
    return {a: float(x[i]) for i, a in enumerate(action_order)}


def apply_management_portfolio(shares, native_cov, invasive_cov, canopy,
                               cr, ct, cep,
                               eff_budget, eff_months, mgmt_scale, params):
    """
    Apply a budget-share portfolio: every action contributes its per-cycle
    effect scaled by its budget share (and by the climate window and the
    joint management factor). The portfolio cost is the share-weighted sum
    of the individual action costs and is deducted from the effective budget.

    Returns (native_cov, invasive_cov, canopy, cr, ct, cep,
             remaining_budget, portfolio_cost).
    """
    intv = params['intervention']
    climate_scale = eff_months / 8.0

    s_inv  = shares.get('invasive_removal', 0.0)
    s_eco  = shares.get('ecological_restoration', 0.0)
    s_infra = shares.get('infrastructure', 0.0)
    s_vis  = shares.get('visitor_experience', 0.0)
    s_es   = shares.get('ecosystem_service', 0.0)

    if s_inv > 0:
        removed = min(intv.get('invrem_removal_rate', 0.08) *
                      climate_scale * mgmt_scale * s_inv, invasive_cov)
        invasive_cov -= removed
        native_cov   += removed * intv.get('invrem_native_boost', 0.30)

    if s_eco > 0:
        native_cov   += intv.get('ecorest_native_gain', 0.04) * climate_scale * mgmt_scale * s_eco
        invasive_cov -= intv.get('ecorest_invasive_reduction', 0.02) * climate_scale * mgmt_scale * s_eco

    if s_infra > 0:
        cr += intv.get('infra_recreation_gain', 0.05) * mgmt_scale * s_infra

    if s_vis > 0:
        ct += intv.get('visitor_ct_gain', 0.04) * mgmt_scale * s_vis

    if s_es > 0:
        cep    += intv.get('ecosvc_cep_gain', 0.03) * mgmt_scale * s_es
        canopy += intv.get('ecosvc_canopy_gain', 0.02) * mgmt_scale * s_es

    cost = (s_inv  * intv.get('invrem_cost', 20.0) +
            s_eco  * intv.get('ecorest_cost', 15.0) +
            s_infra * intv.get('infra_cost', 8.0) +
            s_vis  * intv.get('visitor_cost', 10.0) +
            s_es   * intv.get('ecosvc_cost', 8.0))

    remaining_budget = max(0.0, eff_budget - cost)

    return (np.clip(native_cov,   0, 1),
            np.clip(invasive_cov, 0, 1),
            np.clip(canopy,       0, 1),
            np.clip(cr,           0, 1),
            np.clip(ct,           0, 1),
            np.clip(cep,          0, 1),
            remaining_budget, cost)


def _portfolio_utility(ce, cr, ct, cep, satisfaction, invasive_cov,
                       cost, eff_budget, params):
    """
    Scalar utility of a one-year portfolio outcome:
        + w_ce*Ce + w_cr*Cr + w_ct*Ct + w_cep*Cep + w_sat*satisfaction
        - w_invasive*invasive - w_conflict*conflict_proxy - w_cost*cost_norm
    The conflict proxy rises as invasive cover exceeds the ecological
    threshold and as Ce falls below the conflict threshold.
    """
    o = params['optimization']
    thr = params['ecology'].get('eco_invasive_threshold', 0.4)
    ce_thr = params['conflict'].get('cfl_Ce_threshold', 0.5)
    conflict_proxy = max(0.0, invasive_cov - thr) + max(0.0, ce_thr - ce)
    cost_norm = (cost / eff_budget) if eff_budget > 0 else 0.0
    return (o.get('opt_w_ce', 1.0)  * ce +
            o.get('opt_w_cr', 0.5)  * cr +
            o.get('opt_w_ct', 0.5)  * ct +
            o.get('opt_w_cep', 0.8) * cep +
            o.get('opt_w_sat', 0.8) * satisfaction -
            o.get('opt_w_invasive', 1.0) * invasive_cov -
            o.get('opt_w_conflict', 0.8) * conflict_proxy -
            o.get('opt_w_cost', 0.3) * cost_norm)


def optimize_portfolio(state, cl, eff_budget, eff_months, mgmt_scale,
                       params, rng, method='random'):
    """
    Choose a budget-share portfolio that maximises the one-step utility.

    state  : (native_cov, invasive_cov, canopy, cr, ct, cep) at decision time.
    method : 'random' (random search, default), 'scipy' (random search then
             SLSQP refinement), or 'fixed' (return the default portfolio).

    Each candidate is evaluated by applying it to a copy of the current
    state, recomputing the condition indices, estimating satisfaction from
    the post-intervention landscape, and scoring the result with
    _portfolio_utility(). Returns (best_shares, best_utility).
    """
    if method == 'fixed':
        return default_portfolio(params), np.nan

    native_cov, invasive_cov, canopy, cr, ct, cep = state

    def _evaluate(shares):
        (nn, ii, cc, rr, tt, pp, _rem, cost) = apply_management_portfolio(
            shares, native_cov, invasive_cov, canopy, cr, ct, cep,
            eff_budget, eff_months, mgmt_scale, params)
        ece, ecr, ect, ecep = compute_condition_indices(nn, ii, cc, cl, params)
        sat_proxy = float(np.clip(0.35 * nn + 0.30 * (1 - ii) + 0.35 * cc, 0, 1))
        return _portfolio_utility(ece, ecr, ect, ecep, sat_proxy, ii,
                                  cost, eff_budget, params), cost

    bounds = _portfolio_bounds(params)
    n_samples = int(params['optimization'].get('opt_n_samples', 60))

    # Structured seed candidates ensure good coverage of the simplex; each is
    # projected onto the feasible (bounded) set so a seed cannot violate the
    # configured per-action max shares.
    seeds = [default_portfolio(params),
             {a: (1.0 if a == 'invasive_removal' else 0.0) for a in action_order},
             {a: 1.0 / len(action_order) for a in action_order}]
    candidates = [_project_to_bounds(s, bounds) for s in seeds]
    candidates += [_sample_portfolio(bounds, rng) for _ in range(n_samples)]

    best_shares, best_u = None, -1e18
    for sh in candidates:
        u, _cost = _evaluate(sh)
        if u > best_u:
            best_u, best_shares = u, sh

    if method == 'scipy':
        try:
            from scipy.optimize import minimize
            x0 = np.array([best_shares[a] for a in action_order])
            lo = [bounds[a][0] for a in action_order]
            hi = [bounds[a][1] for a in action_order]

            def neg_u(x):
                x = np.clip(x, 0, None)
                if x.sum() <= 0:
                    return 1e9
                x = x / x.sum()
                sh = {a: float(x[i]) for i, a in enumerate(action_order)}
                u, _ = _evaluate(sh)
                return -u

            res = minimize(neg_u, x0, method='SLSQP',
                           bounds=list(zip(lo, hi)),
                           constraints=[{'type': 'eq',
                                         'fun': lambda x: x.sum() - 1.0}],
                           options={'maxiter': 60, 'ftol': 1e-4})
            if res.success and -res.fun > best_u:
                x = np.clip(res.x, 0, None)
                x = x / x.sum() if x.sum() > 0 else x
                best_shares = {a: float(x[i]) for i, a in enumerate(action_order)}
                best_u = -res.fun
        except Exception:
            pass   # fall back silently to the random-search optimum

    return best_shares, best_u


#----------------------------------------------------------------------------
# Section 4: Annual loop and replicate runner
#----------------------------------------------------------------------------
def simulate_ufp(park_type, service_target, climate_zone,
                 n_years, budget, n_managers, run_id, params, rng,
                 social_params=None, management_mode='single',
                 portfolio_method='random', landscape=None):
    """
    Simulate one urban forest park for n_years and return a pd.DataFrame
    of annual records.

    management_mode  : 'single'    -> one dominant action per year (original
                                      UFPM-SMF rule), or
                       'portfolio' -> annual budget allocated across actions
                                      via the portfolio optimization module.
    portfolio_method : 'random' | 'scipy' | 'fixed' (used only in portfolio
                       mode).
    """
    pt  = params['park_type_params'].get(park_type, {})
    cl  = params['climate_params'].get(climate_zone, {})
    eco = params['ecology']
    dist = params['disturbance']

    native_cov   = np.clip(pt.get('native_init_pct',  0.5)
                           + rng.normal(0, eco.get('eco_init_perturbation_sd', 0.03)),
                           0, 1)
    invasive_cov = np.clip(pt.get('invasive_init_pct', 0.2)
                           + rng.normal(0, eco.get('eco_init_perturbation_sd', 0.03)),
                           0, 1)
    canopy_start = pt.get('canopy_init', 0.6)
    if landscape and 'canopy_observed' in landscape:
        canopy_start = landscape['canopy_observed']
    canopy       = np.clip(canopy_start
                           + rng.normal(0, eco.get('eco_init_perturbation_sd', 0.03)),
                           0, 1)

    gdp_reference     = params['social_coupling'].get('soc_budget_gdp_reference', 25000)
    budget_elasticity = params['social_coupling'].get('soc_budget_elasticity', 0.5)
    gdp_scale = (social_params.get('gdp_per_capita', gdp_reference) / gdp_reference) \
                ** budget_elasticity if social_params else 1.0

    eff_budget_base = budget * gdp_scale
    budget_adequate = params['social_coupling'].get('soc_budget_adequate', 100.0)
    budget_factor   = float(np.clip(eff_budget_base / budget_adequate, 0, 1))
    mgmt_scale      = min(1.0, n_managers / 5.0) * budget_factor
    eff_months      = int(cl.get('mgmt_window', 8))

    # --- Landscape module ---------------------------------------------------
    # terms are neutral when landscape is None, so the model reduces exactly to
    # the non-spatial version.
    lst = landscape_terms(landscape, params)
    ls_par = params['landscape']
    phi_e = float(np.clip(lst['edge_fraction'], 0.0, 0.95))
    use_zones = landscape is not None and phi_e > 0.0

    # Management capacity is diluted by the canopy edge each staff member has
    # to cover, and by how hard that edge is to reach.
    mgmt_scale = mgmt_scale / lst['workload']

    edge_spread   = 1.0 + ls_par.get('lsc_edge_spread', 0.0) * lst['kappa']
    edge_priority = ls_par.get('lsc_edge_priority', 0.0)
    diffusion     = ls_par.get('lsc_diffusion', 0.0)

    # Zone state starts from the park-level draw; the zones diverge from there.
    nat_e, inv_e, can_e = native_cov, invasive_cov, canopy
    nat_c, inv_c, can_c = native_cov, invasive_cov, canopy

    # Dedicated RNG for portfolio search keeps the ecological RNG stream
    # independent of the number of optimisation samples. Only created in
    # portfolio mode, so single mode reproduces the original stream.
    opt_rng = None
    if management_mode == 'portfolio':
        opt_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))

    storm_extra_cost = dist.get('ext_extra_cost_storm', 12.0)
    flood_infra_pressure = dist.get('ext_infra_pressure_flood', 0.05)

    results = []
    for yr in range(n_years):
        anom = generate_climate_anomaly(yr, {}, params, rng, cl)
        ta   = anom['temp_anomaly']
        pa   = anom['precip_anomaly']
        ev   = anom['events']

        ce, cr, ct, cep = compute_condition_indices(
            native_cov, invasive_cov, canopy, cl, params)
        nat_prev, inv_prev, can_prev = native_cov, invasive_cov, canopy

        # --- Management module: single action or optimised portfolio -------
        alloc = {a: 0.0 for a in action_order}
        if management_mode == 'portfolio':
            shares, _util = optimize_portfolio(
                (native_cov, invasive_cov, canopy, cr, ct, cep),
                cl, eff_budget_base, eff_months, mgmt_scale,
                params, opt_rng, method=portfolio_method)
            (native_cov, invasive_cov, canopy,
             cr, ct, cep, eff_budget, _pcost) = apply_management_portfolio(
                shares, native_cov, invasive_cov, canopy, cr, ct, cep,
                eff_budget_base, eff_months, mgmt_scale, params)
            alloc = shares
            action = max(action_order, key=lambda a: shares.get(a, 0.0))
        else:
            action, mps, _ = select_management_action(
                ce, cr, ct, cep, service_target, invasive_cov, params)
            (native_cov, invasive_cov, canopy,
             cr, ct, cep, eff_budget) = apply_management_intervention(
                action, native_cov, invasive_cov, canopy, cr, ct, cep,
                eff_budget_base, eff_months, mgmt_scale, params)
            alloc[action] = 1.0

        # --- Ecological processes (incl. extreme-event acute damage) -------
        if use_zones:
            # Push the management outcome back into the two zones. Invasive
            # removal is steered toward whichever zone is more invaded; the
            # other effects are spread evenly over the park.
            d_nat = native_cov - nat_prev
            d_inv = invasive_cov - inv_prev
            d_can = canopy - can_prev
            psi = float(np.clip(phi_e + edge_priority * (inv_e - inv_c), 0.0, 1.0))
            nat_e, nat_c = nat_e + d_nat, nat_c + d_nat
            can_e, can_c = can_e + d_can, can_c + d_can
            inv_e += d_inv * psi / max(phi_e, 1e-6) * lst['removal_scale']
            inv_c += d_inv * (1.0 - psi) / max(1.0 - phi_e, 1e-6) * lst['removal_scale']

            nat_e, inv_e, can_e, dinfo_e = simulate_ecological_processes(
                nat_e, inv_e, can_e, park_type, cl, pt, params, ta, pa, rng,
                events=ev, spread_mult=edge_spread,
                disturb_mult=lst['disturb_edge'], propagule=lst['propagule'])
            nat_c, inv_c, can_c, dinfo_c = simulate_ecological_processes(
                nat_c, inv_c, can_c, park_type, cl, pt, params, ta, pa, rng,
                events=ev)

            # Invasives leak inward from the edge; the exchange conserves cover.
            flux = diffusion * (inv_e - inv_c)
            inv_e = float(np.clip(inv_e - flux * (1.0 - phi_e), 0, 1))
            inv_c = float(np.clip(inv_c + flux * phi_e, 0, 1))

            native_cov   = phi_e * nat_e + (1.0 - phi_e) * nat_c
            invasive_cov = phi_e * inv_e + (1.0 - phi_e) * inv_c
            canopy       = phi_e * can_e + (1.0 - phi_e) * can_c
            dinfo = {'canopy_loss': phi_e * dinfo_e['canopy_loss']
                                    + (1.0 - phi_e) * dinfo_c['canopy_loss'],
                     'native_loss': phi_e * dinfo_e['native_loss']
                                    + (1.0 - phi_e) * dinfo_c['native_loss']}
        else:
            native_cov, invasive_cov, canopy, dinfo = simulate_ecological_processes(
                native_cov, invasive_cov, canopy, park_type, cl, pt,
                params, ta, pa, rng, events=ev)

        # Storm damage imposes an extra remediation cost on the annual budget.
        extra_cost = storm_extra_cost * ev.get('storm', 0.0)
        eff_budget = max(0.0, eff_budget - extra_cost)

        # --- Visitor dynamics (incl. extreme-event satisfaction loss) ------
        satisfaction, vload = simulate_visitor_dynamics(
            native_cov, invasive_cov, canopy, cl, yr,
            social_params, params, rng, events=ev)
        cr = max(0.0, cr - vload)

        # Heavy rainfall / flood increases infrastructure pressure (Cr penalty).
        cr = max(0.0, cr - flood_infra_pressure * ev.get('heavy_rain', 0.0))

        bonus, inv_red = simulate_community_engagement(
            park_type, social_params, params, rng)
        native_cov   = np.clip(native_cov + bonus,   0, 1)
        invasive_cov = np.clip(invasive_cov - inv_red, 0, 1)

        conflict = detect_management_conflict(
            ce, satisfaction, invasive_cov, eff_budget, cl,
            social_params, params, rng, events=ev)

        ce, cr, ct, cep = compute_condition_indices(
            native_cov, invasive_cov, canopy, cl, params)
        st_w = params['service_target_weights'].get(service_target, {})
        mps  = (st_w.get('w_e',  0.25) * ce  +
                st_w.get('w_r',  0.25) * cr  +
                st_w.get('w_t',  0.25) * ct  +
                st_w.get('w_ep', 0.25) * cep)

        record = {
            'run_id':         run_id,
            'park_type':      park_type,
            'service_target': service_target,
            'climate_zone':   climate_zone,
            'year':           yr,
            'native_cover':   native_cov,
            'invasive_cover': invasive_cov,
            'canopy_cover':   canopy,
            'Ce':             ce,
            'Cr':             cr,
            'Ct':             ct,
            'Cep':            cep,
            'MPS':            mps,
            'satisfaction':   satisfaction,
            'action':         action,
            'mgmt_mode':      management_mode,
            'conflict_total': int(conflict),
            'budget_used':    eff_budget_base - eff_budget,
            'temp_anomaly':   ta,
            'precip_anomaly': pa,
            # Climate-extreme disturbance diagnostics
            'heatwave':       ev.get('heatwave', 0.0),
            'drought_event':  ev.get('drought', 0.0),
            'heavy_rain':     ev.get('heavy_rain', 0.0),
            'storm':          ev.get('storm', 0.0),
            'pest_outbreak':  ev.get('pest', 0.0),
            'disturb_canopy_loss': dinfo['canopy_loss'],
            'disturb_native_loss': dinfo['native_loss'],
            'edge_fraction':   phi_e,
            'invasive_edge':   inv_e if use_zones else invasive_cov,
            'invasive_core':   inv_c if use_zones else invasive_cov,
            'canopy_edge':     can_e if use_zones else canopy,
            'canopy_core':     can_c if use_zones else canopy,
        }
        # Per-action budget allocation (one-hot in single mode)
        for a in action_order:
            record[f'alloc_{a}'] = alloc.get(a, 0.0)

        results.append(record)

    return pd.DataFrame(results)


def run_replicates(park_type, service_target, climate_zone,
                   n_reps, n_years, budget, n_managers, seed, params,
                   social_params=None, management_mode='single',
                   portfolio_method='random', landscape=None):
    """
    Run multiple stochastic replicates. Replicate r uses seed + r * 7919.
    """
    out = []
    for rep in range(n_reps):
        rng = np.random.RandomState(seed + rep * 7919)
        out.append(simulate_ufp(park_type, service_target, climate_zone,
                                n_years, budget, n_managers,
                                run_id=rep, params=params, rng=rng,
                                social_params=social_params,
                                management_mode=management_mode,
                                portfolio_method=portfolio_method,
                                landscape=landscape))
    return pd.concat(out, ignore_index=True)

#----------------------------------------------------------------------------
# Section 5: Sensitivity analysis
#----------------------------------------------------------------------------
def sensitivity_analysis(park_type, service_target, climate_zone, params,
                         budget_range, n_managers_range,
                         n_reps, n_years, seed, output_dir,
                         social_params=None, management_mode='single',
                         portfolio_method='random'):
    """
    Grid sweep over budget and n_managers. Writes
    output_dir/Sensitivity_analysis.csv and returns the DataFrame.
    """
    rows = []
    for budget in budget_range:
        for n_managers in n_managers_range:
            df = run_replicates(park_type, service_target, climate_zone,
                                n_reps, n_years, budget, n_managers,
                                seed, params, social_params=social_params,
                                management_mode=management_mode,
                                portfolio_method=portfolio_method)
            end = df[df['year'] == n_years - 1]
            rows.append({
                'budget':         budget,
                'n_managers':     n_managers,
                'final_Ce':       end['Ce'].mean(),
                'final_invasive': end['invasive_cover'].mean(),
                'final_conflict': end['conflict_total'].mean(),
            })

    df_sens = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    df_sens.to_csv(os.path.join(output_dir, 'Sensitivity_analysis.csv'),
                   index=False, encoding='utf-8-sig')
    return df_sens


def sobol_sensitivity(site_cfg, params, output_dir,
                      n_base=64, n_years=50, n_reps=3, seed=2024):
    """
    Variance-based (Sobol) global sensitivity analysis for one site.
    Sixteen parameters are varied within plausible bounds defined below;
    the three evaluated outputs are final Ce, final invasive cover, and
    mean annual conflict frequency. Writes <site_id>_sobol_indices.csv
    and returns the indices as a DataFrame.
    """
    from SALib.sample.sobol import sample as sobol_sample
    from SALib.analyze.sobol import analyze as sobol_analyze
    from copy import deepcopy

    park_type      = site_cfg['park_type']
    service_target = site_cfg['service_target']
    climate_zone   = site_cfg['climate_zone']
    site_id        = site_cfg.get('site_id', 'site')

    social_params = {k: site_cfg[k] for k in
                     ('gdp_per_capita', 'civic_participation',
                      'park_visitor_density', 'park_area')
                     if k in site_cfg} or None

    problem = {
        'num_vars': 16,
        'names': [
            'budget',                       'n_managers',
            'invrem_removal_rate',          'invrem_native_boost',
            'ecorest_native_gain',          'eco_invasive_threshold',
            'eco_passive_recovery_other',   'eco_passive_canopy_recovery',
            'var_warming_rate',             'var_temp_anomaly_sd',
            'var_precip_anomaly_sd',        'soc_budget_elasticity',
            'soc_engagement_threshold',     'pt_native_init_pct',
            'pt_invasive_init_pct',         'cl_invasive_mult',
        ],
        'bounds': [
            [50.0, 150.0], [1.0, 5.0],
            [0.04, 0.16],  [0.10, 0.50],
            [0.02, 0.08],  [0.30, 0.55],
            [0.002, 0.020],[0.005, 0.040],
            [0.00, 0.05],  [0.20, 1.00],
            [0.10, 0.60],  [0.20, 0.80],
            [0.10, 0.50],  [0.30, 0.95],
            [0.05, 0.70],  [0.50, 2.00],
        ],
    }

    samples = sobol_sample(problem, n_base, calc_second_order=False)
    rng_master = np.random.RandomState(seed)
    seeds = rng_master.randint(0, 10**6, size=len(samples))

    print(f"[Sobol] {site_id}: {len(samples)} model evaluations "
          f"(n_years={n_years}, n_reps={n_reps})")

    outputs = {'final_Ce': [], 'final_invasive': [], 'conflict_rate': []}
    for i, row in enumerate(samples):
        p = deepcopy(params)
        p['intervention']['invrem_removal_rate']         = row[2]
        p['intervention']['invrem_native_boost']         = row[3]
        p['intervention']['ecorest_native_gain']         = row[4]
        p['ecology']['eco_invasive_threshold']           = row[5]
        p['ecology']['eco_passive_recovery_other']       = row[6]
        p['ecology']['eco_passive_canopy_recovery']      = row[7]
        p['climate_variability']['var_warming_rate']     = row[8]
        p['climate_variability']['var_temp_anomaly_sd']  = row[9]
        p['climate_variability']['var_precip_anomaly_sd']= row[10]
        p['social_coupling']['soc_budget_elasticity']    = row[11]
        p['social_coupling']['soc_engagement_threshold'] = row[12]
        if park_type in p['park_type_params']:
            p['park_type_params'][park_type]['native_init_pct']   = row[13]
            p['park_type_params'][park_type]['invasive_init_pct'] = row[14]
        if climate_zone in p['climate_params']:
            p['climate_params'][climate_zone]['invasive_mult']    = row[15]

        budget_i     = float(row[0])
        n_managers_i = max(1, int(round(row[1])))

        ce_runs, inv_runs, cf_runs = [], [], []
        for r in range(n_reps):
            rng = np.random.RandomState(int(seeds[i]) + r * 7919)
            df_r = simulate_ufp(park_type, service_target, climate_zone,
                                n_years, budget_i, n_managers_i,
                                run_id=r, params=p, rng=rng,
                                social_params=social_params)
            end = df_r[df_r['year'] == n_years - 1]
            ce_runs.append(end['Ce'].mean())
            inv_runs.append(end['invasive_cover'].mean())
            cf_runs.append(df_r['conflict_total'].mean())

        outputs['final_Ce'].append(np.mean(ce_runs))
        outputs['final_invasive'].append(np.mean(inv_runs))
        outputs['conflict_rate'].append(np.mean(cf_runs))

        if (i + 1) % 50 == 0:
            print(f"  [Sobol] {site_id}: {i + 1}/{len(samples)} done")

    rows = []
    for out_name, y in outputs.items():
        si = sobol_analyze(problem, np.array(y),
                           calc_second_order=False, print_to_console=False)
        for j, pname in enumerate(problem['names']):
            rows.append({
                'parameter': pname,
                'output':    out_name,
                'S1':        si['S1'][j],
                'S1_conf':   si['S1_conf'][j],
                'ST':        si['ST'][j],
                'ST_conf':   si['ST_conf'][j],
            })

    df_idx = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{site_id}_sobol_indices.csv")
    df_idx.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"[Sobol] {site_id}: indices saved to {out_path}")
    return df_idx

#----------------------------------------------------------------------------
# Section 6: Early-warning and resilience diagnosis module
#----------------------------------------------------------------------------
def _slope(y):
    """Ordinary-least-squares slope of y against its integer index."""
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return 0.0
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _lag1_autocorr(y):
    """Lag-1 autocorrelation of a linearly detrended series."""
    y = np.asarray(y, dtype=float)
    if len(y) < 4:
        return 0.0
    trend = np.polyval(np.polyfit(np.arange(len(y)), y, 1), np.arange(len(y)))
    res = y - trend
    if np.std(res) < 1e-9:
        return 0.0
    return float(np.corrcoef(res[:-1], res[1:])[0, 1])


def compute_early_warning(sim_df, window=10, params=None):
    """
    Diagnose whether a park is approaching a degraded ecological state from
    its simulated annual time series, using rolling-window critical-slowing-
    down indicators computed on the across-replicate mean trajectory:

        * declining Ce trend over the recent window,
        * rising temporal variance of Ce (late vs early window),
        * rising lag-1 autocorrelation of Ce (recent window),
        * repeated / sustained invasive-threshold crossings,
        * a rising and elevated conflict frequency,
        * a sub-failure final Ce.

    Each triggered indicator contributes to a score that maps to a warning
    level: 'stable', 'watch', 'warning', or 'critical'. A hard 'critical'
    is also raised when the failure definition is met (invasive cover above
    threshold for >= ew_consecutive_years AND degraded final Ce).

    Parameters can override window/thresholds via the 'early_warning' group.

    Returns a dict: {window, warning_level, score, flags, indicators}.
    """
    ew = (params or {}).get('early_warning', {}) if params else {}
    window = int(ew.get('ew_window', window))
    ce_fail        = ew.get('ew_ce_fail', 0.5)
    inv_thr        = ew.get('ew_invasive_threshold', 0.4)
    consec_req     = int(ew.get('ew_consecutive_years', 5))
    slope_thr      = ew.get('ew_ce_trend_slope', -0.002)
    var_ratio_thr  = ew.get('ew_var_ratio', 1.2)
    ac_thr         = ew.get('ew_ac_threshold', 0.5)
    conflict_warn  = ew.get('ew_conflict_rate_warn', 0.5)
    watch_score    = ew.get('ew_watch_score', 1)
    warning_score  = ew.get('ew_warning_score', 3)
    critical_score = ew.get('ew_critical_score', 5)
    canopy_low_thr = ew.get('ew_canopy_low', 0.30)

    g = sim_df.groupby('year')
    ce   = g['Ce'].mean()
    inv  = g['invasive_cover'].mean()
    conf = g['conflict_total'].mean()
    can  = g['canopy_cover'].mean()
    ce_v, inv_v, conf_v = ce.values, inv.values, conf.values
    can_v = can.values
    n = len(ce_v)
    w = int(min(window, n))

    ce_trend  = _slope(ce_v[-w:])
    ac_recent = _lag1_autocorr(ce_v[-w:])

    # variance ratio: late-window variance vs early-window variance
    if n >= 2 * window:
        var_start = float(np.var(ce_v[:window]))
        var_end   = float(np.var(ce_v[-window:]))
    else:
        half = max(3, n // 2)
        var_start = float(np.var(ce_v[:half]))
        var_end   = float(np.var(ce_v[-half:]))
    if var_start > 1e-6:
        var_ratio = var_end / var_start
    elif var_end > 1e-5:
        var_ratio = var_ratio_thr + 1.0
    else:
        var_ratio = 1.0

    over = inv_v > inv_thr
    years_over = int(over.sum())
    max_consec = consec = 0
    for o in over:
        consec = consec + 1 if o else 0
        max_consec = max(max_consec, consec)

    conf_trend = _slope(conf_v[-w:])
    final_conf = float(conf_v[-w:].mean())
    final_ce   = float(ce_v[-w:].mean())
    final_inv  = float(inv_v[-w:].mean())
    final_canopy = float(can_v[-w:].mean())

    flags = {
        'ce_declining':       bool(ce_trend < slope_thr),
        'variance_rising':    bool(var_ratio > var_ratio_thr),
        'autocorr_high':      bool(ac_recent > ac_thr),
        'invasive_persistent': bool(max_consec >= consec_req),
        'conflict_rising':    bool(conf_trend > 0 and final_conf > conflict_warn),
        'ce_below_fail':      bool(final_ce < ce_fail),
        'canopy_collapse':    bool(final_canopy < canopy_low_thr),
    }
    score = int(sum(flags.values()))

    failure = flags['invasive_persistent'] and (
        flags['ce_below_fail'] or final_inv > inv_thr + 0.10)

    if failure or score >= critical_score:
        level = 'critical'
    elif score >= warning_score:
        level = 'warning'
    elif score >= watch_score:
        level = 'watch'
    else:
        level = 'stable'

    indicators = {
        'final_Ce':                      round(final_ce, 4),
        'final_invasive':                round(final_inv, 4),
        'final_canopy':                  round(final_canopy, 4),
        'final_conflict_rate':           round(final_conf, 4),
        'ce_trend_slope':                round(ce_trend, 5),
        'ce_variance_ratio':             round(float(var_ratio), 4),
        'ce_lag1_autocorr':              round(ac_recent, 4),
        'invasive_years_over_threshold': years_over,
        'max_consecutive_invasive_over': max_consec,
        'conflict_trend_slope':          round(conf_trend, 5),
    }

    return {'window': window, 'warning_level': level, 'score': score,
            'flags': flags, 'indicators': indicators}


#----------------------------------------------------------------------------
# Section 7: Summary statistics and site driver
#----------------------------------------------------------------------------
def compute_statistics(sim_df):
    """
    Compute summary statistics from a simulation results DataFrame.

    Returns dict: annual_means, end_summary, conflict_summary,
    strategy_profile, pct_runs_threshold, pearson_Ce_sat.
    """
    annual_means = sim_df.groupby('year')[
        ['Ce', 'Cr', 'Ct', 'Cep',
         'invasive_cover', 'canopy_cover',
         'native_cover', 'satisfaction']].mean()

    end = sim_df[sim_df['year'] == sim_df['year'].max()]
    end_summary = {
        'final_Ce':       end['Ce'].mean(),
        'final_Cr':       end['Cr'].mean(),
        'final_Ct':       end['Ct'].mean(),
        'final_Cep':      end['Cep'].mean(),
        'final_invasive': end['invasive_cover'].mean(),
        'final_canopy':   end['canopy_cover'].mean(),
        'final_native':   end['native_cover'].mean(),
    }

    conflict_summary = {
        'total_conflicts':     int(sim_df['conflict_total'].sum()),
        'conflict_percentage': 100 * sim_df['conflict_total'].mean(),
    }

    strategy_counts  = sim_df['action'].value_counts().to_dict()
    strategy_profile = {a: strategy_counts.get(a, 0) for a in action_order}

    runs_over = sim_df.groupby('run_id')['invasive_cover'].max() > 0.4
    pct_runs_threshold = 100 * runs_over.mean()

    pearson = (sim_df[['Ce', 'satisfaction']].corr().iloc[0, 1]
               if len(sim_df) > 1 else np.nan)

    return {
        'annual_means':       annual_means,
        'end_summary':        end_summary,
        'conflict_summary':   conflict_summary,
        'strategy_profile':   strategy_profile,
        'pct_runs_threshold': pct_runs_threshold,
        'pearson_Ce_sat':     pearson,
    }


def run_site(site_cfg, params, output_dir, landscape=None):
    """
    Run the full simulation workflow for one site. Writes
    Simulation_results.csv, Annual_summary.csv, and Early_warning.csv into
    output_dir and prints a compact summary. Returns the full simulation
    DataFrame. Figures are produced separately by ADse-UFPM_Figures.py.

    The management mode is read from the site configuration:
        management_mode  = single | portfolio   (default single)
        portfolio_method = random | scipy | fixed (default random)
    """
    os.makedirs(output_dir, exist_ok=True)

    park_type      = site_cfg.get('park_type')
    service_target = site_cfg.get('service_target')
    climate_zone   = site_cfg.get('climate_zone')

    n_years    = site_cfg.get('n_years', 30)
    n_reps     = site_cfg.get('n_reps', 10)
    n_managers = site_cfg.get('n_managers', 3)
    seed       = site_cfg.get('seed', 42)
    budget     = site_cfg.get('budget', 10000.0)

    management_mode  = site_cfg.get('management_mode', 'single')
    portfolio_method = site_cfg.get('portfolio_method', 'random')

    social_params = {k: site_cfg[k] for k in
                     ('gdp_per_capita', 'civic_participation',
                      'park_visitor_density', 'park_area')
                     if k in site_cfg} or None

    print(f"Running {n_reps} replicates for "
          f"{park_type} / {service_target} / {climate_zone} "
          f"[mode={management_mode}"
          f"{'/' + portfolio_method if management_mode == 'portfolio' else ''}]")

    if landscape:
        print(f"  Landscape    : edge fraction {landscape.get('edge_fraction', 0):.2f}, "
              f"canopy edge {landscape.get('canopy_edge_density', 0):.0f} m/ha, "
              f"{landscape.get('edge_per_staff', 0):.0f} m per staff")

    sim_df = run_replicates(park_type, service_target, climate_zone,
                            n_reps, n_years, budget, n_managers, seed, params,
                            social_params=social_params,
                            management_mode=management_mode,
                            portfolio_method=portfolio_method,
                            landscape=landscape)

    sim_df.to_csv(os.path.join(output_dir, 'Simulation_results.csv'),
                  index=False, encoding='utf-8-sig')

    annual_summary = sim_df.groupby('year')[
        ['native_cover', 'invasive_cover', 'canopy_cover',
         'Ce', 'Cr', 'Ct', 'Cep', 'MPS', 'satisfaction']
    ].agg(['mean', 'std'])
    annual_summary.to_csv(os.path.join(output_dir, 'Annual_summary.csv'),
                          encoding='utf-8-sig')

    stats = compute_statistics(sim_df)

    # --- Early-warning and resilience diagnosis ---------------------------
    ew_window = int(params.get('early_warning', {}).get('ew_window', 10))
    ew = compute_early_warning(sim_df, window=ew_window, params=params)
    ew_row = {'site_id': site_cfg.get('site_id', ''),
              'warning_level': ew['warning_level'],
              'score': ew['score']}
    ew_row.update(ew['indicators'])
    ew_row.update({f'flag_{k}': int(v) for k, v in ew['flags'].items()})
    pd.DataFrame([ew_row]).to_csv(
        os.path.join(output_dir, 'Early_warning.csv'),
        index=False, encoding='utf-8-sig')

    print("\nFinal state summary:")
    for k, v in stats['end_summary'].items():
        print(f"  {k}: {v:.4f}")

    print("\nConflict summary:")
    for k, v in stats['conflict_summary'].items():
        print(f"  {k}: {v}")

    print("\nManagement strategy adoption:")
    for a in action_order:
        print(f"  {action_labels.get(a, a)}: {stats['strategy_profile'].get(a, 0)}")

    # Mean annual disturbance-event frequencies (share of years with an event)
    if 'storm' in sim_df.columns:
        print("\nDisturbance-event frequency (mean share of years):")
        for col, lab in (('heatwave', 'Heatwave'), ('drought_event', 'Drought'),
                         ('heavy_rain', 'Heavy rain/flood'), ('storm', 'Storm'),
                         ('pest_outbreak', 'Pest/disease')):
            print(f"  {lab}: {(sim_df[col] > 0).mean():.3f}")

    print(f"\nRuns with invasive > 40%: {stats['pct_runs_threshold']:.1f}%")
    print(f"Pearson(Ce, satisfaction): {stats['pearson_Ce_sat']:.3f}")
    print(f"\nEarly-warning level: {ew['warning_level'].upper()} "
          f"(score {ew['score']}/6); "
          f"final Ce={ew['indicators']['final_Ce']}, "
          f"final invasive={ew['indicators']['final_invasive']}, "
          f"Ce var ratio={ew['indicators']['ce_variance_ratio']}, "
          f"lag-1 AC={ew['indicators']['ce_lag1_autocorr']}")
    print(f"\nResults saved to: {output_dir}")

    return sim_df

#----------------------------------------------------------------------------
# Section 8: Landscape descriptor builder
#----------------------------------------------------------------------------
def build_landscape(boundary_path, utm_epsg, staff, site_dir,
                    date_window="2025-06-01/2025-09-30", edge_depth=100,
                    canopy_ndvi=0.7, min_patch_ha=0.1, matrix_ring=1000):
    # Turn one park boundary into the landscape descriptor block the model
    # reads. Writes Landscape_descriptors.csv into site_dir and returns the
    # descriptors as a dict. Needs geopandas, rasterio, scipy, pystac-client
    # and planetary-computer; run it once per park, not every simulation.
    if not landscape_tools:
        raise ImportError("build_landscape needs geopandas, rasterio, scipy, "
                          "pystac-client and planetary-computer")

    # --- Boundary, area and shape ---
    parks = gpd.read_file(boundary_path)
    parks = parks.set_crs(4326) if parks.crs is None else parks
    park_ll = unary_union(parks.to_crs(4326).geometry.values).buffer(0)
    park_m = gpd.GeoSeries([park_ll], crs=4326).to_crs(utm_epsg).iloc[0]

    area_ha = park_m.area / 1e4
    perimeter_m = park_m.length
    edge_density = perimeter_m / area_ha
    shape_index = perimeter_m / (2 * np.sqrt(np.pi * park_m.area))
    edge_fraction = 1 - park_m.buffer(-edge_depth).area / park_m.area

    # --- Canopy from a cloud-free growing-season Sentinel-2 scene ---
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    scenes = catalog.search(collections=["sentinel-2-l2a"],
                            intersects=park_ll.__geo_interface__,
                            datetime=date_window,
                            query={"eo:cloud_cover": {"lt": 30}}).item_collection()
    scene = sorted(scenes, key=lambda s: s.properties["eo:cloud_cover"])[0]
    baseline = float(str(scene.properties.get("s2:processing_baseline", "05.00")))
    offset = 1000.0 if baseline >= 4.0 else 0.0
    outer_m = park_m.buffer(matrix_ring)

    def read_band(href):
        with rasterio.open(href) as band:
            box = transform_bounds(utm_epsg, band.crs, *outer_m.bounds)
            window = from_bounds(*box, band.transform).round_offsets().round_lengths()
            return (band.read(1, window=window).astype("float32"),
                    band.window_transform(window), band.crs)

    red, grid, grid_crs = read_band(scene.assets["B04"].href)
    nir, _, _ = read_band(scene.assets["B08"].href)
    ndvi = ((nir - offset) - (red - offset)) / np.maximum((nir - offset) + (red - offset), 1e-6)

    park_px = rasterize([(gpd.GeoSeries([park_m], crs=utm_epsg).to_crs(grid_crs).iloc[0], 1)],
                        out_shape=ndvi.shape, transform=grid, dtype="uint8").astype(bool)
    pixel_m = abs(grid.a)
    cell_ha = (pixel_m ** 2) / 1e4

    canopy = park_px & (ndvi > canopy_ndvi)
    labels, _ = ndimage.label(canopy, structure=np.ones((3, 3)))
    patch_ha = np.bincount(labels.ravel())[1:] * cell_ha
    canopy = np.isin(labels, np.where(patch_ha >= min_patch_ha)[0] + 1)
    labels, patch_count = ndimage.label(canopy, structure=np.ones((3, 3)))
    patch_ha = np.bincount(labels.ravel())[1:] * cell_ha
    canopy_ha = patch_ha.sum()
    canopy_observed = canopy_ha / (park_px.sum() * cell_ha)

    # --- Fragmentation ---
    # Total canopy edge counts every cell face between canopy and non-canopy,
    # so internal gaps count as well as the park outline.
    def face_length(binary):
        padded = np.pad(binary, 1)
        middle = padded[1:-1, 1:-1]
        faces = ((middle & ~padded[:-2, 1:-1]).sum() + (middle & ~padded[2:, 1:-1]).sum()
                 + (middle & ~padded[1:-1, :-2]).sum() + (middle & ~padded[1:-1, 2:]).sum())
        return faces * pixel_m

    canopy_edge_m = face_length(canopy)
    canopy_edge_density = canopy_edge_m / area_ha
    division_index = 1 - ((patch_ha / canopy_ha) ** 2).sum()
    largest_patch = patch_ha.max() / (park_px.sum() * cell_ha)
    core_canopy_30 = ((ndimage.distance_transform_edt(canopy) * pixel_m > 30).sum()
                      * cell_ha / canopy_ha)
    edge_per_staff = canopy_edge_m / max(staff, 1)

    # --- Surrounding matrix ---
    outer_px = rasterize([(gpd.GeoSeries([outer_m], crs=utm_epsg).to_crs(grid_crs).iloc[0], 1)],
                         out_shape=ndvi.shape, transform=grid, dtype="uint8").astype(bool)
    matrix_built = float((ndvi[outer_px & ~park_px] < 0.25).mean())

    # --- Access network from OpenStreetMap ---
    vehicle_tags = {"motorway", "trunk", "primary", "secondary", "tertiary",
                    "unclassified", "residential", "service", "track"}
    foot_tags = {"footway", "path", "steps", "pedestrian", "cycleway", "bridleway"}
    mirrors = ["https://overpass.kumi.systems/api/interpreter",
               "https://overpass-api.de/api/interpreter",
               "https://overpass.osm.jp/api/interpreter"]
    south, west, north, east = (park_ll.bounds[1], park_ll.bounds[0],
                                park_ll.bounds[3], park_ll.bounds[2])
    query = f'[out:json][timeout:180];way["highway"]({south},{west},{north},{east});out geom tags;'
    payload = urllib.parse.urlencode({"data": query}).encode()
    headers = {"User-Agent": "ADse-UFPM landscape preprocessing"}

    answer = None
    for attempt in range(6):
        try:
            request = urllib.request.Request(mirrors[attempt % len(mirrors)],
                                             data=payload, headers=headers)
            answer = json.load(urllib.request.urlopen(request, timeout=240))
            break
        except Exception:
            time.sleep(8 * (attempt + 1))

    vehicle_km = foot_km = 0.0
    if answer is None:
        print("  Overpass unavailable, access descriptors left at 0")
    else:
        ways = [{"highway": el["tags"].get("highway", ""),
                 "geometry": LineString([(q["lon"], q["lat"]) for q in el["geometry"]])}
                for el in answer["elements"] if len(el.get("geometry", [])) >= 2]
        lines = gpd.GeoDataFrame(ways, crs=4326).to_crs(utm_epsg)
        lines["geometry"] = lines.geometry.intersection(park_m)
        lines = lines[~lines.geometry.is_empty]
        lines["km"] = lines.geometry.length / 1000
        vehicle_km = float(lines[lines.highway.isin(vehicle_tags)].km.sum())
        foot_km = float(lines[lines.highway.isin(foot_tags)].km.sum())

    access_density = 1000 * (vehicle_km + foot_km) / area_ha
    vehicle_density = 1000 * vehicle_km / area_ha

    # --- Write the block ---
    rows = [
        ("park_area", area_ha, "ha", "Measured from the boundary polygon"),
        ("edge_density", edge_density, "m/ha", "Park outline only"),
        ("shape_index", shape_index, "", "1.0 for a circle"),
        ("edge_fraction", edge_fraction, "",
         f"Share of the park within {edge_depth} m of the boundary"),
        ("canopy_observed", canopy_observed, "",
         f"Share of the park with ndvi above {canopy_ndvi}"),
        ("canopy_edge_density", canopy_edge_density, "m/ha",
         "All canopy edge including internal gaps"),
        ("division_index", division_index, "", "0 for one unbroken canopy patch"),
        ("largest_patch", largest_patch, "", "Largest canopy patch as a share of the park"),
        ("core_canopy_30", core_canopy_30, "", "Canopy deeper than 30 m inside a patch"),
        ("matrix_built", matrix_built, "",
         f"Built share of the {matrix_ring} m ring around the park"),
        ("access_density", access_density, "m/ha", "Roads and paths inside the park"),
        ("vehicle_density", vehicle_density, "m/ha", "Vehicle-usable roads only"),
        ("edge_per_staff", edge_per_staff, "m", "Canopy edge divided by permanent staff"),
    ]
    os.makedirs(str(site_dir), exist_ok=True)
    out_file = os.path.join(str(site_dir), "Landscape_descriptors.csv")
    with open(out_file, "w", encoding="utf-8") as handle:
        handle.write("#--- Landscape descriptors, from build_landscape() ---\n")
        handle.write(f"# Scene {scene.id}, {scene.properties['datetime'][:10]}.\n")
        for name, value, units, detail in rows:
            handle.write(f"{name},{value:.3f},{units},{detail}\n")

    print(f"  Landscape    : {patch_count} canopy patches, "
          f"{canopy_edge_density:,.0f} m/ha canopy edge, "
          f"{edge_per_staff:,.0f} m per staff")
    print(f"  Wrote {out_file}")
    return {name: float(value) for name, value, _, _ in rows}
