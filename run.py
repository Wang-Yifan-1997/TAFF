"""Calibrate the model, solve the policy scenarios and write the results.

    python run.py            # calibrate, solve, write output/
    python run.py --plot     # also draw output/figures/*.png

Outputs:
    output/results.csv     one row per scenario x year
    output/summary.json    calibration, model checks and headline numbers
"""

import argparse
import json
import pathlib
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from taff.calibrate import Targets, calibrate
from taff.model import CLEAN, TYPES, Model, Params, Policy, solve_with_continuation

OUT = pathlib.Path(__file__).parent / "output"


def make_scenarios(par: Params, w0: float):
    """Policies announced at t = 0 (perfect foresight)."""
    t = np.arange(par.T + 1)
    return {
        "baseline": Policy(),
        # $30/tCO2, phased in linearly over 5 years, permanent.
        "carbon_tax": Policy(tau=30.0 * np.minimum(t / 5, 1.0)),
        # Feed-in premium of $20/MWh on all clean output for 15 years.
        "feed_in_premium": Policy(fit_premium=np.where(t < 15, 0.020, 0.0)),
        # Clean R&D subsidy of 50% of the initial wage per scientist for 15 years.
        "rd_subsidy": Policy(xi=np.where(t < 15, 0.5 * w0, 0.0)),
    }


def to_frame(name, m: Model, o: dict) -> pd.DataFrame:
    T = m.T
    df = pd.DataFrame({"scenario": name, "year": np.arange(T)})
    df["electricity_price"] = o["PE"][:T] * 1e3  # $/MWh
    df["wage"] = o["w"][:T]
    df["output"] = o["Y"][:T]
    df["consumption"] = o["C"][:T]
    df["interest_rate"] = o["r"]
    df["electricity_TWh"] = o["E"][:T]
    for j, k in enumerate(TYPES):
        df[f"capacity_{k}"] = o["K"][:T, j]
        df[f"generation_{k}"] = o["gen"][:T, j]
        df[f"build_{k}"] = o["build"][:, j]
        df[f"build_share_{k}"] = o["s"][:, j]
        df[f"capex_{k}"] = o["F"][:, j] * 1e3  # $/kW
        df[f"project_value_{k}"] = o["W"][:, j] * 1e3  # $/kW
    df["build_share_none"] = o["s0"]
    df["clean_build_share"] = o["s"][:, CLEAN].sum(1)
    df["clean_share_of_new_capacity"] = df["clean_build_share"] / (1 - o["s0"])
    df["clean_generation_share"] = o["gen"][:T, CLEAN].sum(1) / o["E"][:T]
    df["capital_price_clean"] = o["Pz"][:T, 0]
    df["capital_price_dirty"] = o["Pz"][:T, 1]
    df["quality_clean"] = o["Qz"][:T, 0]
    df["quality_dirty"] = o["Qz"][:T, 1]
    df["capex_revenue_clean"] = o["Rz"][:, 0]
    df["capex_revenue_dirty"] = o["Rz"][:, 1]
    df["scientists_clean"] = np.maximum(o["S"][:, 0], 0)
    df["scientists_dirty"] = np.maximum(o["S"][:, 1], 0)
    df["innovation_rate_clean"] = np.maximum(o["d"][:, 0], 0)
    df["innovation_rate_dirty"] = np.maximum(o["d"][:, 1], 0)
    df["labour_variety_production"] = o["LM"]
    df["emissions_Mt"] = o["emissions"][:T]
    df["carbon_tax_revenue"] = o["tax_rev"][:T]
    df["fit_outlays"] = o["fit_cost"][:T]
    df["rd_subsidy_outlays"] = o["rd_sub"]
    df["lump_sum_tax"] = o["T_lump"]
    df["fossil_cost_gas"] = o["c"][:T, 2] * 1e3
    df["fossil_cost_coal"] = o["c"][:T, 3] * 1e3
    return df


def summarise(name, df, base, par):
    beta = 1 / (1 + par.rho)
    disc = beta ** df["year"].to_numpy()
    if par.theta == 1:
        u = np.log
    else:
        u = lambda c: c ** (1 - par.theta) / (1 - par.theta)
    # Consumption-equivalent change vs baseline (log utility closed form;
    # the model has no climate damages, so this is the economic cost only).
    U, U0 = (disc * u(df["consumption"])).sum(), (disc * u(base["consumption"])).sum()
    cev = np.exp((U - U0) / disc.sum()) - 1 if par.theta == 1 else np.nan
    clean_majority = df.loc[df["clean_share_of_new_capacity"] > 0.5, "year"]

    def at(col, y):
        return float(df.loc[df["year"] == y, col].iloc[0])

    return {
        "cumulative_emissions_50y_Gt": float(df.loc[df.year < 50, "emissions_Mt"].sum() / 1e3),
        "emissions_cut_vs_baseline_50y_pct": float(
            100 * (1 - df.loc[df.year < 50, "emissions_Mt"].sum()
                   / base.loc[base.year < 50, "emissions_Mt"].sum())),
        "clean_generation_share": {y: at("clean_generation_share", y) for y in (0, 10, 20, 30, 50)},
        "first_year_clean_majority_of_new_capacity":
            int(clean_majority.iloc[0]) if len(clean_majority) else None,
        "solar_capex_per_kW": {y: at("capex_solar", y) for y in (0, 10, 20, 30)},
        "consumption_equivalent_change_pct": float(100 * cev),
        "pv_net_fiscal_cost_bn": float(
            (disc * (df["fit_outlays"] + df["rd_subsidy_outlays"]
                     - df["carbon_tax_revenue"])).sum()),
    }


def main(plot=False):
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    targets = Targets()
    par, x = calibrate(Params(), targets)
    print(f"calibrated in {time.time() - t0:.0f}s: L={par.L:.1f} Qz0={par.Qz0} M={par.M}")

    base_model = Model(par)
    w0 = base_model.evaluate(x[None], full=True)["w"][0]
    frames, checks, x_base = {}, {}, x
    for name, pol in make_scenarios(par, w0).items():
        m, xs = solve_with_continuation(par, pol, x_base, verbose=True)
        o = m.evaluate(xs[None], full=True)
        frames[name] = to_frame(name, m, o)
        checks[name] = o["checks"]
        print(f"{name:16s} solved; checks: {o['checks']}")

    results = pd.concat(frames.values(), ignore_index=True)
    results.to_csv(OUT / "results.csv", index=False)

    base = frames["baseline"]
    summary = {
        "calibration_targets": asdict(targets),
        "parameters": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in asdict(par).items()},
        "plant_types": list(TYPES),
        "checks": checks,
        "scenarios": {n: summarise(n, df, base, par) for n, df in frames.items()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["scenarios"], indent=2))

    if plot:
        from figures import make_figures
        make_figures(results, OUT / "figures")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    main(**vars(ap.parse_args()))
