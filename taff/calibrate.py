"""Calibrate L, the initial quality aggregates Qz0 and the variety masses M so
that the equilibrium hits a set of targets.

The targets are added to the equilibrium system (Model in calibration mode),
so calibration is one Newton solve rather than an outer loop of solves.
"""

from dataclasses import dataclass

import numpy as np

from .model import Model, Params


@dataclass
class Targets:
    PE0: float = 0.060  # wholesale electricity price at t=0, $bn/TWh (= $60/MWh)
    Pc0: float = 3.0  # clean capital price index at t=0 (= solar capex, $bn/GW)
    Pd0: float = 1.0  # dirty capital price index at t=0 (= gas capex, $bn/GW)
    # Average annual fall in the capital price indices over the first `window`
    # years. (The period-0 innovation rate is a poor target: when research in a
    # sector stops, the discrete-time free-entry condition makes the last few
    # years of research zig-zag, so d_0 jumps with the stopping date.)
    decline_c: float = 0.07
    decline_d: float = 0.01
    window: int = 10


def calibrate(par: Params, tg: Targets = Targets(), verbose=False):
    """Returns (calibrated params, baseline equilibrium solution vector)."""
    # Solve the uncalibrated model first to get a good starting path.
    x = Model(par).solve(verbose=verbose)
    mc = Model(par, targets=tg)
    xc = mc.solve(mc.to_calibration(x), verbose=verbose)
    par = mc.calibrated_params(xc)

    # Re-solve in normal mode and confirm the targets are met.
    # In normal mode x[0] is log L^Y_0, which calibration pinned at mc.LY0;
    # the scientist unknowns are scaled by M, which calibration changed.
    m, T = Model(par), par.T
    x0 = np.concatenate([[np.log(mc.LY0)], xc[1:7 * T]])
    x0[T:2 * T] *= mc.S_scale[0] / m.S_scale[0]
    x0[2 * T:3 * T] *= mc.S_scale[1] / m.S_scale[1]
    x = m.solve(x0, verbose=verbose)
    return par, x


def check_targets(par: Params, x, tg: Targets = Targets()):
    o = Model(par).evaluate(x[None], full=True)
    n = tg.window
    decline = 1 - np.exp(np.log(o["Pz"][n] / o["Pz"][0]) / n)
    return {
        "PE0": (o["PE"][0], tg.PE0),
        "Pc0": (o["Pz"][0, 0], tg.Pc0),
        "Pd0": (o["Pz"][0, 1], tg.Pd0),
        "decline_c": (decline[0], tg.decline_c),
        "decline_d": (decline[1], tg.decline_d),
    }
