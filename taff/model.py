"""Closed-economy model of power-plant investment with directed innovation.

Implements docs/model.tex as a perfect-foresight transition path.

Unknowns, for t = 0..T-1:
    log L^Y_t, S^c_t, S^d_t, log K_{k,t+1} (k = solar, wind, gas, coal)
Everything else follows from these paths:
    statics (Y, w, P^E, C, r), quality aggregates (forward), project values
    Q^k and patent values v^z (backward), nested-logit shares, capex R^z.
Residuals, for t = 0..T-1:
    labour-market clearing, free entry of scientists in c and d (written as a
    complementarity condition: S^z >= 0, w - value >= 0, one holds with
    equality), and the capacity law of motion.

Terminal condition: after period T the economy is treated as stationary at its
period-T allocation (project and patent values are perpetuities at rate rho).
With T large this has no visible effect on the plotted horizon.

Notation: the paper uses Q for both the project value Q^k_t and the quality
aggregate Q^z_t. Here the project value is `PV` and the quality aggregate is
`Qz`.
"""

from dataclasses import dataclass, field

import numpy as np

TYPES = ("solar", "wind", "gas", "coal")
CLEAN = np.array([True, True, False, False])
FOSSIL = ~CLEAN
SECTOR = np.array([0, 0, 1, 1])  # capital sector of each plant type: 0 = c, 1 = d
NK = len(TYPES)

HOURS = 8.76  # TWh per GW-year at a capacity factor of 1


def _vec(*values):
    return field(default_factory=lambda: np.array(values, dtype=float))


@dataclass
class Params:
    """Model parameters. Units: final good in $bn, capacity in GW,
    electricity in TWh, fuel in TWh thermal, labour in model units."""

    T: int = 200  # solution horizon (years)

    # Households
    rho: float = 0.05  # discount rate
    theta: float = 1.0  # CRRA coefficient (1 = log utility)
    L: float = 18_000.0  # labour supply (calibrated so that P^E_0 hits its target)

    # Final good
    beta: float = 0.012  # electricity cost share of output

    # Plant types, ordered as TYPES
    A: np.ndarray = _vec(0.18 * HOURS, 0.33 * HOURS, 0.45 * HOURS, 0.62 * HOURS)
    delta: np.ndarray = _vec(1 / 25, 1 / 25, 1 / 30, 1 / 40)
    iota: np.ndarray = _vec(1.0, 0.75, 1.0, 2.5)  # capital bundle per GW
    h: np.ndarray = _vec(1.0, 1.0, 0.50, 0.37)  # thermal efficiency (fossil only)
    p_fuel: np.ndarray = _vec(0.0, 0.0, 0.015, 0.008)  # $bn per TWh_th ($15, $8 /MWh_th)
    em: np.ndarray = _vec(0.0, 0.0, 0.20, 0.34)  # tCO2 per MWh_th

    # Developers
    D: float = 34.0  # GW of project opportunities per year
    mu: float = 0.4  # dispersion of project shocks ($bn per GW)
    lam_C: float = 0.8  # nesting parameter, clean nest
    lam_F: float = 0.8  # nesting parameter, fossil nest

    # Capital varieties
    sigma: float = 5.0  # 25% markup on capital varieties
    psi: float = 1.0
    M: np.ndarray = _vec(6.0, 75.0)  # mass of varieties (c, d); calibrated

    # Innovation. Local determinacy needs (1 + r) * G > 1 with
    # G = (1 + gamma)^(sigma - 1) - 1, and the market-size feedback erodes the
    # margin, so keep well above 1 (here 1.95); see README ("Determinacy").
    eta: float = 0.5
    gamma: float = 0.3

    # Initial states
    K0: np.ndarray = _vec(20.0, 60.0, 450.0, 330.0)  # GW
    Qz0: np.ndarray = _vec(0.3, 2.7)  # quality aggregates (c, d); calibrated

    @property
    def e(self):
        """Emissions intensity e_k in the paper's units: tau^E e_k is the
        ad-valorem fuel tax when tau^E is measured in $/tCO2."""
        return np.divide(self.em * 1e-3, self.p_fuel,
                         out=np.zeros(NK), where=self.p_fuel > 0)


@dataclass
class Policy:
    """Policy paths indexed by t = 0, 1, ... Shorter paths are extended with
    their last value; None means no policy."""

    tau: np.ndarray | None = None  # carbon tax, $/tCO2
    fit_premium: np.ndarray | None = None  # P^F - P^E for clean plants, $bn/TWh
    xi: np.ndarray | None = None  # clean R&D subsidy per scientist, $bn


def _path(x, T, default=0.0):
    if x is None:
        return np.full(T + 1, default)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if len(x) >= T + 1:
        return x[: T + 1].copy()
    return np.concatenate([x, np.full(T + 1 - len(x), x[-1])])


def _logsumexp(a, axis=-1):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def nested_logit(v, lam_C, lam_F):
    """Choice shares for nests {0}, C = {solar, wind}, F = {gas, coal}.
    v: (..., 4) utilities W/mu. Returns (s_outside, s) with s of shape (..., 4)."""
    vc = v[..., CLEAN] / lam_C
    vf = v[..., FOSSIL] / lam_F
    lse_c = _logsumexp(vc)
    lse_f = _logsumexp(vf)
    incl = np.stack([np.zeros_like(lse_c), lam_C * lse_c, lam_F * lse_f], -1)
    s_nest = np.exp(incl - _logsumexp(incl)[..., None])
    s = np.empty(v.shape)
    s[..., CLEAN] = np.exp(vc - lse_c[..., None]) * s_nest[..., 1:2]
    s[..., FOSSIL] = np.exp(vf - lse_f[..., None]) * s_nest[..., 2:3]
    return s_nest[..., 0], s


class Model:
    """Equilibrium path for given parameters and policy.

    With `targets` (a calibrate.Targets), the model runs in calibration mode:
    L, Qz0 and M become endogenous and the period-0 targets are added to the
    system, so a single Newton solve delivers the calibrated equilibrium.
    In that mode x[0] holds log L (L^Y_0 is pinned by the P^E_0 target) and
    two extra unknowns at the end hold log M^c, log M^d.
    """

    def __init__(self, par: Params, pol: Policy | None = None, targets=None):
        self.par = par
        self.pol = pol or Policy()
        self.targets = targets
        T = self.T = par.T

        self.Dt = np.full(T + 1, par.D)
        self.A = np.tile(par.A, (T + 1, 1))
        self.p = np.tile(par.p_fuel, (T + 1, 1))

        self.tau = _path(self.pol.tau, T)
        prem = _path(self.pol.fit_premium, T)
        self.premium = np.where(CLEAN, prem[:, None], 0.0)
        self.xi = _path(self.pol.xi, T)

        # Fossil marginal cost c^k_t ($bn per TWh_e); zero for clean plants.
        self.c = np.where(FOSSIL, (1 + self.tau[:, None] * par.e) * self.p / par.h, 0.0)

        self.G = (1 + par.gamma) ** (par.sigma - 1) - 1
        self.S_scale = 0.1 * par.M / par.eta  # S / S_scale = 10 x displacement rate
        self.n = 7 * T + (2 if targets is not None else 0)
        self.fb_eps = 0.0
        if targets is not None:
            E0 = par.A @ par.K0
            self.LY0 = E0 * (targets.PE0 / par.beta) ** (1 / (1 - par.beta))

    # ------------------------------------------------------------------
    def initial_guess(self):
        T = self.T
        x = np.empty(self.n)
        x[:T] = np.log(0.995 * self.par.L)
        x[T:3 * T] = 1.0
        x[3 * T:7 * T] = np.tile(np.log(self.par.K0), T)
        if self.targets is not None:
            x[0] = np.log(self.par.L)
            x[7 * T:] = np.log(self.par.M)
        return x

    def to_calibration(self, x):
        """Map a normal-mode solution to a calibration-mode starting point."""
        return np.concatenate([[np.log(self.par.L)], x[1:7 * self.T], np.log(self.par.M)])

    def calibrated_params(self, x):
        """Parameters implied by a calibration-mode solution."""
        from dataclasses import replace
        o = self.evaluate(x[None], full=True)
        return replace(self.par, L=float(o["L"]), M=o["M"], Qz0=o["Qz0"])

    def _unpack(self, X):
        T = self.T
        B = X.shape[0]
        lLY = X[:, :T]
        S = np.stack([X[:, T:2 * T], X[:, 2 * T:3 * T]], -1) * self.S_scale
        lK = X[:, 3 * T:7 * T].reshape(B, T, NK)
        if self.targets is None:
            L = np.full((B, 1), self.par.L)
            M = np.broadcast_to(self.par.M, (B, 1, 2))
        else:
            L = np.exp(X[:, :1])
            M = np.exp(X[:, None, 7 * T:])
            lLY = lLY.copy()
            lLY[:, 0] = np.log(self.LY0)
        return lLY, S, lK, L, M

    def evaluate(self, X, full=False):
        """Residuals for a batch of candidate paths X (B, n) -> (B, n).
        With full=True (single path), return a dict of all equilibrium objects."""
        par, T = self.par, self.T
        X = np.atleast_2d(X)
        B = X.shape[0]
        lLY, S, lK, L, M = self._unpack(X)
        b, sig = par.beta, par.sigma

        # States and statics for t = 0..T (L^Y_T := L^Y_{T-1}).
        LY = np.exp(np.concatenate([lLY, lLY[:, -1:]], 1))
        K = np.exp(np.concatenate(
            [np.broadcast_to(np.log(par.K0), (B, 1, NK)), lK], 1))
        E = (self.A * K).sum(-1)
        Y = LY ** (1 - b) * E ** b
        w = (1 - b) * Y / LY
        PE = b * Y / E
        price = PE[..., None] + self.premium  # P^F for clean, P^E for fossil
        flow = self.A * (price - self.c)  # operating profit per GW
        fuel_use = np.where(FOSSIL, self.A * K / par.h, 0.0)  # TWh_th
        fuel_cost = (self.p * fuel_use).sum(-1)
        C = Y - fuel_cost
        R1 = (1 + par.rho) * (C[:, 1:] / C[:, :-1]) ** par.theta  # 1 + r_{t+1}

        # Project values Q^k_t (backward).
        disc_K = 1 - par.delta
        PV = np.empty((B, T + 1, NK))
        PV[:, T] = flow[:, T] / (1 - disc_K / (1 + par.rho))
        for t in range(T - 1, -1, -1):
            PV[:, t] = flow[:, t] + disc_K / R1[:, t, None] * PV[:, t + 1]

        # Quality aggregates Q^z_t (forward) and capital price indices.
        d = par.eta * S / M  # displacement probability (B, T, 2)
        if self.targets is None:
            Qz0 = np.broadcast_to(par.Qz0, (B, 1, 2))
        else:  # pin P^z_0 at its targets
            P0 = np.array([self.targets.Pc0, self.targets.Pd0])
            Qz0 = (P0 * (sig - 1) / (sig * par.psi * w[:, :1, None])) ** (1 - sig)
        Qz = Qz0 * np.concatenate(
            [np.ones((B, 1, 2)), np.cumprod(1 + d * self.G, 1)], 1)
        Pz = sig / (sig - 1) * par.psi * w[..., None] * Qz ** (1 / (1 - sig))

        # Developers.
        F = par.iota * Pz[:, :T][..., SECTOR]
        W = -F + PV[:, 1:] / R1[..., None]
        s0, s = nested_logit(W / par.mu, par.lam_C, par.lam_F)
        build = s * self.Dt[:T, None]
        bundle = np.stack([(par.iota * build)[..., CLEAN].sum(-1),
                           (par.iota * build)[..., FOSSIL].sum(-1)], -1)
        Rz = Pz[:, :T] * bundle  # capex revenue of each variety sector
        LM = (sig - 1) / sig * Rz.sum(-1) / w[:, :T]

        # Patent values v^z_t (backward).
        v = np.empty((B, T + 1, 2))
        RzT = Rz[:, -1] * Pz[:, T] / Pz[:, T - 1]
        v[:, T] = RzT / (sig * Qz[:, T]) / (1 - (1 - d[:, -1]) / (1 + par.rho))
        for t in range(T - 1, -1, -1):
            v[:, t] = Rz[:, t] / (sig * Qz[:, t]) + (1 - d[:, t]) / R1[:, t, None] * v[:, t + 1]
        value = (par.eta * (1 + par.gamma) ** (sig - 1) * Qz[:, :T] / M
                 * v[:, 1:] / R1[..., None])
        value[..., 0] += self.xi[:T]

        # Residuals.
        r_L = (L - LY[:, :T] - LM - S.sum(-1)) / L
        a = S / self.S_scale
        gap = (w[:, :T, None] - value) / w[:, :T, None]
        # Smoothed Fischer-Burmeister: zero iff a, gap >= 0 and a * gap = fb_eps.
        r_FE = a + gap - np.sqrt(a * a + gap * gap + 2 * self.fb_eps)
        r_K = lK - np.log(disc_K * K[:, :T] + build)
        res = [r_L, r_FE[..., 0], r_FE[..., 1], r_K.reshape(B, -1)]
        if self.targets is not None:
            # Average annual fall in the capital price indices over the window.
            n = self.targets.window
            achieved = -np.log(Pz[:, n] / Pz[:, 0]) / n
            target = -np.log(1 - np.array([self.targets.decline_c, self.targets.decline_d]))
            res.append(achieved / target - 1)
        res = np.concatenate(res, 1)
        if not full:
            return res

        # Remaining equilibrium objects (single path).
        i = 0
        tax_rev = (self.tau[:, None] * par.e * self.p * fuel_use[i]).sum(-1)
        fit_cost = (self.premium * self.A * K[i]).sum(-1)
        rd_sub = self.xi[:T] * S[i, :, 0]
        emissions = (par.em * fuel_use[i]).sum(-1)  # MtCO2
        out = dict(
            Y=Y[i], C=C[i], w=w[i], PE=PE[i], E=E[i], LY=LY[i], K=K[i],
            r=R1[i] - 1, PV=PV[i], Qz=Qz[i], Pz=Pz[i], F=F[i], W=W[i],
            s0=s0[i], s=s[i], build=build[i], Rz=Rz[i], LM=LM[i], S=S[i], d=d[i],
            v=v[i], value=value[i], c=self.c, fuel_use=fuel_use[i],
            L=L[i, 0], M=M[i, 0].copy(), Qz0=Qz0[i, 0].copy(),
            fuel_cost=fuel_cost[i], gen=self.A * K[i], tax_rev=tax_rev,
            fit_cost=fit_cost, rd_sub=rd_sub, emissions=emissions,
            residual=res[i],
        )
        out["T_lump"] = fit_cost[:T] + rd_sub - tax_rev[:T]
        out["checks"] = self._checks(out)
        return out

    def _checks(self, o):
        """Accounting identities and model assumptions along the path."""
        par, T = self.par, self.T
        sig = par.sigma
        # Household budget: C = wL + Pi - T must equal Y - fuel (Walras' law).
        plant_profit = (o["gen"][:T] * (o["PE"][:T, None] + self.premium[:T] - self.c[:T])).sum(-1)
        Pi = (o["Rz"].sum(-1) / sig + plant_profit - (o["F"] * o["build"]).sum(-1)
              - ((o["w"][:T] - self.xi[:T]) * o["S"][:, 0] + o["w"][:T] * o["S"][:, 1]))
        C_budget = o["w"][:T] * o["L"] + Pi - o["T_lump"]
        fossil_margin = (o["PE"][:T, None] - self.c[:T])[:, FOSSIL]
        return dict(
            max_abs_residual=float(np.abs(o["residual"]).max()),
            walras_gap=float(np.abs(C_budget / o["C"][:T] - 1).max()),
            interior_dispatch=bool((fossil_margin >= 0).all()),
            min_fossil_margin=float(fossil_margin.min()),
            first_violation=(int(np.argmax((fossil_margin < 0).any(1)))
                             if (fossil_margin < 0).any() else None),
            max_displacement=float(o["d"].max()),
            determinacy_index=float((1 + par.rho) * self.G),  # must exceed 1
        )

    # ------------------------------------------------------------------
    def residual(self, x):
        return self.evaluate(x[None])[0]

    def jacobian(self, x, f0=None, chunk=400):
        """Forward-difference Jacobian, evaluated in batches."""
        f0 = self.residual(x) if f0 is None else f0
        n = len(x)
        J = np.empty((len(f0), n))
        h = 1e-7 * np.maximum(1.0, np.abs(x))
        for lo in range(0, n, chunk):
            idx = np.arange(lo, min(n, lo + chunk))
            Xb = np.repeat(x[None], len(idx), 0)
            Xb[np.arange(len(idx)), idx] += h[idx]
            J[:, idx] = ((self.evaluate(Xb) - f0) / h[idx, None]).T
        return J

    def solve(self, x0=None, tol=1e-10, verbose=False):
        """Solve for the equilibrium path. Newton on the exact system first;
        if that stalls (typically at a kink where research in a sector switches
        on or off), solve a sequence of smoothed problems and tighten."""
        x0 = self.initial_guess() if x0 is None else x0
        try:
            return self._newton(x0, tol, verbose=verbose)
        except RuntimeError:
            pass
        x = x0
        for eps in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
            self.fb_eps = eps
            try:
                x = self._newton(x, tol, verbose=verbose)
            finally:
                self.fb_eps = 0.0
        # At fb_eps = 1e-12 the complementarity slack S * gap is negligible.
        return x

    def _newton(self, x0, tol, maxit=60, verbose=False):
        """Damped Newton on the stacked path residuals."""
        x = x0.copy()
        f = self.residual(x)
        for it in range(maxit):
            norm = np.abs(f).max()
            if norm < tol:
                break
            J = self.jacobian(x, f)
            try:
                dx = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                raise RuntimeError(f"singular Jacobian at iteration {it}")
            # Backtracking on the merit function 0.5 |F|^2, for which the
            # Newton direction is a descent direction.
            merit = f @ f
            step = 1.0
            while step > 1e-8:
                xn = x + step * dx
                with np.errstate(all="ignore"):
                    fn = self.residual(xn)
                mn = fn @ fn
                if np.isfinite(mn) and mn <= (1 - 1e-4 * step) * merit:
                    break
                step *= 0.5
            else:
                raise RuntimeError(f"line search failed at iteration {it}, |F|={norm:.2e}")
            x, f = xn, fn
            if verbose:
                print(f"  newton {it:2d}  |F|={np.abs(f).max():.2e}  step={step:.3g}")
        norm = np.abs(f).max()
        if norm >= tol:
            raise RuntimeError(f"no convergence: |F|={norm:.2e}")
        return x


def scale_policy(pol: Policy, lam: float) -> Policy:
    """Policy with every instrument multiplied by lam (for continuation)."""
    f = lambda v: None if v is None else lam * np.asarray(v, dtype=float)
    return Policy(tau=f(pol.tau), fit_premium=f(pol.fit_premium), xi=f(pol.xi))


def solve_policy(par: Params, pol: Policy, x_start, steps=(1.0,), verbose=False):
    """Solve under `pol`, starting from a no-policy solution x_start. If a
    direct solve fails, phase the policy in (0 -> 1) and warm-start each step."""
    x = x_start
    for lam in steps:
        m = Model(par, scale_policy(pol, lam))
        x = m.solve(x, verbose=verbose)
    return m, x


def solve_with_continuation(par: Params, pol: Policy, x_start, verbose=False):
    for n in (1, 4, 10, 25):
        try:
            return solve_policy(par, pol, x_start, steps=np.linspace(1 / n, 1, n), verbose=verbose)
        except RuntimeError as err:
            if verbose:
                print(f"  continuation with {n} steps failed: {err}")
    raise RuntimeError("policy solve failed even with continuation")

