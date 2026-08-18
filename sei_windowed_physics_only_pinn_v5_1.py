#!/usr/bin/env python3
"""
V5.1: sequential/windowed, strictly physics-only PINN with transformed-state residuals.

Why this version exists
-----------------------
The global long-time PINN can minimize a weighted residual while missing the
transient/endemic branch.  V5 replaces one global network by a sequence of
local PINNs on causally ordered time windows.  Each local network enforces its
left-end state exactly; that state is either the prescribed initial condition
(first window) or the previous PINN's predicted endpoint (later windows).
No RK4 value is used in training.

Training information allowed
----------------------------
  * SEI differential equations and fixed model parameters;
  * prescribed initial condition;
  * positivity/invariant-region structure;
  * model-derived window schedule from the DFE infected-subsystem eigenvalue.

Training information NOT used
-----------------------------
  * RK4 trajectories or anchors;
  * RK4 peaks/settling times;
  * RK4-derived scaling, window boundaries, or terminal values;
  * analytic endemic equilibrium as a training target.

RK4 is called only after all PINN windows are trained, for independent
validation and plotting.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# -------------------------- model parameters --------------------------
K = 5_000_000.0
b = 0.001
mu = 1.2e-8
c = 0.1
gamma_rate = 0.05
S0, E0, I0 = 4_999_700.0, 200.0, 100.0

M_CASES: Dict[str, float] = {"m=0": 0.0, "m=1e-6": 1e-6, "m=6e-6": 6e-6}
REGIMES: Dict[str, float] = {"DFE": 0.1, "EE": 0.001}
FIXED_HORIZONS: Dict[str, float] = {"DFE": 20_000.0, "EE": 20_000.0}


@dataclass
class Config:
    adam_iters_per_window: int = 1400
    adam_lr: float = 1.5e-3
    lbfgs_iters_per_window: int = 600
    lbfgs_lr: float = 0.5
    lbfgs_history: int = 50
    n_uniform: int = 650
    n_chebyshev: int = 350
    resample_every: int = 200
    hidden_width: int = 64
    hidden_layers: int = 4
    # Model-based scales for residuals in transformed coordinates.
    # These are rate scales (1/time), not population-rate floors.
    transformed_scale_s: float = max(b, mu * K)
    transformed_scale_e: float = c + REGIMES["EE"] + mu * K
    transformed_scale_i: float = c + gamma_rate
    endpoint_physics_weight: float = 1.0
    bound_weight: float = 1.0
    l2_weight: float = 1.0e-12
    grad_clip: float = 10.0
    latent_scale_s: float = 10.0
    latent_scale_e: float = 8.0
    latent_scale_i: float = 8.0
    growth_efolds_first_window: float = 1.5
    window_growth: float = 1.4
    max_window_width: float = 3_000.0
    n_plot_per_window: int = 501
    dtype: str = "float64"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="physics_only_v4_windowed_probe")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--regime", choices=["DFE", "EE"], default="EE")
    p.add_argument("--media", choices=["0", "1e-6", "6e-6"], default="0")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--adam-iters-per-window", type=int, default=None)
    p.add_argument("--lbfgs-iters-per-window", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="Smoke test only; do not report.")
    p.add_argument("--max-windows", type=int, default=None, help="Train only the first N windows for diagnosis.")
    return p.parse_args()


def select_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reproduction_number(d: float) -> float:
    return mu * c * K / (gamma_rate * (c + d))


def infected_linear_eigenvalues_at_dfe(d: float) -> np.ndarray:
    A = np.array([[-(c + d), mu * K], [c, -gamma_rate]], dtype=float)
    return np.linalg.eigvals(A)


def build_window_edges(d: float, t_final: float, cfg: Config) -> np.ndarray:
    """Model-derived causal windows, with no numerical trajectory inspection.

    For an unstable DFE, the first width is a short model-derived number of e-folding times of the
    positive infected-subsystem eigenvalue.  Widths then grow geometrically.
    For a stable DFE, use three e-folding times of the slowest infected decay.
    """
    eig = infected_linear_eigenvalues_at_dfe(d)
    re = np.real(eig)
    pos = re[re > 0]
    if len(pos):
        rate = float(np.max(pos))
    else:
        rate = float(np.min(np.abs(re[re < 0])))
    first_width = cfg.growth_efolds_first_window / max(rate, 1e-8)
    first_width = float(np.clip(first_width, 250.0, 1500.0))

    edges = [0.0]
    width = first_width
    while edges[-1] < t_final - 1e-12:
        nxt = min(t_final, edges[-1] + width)
        edges.append(float(nxt))
        width = min(width * cfg.window_growth, cfg.max_window_width)
    return np.array(edges, dtype=float)


def invariant_bound_fraction(d: float) -> float:
    delta = min(d, gamma_rate)
    return (K + b * K / (4.0 * delta)) / K


def analytic_equilibrium(m: float, d: float) -> Tuple[float, float, float]:
    """Diagnostic only. Never used by train_window(). Returns population units."""
    R0 = reproduction_number(d)
    if R0 <= 1.0:
        return K, 0.0, 0.0
    if m == 0.0:
        i = (b / mu) * (1.0 - 1.0 / R0)
        s = K / R0
        e = (gamma_rate / c) * i
        return s, e, i

    def f(i):
        mi = min(m * i, 700.0)
        A = K * (1.0 - (mu / b) * i * math.exp(-mi))
        B = (K / R0) * math.exp(mi)
        return A - B

    lo, hi = 0.0, max(I0, 1.0)
    while f(hi) > 0 and hi < 1e9:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    i = 0.5 * (lo + hi)
    s = (K / R0) * math.exp(m * i)
    e = (gamma_rate / c) * i
    return s, e, i


# ---------------------------- local PINN ----------------------------
def safe_logit(x: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(x.dtype).eps
    x = torch.clamp(x, 100 * eps, 1.0 - 100 * eps)
    return torch.log(x) - torch.log1p(-x)


class LocalNet(nn.Module):
    def __init__(self, y0_frac: np.ndarray, cfg: Config):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = 5  # u, u^2, sin(pi u), sin(2pi u), sin(3pi u)
        for j in range(cfg.hidden_layers):
            layers.append(nn.Linear(in_dim if j == 0 else cfg.hidden_width, cfg.hidden_width))
            layers.append(nn.Tanh())
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(cfg.hidden_width, 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        y0 = torch.as_tensor(y0_frac, dtype=torch.get_default_dtype())
        z_s0 = safe_logit(y0[0:1])
        z_e0 = torch.log(torch.clamp(y0[1:2], min=1e-14))
        z_i0 = torch.log(torch.clamp(y0[2:3], min=1e-14))
        self.register_buffer("z0", torch.cat([z_s0, z_e0, z_i0]))
        self.register_buffer("latent_scale", torch.tensor([
            cfg.latent_scale_s, cfg.latent_scale_e, cfg.latent_scale_i
        ]))

    def raw(self, u: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([
            u,
            u.square(),
            torch.sin(math.pi * u),
            torch.sin(2.0 * math.pi * u),
            torch.sin(3.0 * math.pi * u),
        ], dim=1)
        return self.out(self.body(feat))

    def state(self, u: torch.Tensor) -> torch.Tensor:
        # Exact continuity at u=0. Latent transforms enforce positivity, and
        # sigmoid enforces 0<S<K without a data penalty.  E and I have no
        # artificial lower log-clamp, so stable DFE trajectories can decay
        # far below one individual while remaining representable in float64.
        z = self.z0.view(1, 3) + u * self.latent_scale.view(1, 3) * self.raw(u)
        s = torch.sigmoid(z[:, 0:1])
        e = torch.exp(torch.clamp(z[:, 1:2], max=0.30))
        i = torch.exp(torch.clamp(z[:, 2:3], max=0.30))
        return torch.cat([s, e, i], dim=1)


def rhs_fraction_torch(y: torch.Tensor, m: float, d: float) -> torch.Tensor:
    s, e, i = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    expo = torch.exp(torch.clamp(-m * K * i, min=-80.0, max=20.0))
    incidence = mu * K * expo * s * i
    return torch.cat([
        b * s * (1.0 - s) - incidence,
        incidence - (c + d) * e,
        c * e - gamma_rate * i,
    ], dim=1)


def rhs_fraction_numpy(y: np.ndarray, m: float, d: float) -> np.ndarray:
    s, e, i = y
    expo = math.exp(float(np.clip(-m * K * i, -80.0, 20.0)))
    incidence = mu * K * expo * s * i
    return np.array([
        b * s * (1.0 - s) - incidence,
        incidence - (c + d) * e,
        c * e - gamma_rate * i,
    ])


def transformed_residual_scales(d: float, cfg: Config) -> np.ndarray:
    """Fixed model-based scales for latent/log-rate residuals (1/time).

    Training E and I with log-state residuals prevents the near-zero disease
    branch from becoming an artificially easy small-residual solution.  The
    scales depend only on fixed model coefficients, never on RK4 or a trajectory.
    """
    return np.array([
        max(b, mu * K),
        c + d + mu * K,
        c + gamma_rate,
    ], dtype=float)


def sample_u(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    uni = rng.uniform(0.0, 1.0, cfg.n_uniform)
    k = np.arange(cfg.n_chebyshev)
    cheb = 0.5 * (1.0 - np.cos(math.pi * (k + 0.5) / cfg.n_chebyshev))
    u = np.unique(np.concatenate(([0.0, 1.0], uni, cheb)))
    return np.sort(u)


def loss_parts(model: LocalNet, u: torch.Tensor, dt: float, m: float, d: float,
               scales: torch.Tensor, cfg: Config) -> Dict[str, torch.Tensor]:
    """Physics residual in positivity-preserving transformed coordinates.

    z_s = logit(s), z_e = log(e), z_i = log(i).  Because s in (0,1) and
    e,i>0, the transformed ODE is exactly equivalent to the original ODE:

      z_s' = f_s/[s(1-s)],  z_e' = f_e/e,  z_i' = f_i/i.

    Crucially, f_e and f_i becoming numerically small when e,i approach zero
    no longer makes extinction an easy false minimum: the log-rate equations
    retain O(1/time) information about relative growth/decay.
    """
    y = model.state(u)
    s, e, i = y[:,0:1], y[:,1:2], y[:,2:3]

    # Derivatives of transformed states with respect to u.
    z_s = torch.log(s) - torch.log1p(-s)
    z_e = torch.log(e)
    z_i = torch.log(i)
    dz_du = []
    for z in (z_s, z_e, z_i):
        dz_du.append(torch.autograd.grad(z, u, torch.ones_like(z),
                                         create_graph=True, retain_graph=True)[0])
    dz_dt = torch.cat(dz_du, dim=1) / dt

    f = rhs_fraction_torch(y, m, d)
    eps = torch.finfo(y.dtype).eps
    rhs_z = torch.cat([
        f[:,0:1] / torch.clamp(s * (1.0 - s), min=100*eps),
        f[:,1:2] / torch.clamp(e, min=100*eps),
        f[:,2:3] / torch.clamp(i, min=100*eps),
    ], dim=1)
    rz = (dz_dt - rhs_z) / scales.view(1,3)
    comp = torch.mean(rz.square(), dim=0)
    physics = comp.sum()

    rend = torch.cat([rz[0:1], rz[-1:]], dim=0)
    endpoint_physics = torch.mean(rend.square())

    nfrac = torch.sum(y, dim=1, keepdim=True)
    bound = torch.mean(torch.relu(nfrac - invariant_bound_fraction(d)).square())

    # Original-coordinate residuals are diagnostics only.
    dy_du=[]
    for j in range(3):
        dy_du.append(torch.autograd.grad(y[:,j:j+1], u, torch.ones_like(y[:,j:j+1]),
                                         create_graph=True, retain_graph=True)[0])
    r_orig = torch.cat(dy_du, dim=1) / dt - f
    orig_mse = torch.mean(r_orig.square(), dim=0)

    return {
        "physics": physics,
        "res_S": comp[0], "res_E": comp[1], "res_I": comp[2],
        "endpoint_physics": endpoint_physics, "bound": bound,
        "orig_mse_S": orig_mse[0], "orig_mse_E": orig_mse[1], "orig_mse_I": orig_mse[2],
    }


def total_loss(model: LocalNet, parts: Dict[str, torch.Tensor], cfg: Config) -> torch.Tensor:
    l2 = sum(p.square().sum() for p in model.parameters())
    return (parts["physics"]
            + cfg.endpoint_physics_weight * parts["endpoint_physics"]
            + cfg.bound_weight * parts["bound"]
            + cfg.l2_weight * l2)


def train_window(index: int, t0: float, t1: float, y0_frac: np.ndarray,
                 m: float, d: float, seed: int, cfg: Config, device: torch.device):
    rng = np.random.default_rng(seed + 7919 * (index + 1))
    model = LocalNet(y0_frac, cfg).to(device)
    dt = float(t1 - t0)
    scales_np = transformed_residual_scales(d, cfg)
    scales = torch.as_tensor(scales_np, device=device)
    history = []

    def make_u():
        arr = sample_u(cfg, rng)
        return torch.as_tensor(arr, device=device).reshape(-1, 1).requires_grad_(True)

    u = make_u()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.adam_lr)
    start = time.perf_counter()

    for it in range(cfg.adam_iters_per_window):
        if it > 0 and it % cfg.resample_every == 0:
            u = make_u()
        opt.zero_grad(set_to_none=True)
        parts = loss_parts(model, u, dt, m, d, scales, cfg)
        loss = total_loss(model, parts, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if it == 0 or (it + 1) % 25 == 0 or it == cfg.adam_iters_per_window - 1:
            history.append({"window": index, "stage": "adam", "step": it,
                            "t0": t0, "t1": t1, "total": float(loss.detach().cpu()),
                            **{k: float(v.detach().cpu()) for k,v in parts.items()}})

    # deterministic L-BFGS set
    u = make_u()
    opt2 = torch.optim.LBFGS(model.parameters(), lr=cfg.lbfgs_lr,
                             max_iter=cfg.lbfgs_iters_per_window,
                             history_size=cfg.lbfgs_history,
                             line_search_fn="strong_wolfe",
                             tolerance_grad=1e-12, tolerance_change=1e-14)
    ev = 0
    def closure():
        nonlocal ev
        opt2.zero_grad(set_to_none=True)
        parts = loss_parts(model, u, dt, m, d, scales, cfg)
        loss = total_loss(model, parts, cfg)
        loss.backward()
        if ev == 0 or (ev + 1) % 10 == 0:
            history.append({"window": index, "stage": "lbfgs", "step": cfg.adam_iters_per_window + ev,
                            "t0": t0, "t1": t1, "total": float(loss.detach().cpu()),
                            **{k: float(v.detach().cpu()) for k,v in parts.items()}})
        ev += 1
        return loss
    opt2.step(closure)
    elapsed = time.perf_counter() - start

    # dense physics diagnostic
    udiag = torch.linspace(0.0, 1.0, 1001, device=device).reshape(-1,1).requires_grad_(True)
    parts = loss_parts(model, udiag, dt, m, d, scales, cfg)
    with torch.no_grad():
        y_end = model.state(torch.ones((1,1), device=device)).cpu().numpy()[0]
    diag = {
        "window": index, "t0": t0, "t1": t1, "width": dt,
        "S0": y0_frac[0]*K, "E0": y0_frac[1]*K, "I0": y0_frac[2]*K,
        "S1": y_end[0]*K, "E1": y_end[1]*K, "I1": y_end[2]*K,
        "transformed_scale_S": scales_np[0], "transformed_scale_E": scales_np[1], "transformed_scale_I": scales_np[2],
        "physics": float(parts["physics"].detach().cpu()),
        "res_S": float(parts["res_S"].detach().cpu()),
        "res_E": float(parts["res_E"].detach().cpu()),
        "res_I": float(parts["res_I"].detach().cpu()),
        "endpoint_physics": float(parts["endpoint_physics"].detach().cpu()),
        "bound": float(parts["bound"].detach().cpu()),
        "abs_rms_res_S_people_per_time": math.sqrt(float(parts["orig_mse_S"].detach().cpu())) * K,
        "abs_rms_res_E_people_per_time": math.sqrt(float(parts["orig_mse_E"].detach().cpu())) * K,
        "abs_rms_res_I_people_per_time": math.sqrt(float(parts["orig_mse_I"].detach().cpu())) * K,
        "train_seconds": elapsed, "lbfgs_function_evals": ev,
    }
    return model, y_end, history, diag


def eval_window(model: LocalNet, t0: float, t1: float, n: int, device: torch.device):
    t = np.linspace(t0, t1, n)
    u = ((t - t0) / (t1 - t0)).reshape(-1,1)
    with torch.no_grad():
        y = model.state(torch.as_tensor(u, device=device)).cpu().numpy() * K
    return t, y


# ----------------------- independent validation -----------------------
def rhs_people(x: np.ndarray, m: float, d: float) -> np.ndarray:
    S,E,I=x
    q=math.exp(float(np.clip(-m*I,-80,20)))
    inf=mu*q*S*I
    return np.array([b*S*(1-S/K)-inf, inf-(c+d)*E, c*E-gamma_rate*I])


def rk4(m: float, d: float, t_final: float, h: float):
    n=int(math.ceil(t_final/h)); h=t_final/n
    t=np.linspace(0,t_final,n+1); x=np.empty((n+1,3)); x[0]=[S0,E0,I0]
    for k in range(n):
        y=x[k]
        k1=rhs_people(y,m,d); k2=rhs_people(y+.5*h*k1,m,d)
        k3=rhs_people(y+.5*h*k2,m,d); k4=rhs_people(y+h*k3,m,d)
        x[k+1]=y+h*(k1+2*k2+2*k3+k4)/6
    return t,x


def interp(t0,x0,t):
    return np.column_stack([np.interp(t,t0,x0[:,j]) for j in range(3)])


def metrics(pred,true):
    rows=[]
    for j,name in enumerate(["S","E","I"]):
        p=pred[:,j]; y=true[:,j]
        rel=np.linalg.norm(p-y)/np.linalg.norm(y)
        ss=np.sum((y-p)**2); st=np.sum((y-y.mean())**2)
        r2=1-ss/st if st>1e-14 else np.nan
        rmse=np.sqrt(np.mean((p-y)**2)); span=y.max()-y.min()
        nrmse=rmse/(span if span>1e-14 else max(np.mean(np.abs(y)),1.0))
        rows.append(dict(component=name,rel_l2=rel,r2=r2,rmse=rmse,nrmse=nrmse,max_abs_error=np.max(np.abs(p-y))))
    return rows


def write_csv(path: Path, rows: Sequence[dict]):
    if not rows: return
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def main():
    args=parse_args(); cfg=Config()
    if args.adam_iters_per_window is not None: cfg.adam_iters_per_window=args.adam_iters_per_window
    if args.lbfgs_iters_per_window is not None: cfg.lbfgs_iters_per_window=args.lbfgs_iters_per_window
    if args.quick:
        cfg.adam_iters_per_window=2; cfg.lbfgs_iters_per_window=1
        cfg.n_uniform=20; cfg.n_chebyshev=12; cfg.hidden_width=16; cfg.hidden_layers=2
        cfg.n_plot_per_window=31
    torch.set_default_dtype(torch.float64)
    device=select_device(args.device); seed_everything(args.seed)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)

    regime=args.regime; d=REGIMES[regime]; m_label={"0":"m=0","1e-6":"m=1e-6","6e-6":"m=6e-6"}[args.media]; m=M_CASES[m_label]
    T=FIXED_HORIZONS[regime]; edges=build_window_edges(d,T,cfg)
    if args.max_windows is not None:
        n=max(1, min(int(args.max_windows), len(edges)-1))
        edges=edges[:n+1]
        T=float(edges[-1])
    print(f"device={device}; regime={regime}; d={d}; R0={reproduction_number(d):.6f}; {m_label}")
    print("model-derived window edges:", ", ".join(f"{x:.1f}" for x in edges))

    y0=np.array([S0/K,E0/K,I0/K],dtype=float)
    models=[]; all_hist=[]; all_diag=[]
    for j in range(len(edges)-1):
        print(f"window {j+1}/{len(edges)-1}: [{edges[j]:.1f}, {edges[j+1]:.1f}], init=({y0[0]*K:.1f},{y0[1]*K:.1f},{y0[2]*K:.1f})")
        model,y1,hist,diag=train_window(j,edges[j],edges[j+1],y0,m,d,args.seed,cfg,device)
        print(f"  physics={diag['physics']:.3e}; abs RMS residuals=({diag['abs_rms_res_S_people_per_time']:.3g}, {diag['abs_rms_res_E_people_per_time']:.3g}, {diag['abs_rms_res_I_people_per_time']:.3g}); end=({diag['S1']:.1f},{diag['E1']:.1f},{diag['I1']:.1f})")
        models.append(model); all_hist.extend(hist); all_diag.append(diag); y0=y1

    # Assemble global PINN prediction first. RK4 has not been called above.
    ts=[]; ys=[]
    for j,model in enumerate(models):
        tt,yy=eval_window(model,edges[j],edges[j+1],cfg.n_plot_per_window,device)
        if j>0: tt,yy=tt[1:],yy[1:]
        ts.append(tt); ys.append(yy)
    t_pred=np.concatenate(ts); pred=np.vstack(ys)

    # Diagnostic analytic equilibrium (still not a training target).
    eq=np.array(analytic_equilibrium(m,d))
    endpoint=pred[-1]
    eq_rel=np.array([(endpoint[k]-eq[k])/max(abs(eq[k]),1.0) for k in range(3)])

    print("training complete; starting independent RK4 validation")
    tr,xr=rk4(m,d,T,h=T/200000.0)
    ref=interp(tr,xr,t_pred)
    mets=metrics(pred,ref)
    for r in mets:
        r.update({"regime":regime,"d":d,"R0":reproduction_number(d),"m_label":m_label,"m":m,"seed":args.seed})
    print("validation metrics:")
    for r in mets: print(f"  {r['component']}: relL2={r['rel_l2']:.4e}, R2={r['r2']:.6f}, NRMSE={r['nrmse']:.4e}, maxAE={r['max_abs_error']:.4g}")
    print("analytic equilibrium (diagnostic only):",eq)
    print("PINN endpoint:",endpoint,"relative endpoint-equilibrium error:",eq_rel)

    write_csv(out/"window_diagnostics.csv",all_diag)
    write_csv(out/"training_history.csv",all_hist)
    write_csv(out/"physics_only_metrics_by_seed.csv",mets)
    np.savez_compressed(out/f"prediction_{regime}_{m_label.replace('=','_')}_seed{args.seed}.npz",t=t_pred,pinn=pred,rk4=ref,window_edges=edges)
    with (out/"run_config.json").open("w") as f:
        json.dump({"config":asdict(cfg),"args":vars(args),"window_edges":edges.tolist(),"dfe_infected_eigenvalues":[str(z) for z in infected_linear_eigenvalues_at_dfe(d)]},f,indent=2)

    fig,ax=plt.subplots(3,1,figsize=(10,11),sharex=True)
    for j,name in enumerate(["S","E","I"]):
        ax[j].plot(t_pred,ref[:,j],lw=1.6,label="independent RK4")
        ax[j].plot(t_pred,pred[:,j],"--",lw=1.3,label="windowed physics-only PINN")
        for e in edges[1:-1]: ax[j].axvline(e,lw=.5,alpha=.25)
        ax[j].set_ylabel(name); ax[j].grid(alpha=.25); ax[j].legend(fontsize=8)
    ax[-1].set_xlabel("time")
    fig.suptitle(f"Windowed physics-only PINN vs independent RK4: {regime}, {m_label}, R0={reproduction_number(d):.3f}")
    fig.tight_layout(); fig.savefig(out/f"sei_{regime}_windowed_physics_only_rk4_vs_pinn.png",dpi=180,bbox_inches="tight"); plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,5))
    for j in range(len(edges)-1):
        vals=[q for q in all_hist if q['window']==j]
        if vals:
            ax.plot([q['step'] for q in vals],[q['total'] for q in vals],lw=1,label=f"W{j+1}")
    ax.set_yscale("log"); ax.set_xlabel("optimizer evaluation within window"); ax.set_ylabel("local physics loss")
    ax.grid(alpha=.25,which="both"); ax.legend(ncol=2,fontsize=7); fig.tight_layout()
    fig.savefig(out/f"sei_{regime}_windowed_training_losses.png",dpi=180,bbox_inches="tight"); plt.close(fig)

    print("saved outputs to",out)

if __name__=="__main__":
    main()
