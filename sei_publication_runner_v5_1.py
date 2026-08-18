#!/usr/bin/env python3
"""
Publication runner for the frozen SEI causal/windowed physics-only PINN v5.1.

This script does NOT alter the PINN method. It launches the frozen core script
for selected regimes, media strengths, and random seeds; then aggregates
validation metrics and diagnostics. RK4 is used only inside the frozen core
after each PINN has finished training, plus an optional one-time RK4
step-refinement check performed after all requested PINN runs are complete.

Typical use now (seed 0 already validated separately):
    python sei_publication_runner_v5_1.py --seeds 1 2 --device cuda

Full reproducibility run from scratch:
    python sei_publication_runner_v5_1.py --seeds 0 1 2 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MEDIA_VALUES = ["0", "1e-6", "6e-6"]
REGIMES = ["DFE", "EE"]
METRIC_COLS = ["rel_l2", "r2", "rmse", "nrmse", "max_abs_error"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core-script", default=None,
                   help="Path to sei_windowed_physics_only_pinn_v5_1.py; default: beside this runner")
    p.add_argument("--output-root", default="sei_publication_v5_1")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--regimes", nargs="+", choices=REGIMES, default=REGIMES)
    p.add_argument("--media", nargs="+", choices=MEDIA_VALUES, default=MEDIA_VALUES)
    p.add_argument("--skip-existing", action="store_true",
                   help="Reuse a completed run directory if its metrics CSV already exists")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Do not train; aggregate already completed standardized run directories")
    p.add_argument("--no-rk4-refinement", action="store_true",
                   help="Skip the post-training RK4 h/h/2/h/4 convergence check")
    return p.parse_args()


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("sei_v5_1_core", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def media_tag(m: str) -> str:
    return {"0": "m0", "1e-6": "m1e-6", "6e-6": "m6e-6"}[m]


def run_dir(root: Path, regime: str, media: str, seed: int) -> Path:
    return root / f"{regime}_{media_tag(media)}_seed{seed}"


def launch_one(core_script: Path, out: Path, regime: str, media: str,
               seed: int, device: str):
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(core_script),
        "--regime", regime,
        "--media", media,
        "--seed", str(seed),
        "--device", device,
        "--output-dir", str(out),
    ]
    log_path = out / "console.log"
    print("\n>>> RUN", " ".join(cmd))
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = ""
        try:
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:])
        except Exception:
            pass
        raise RuntimeError(f"Run failed: {regime}, m={media}, seed={seed}\n{tail}")
    print(f"    completed in {elapsed/60:.2f} min; log: {log_path}")


def collect_metrics(root: Path, regimes, media_values, seeds):
    frames = []
    diag_frames = []
    peak_rows = []
    endpoint_rows = []

    for regime in regimes:
        for media in media_values:
            for seed in seeds:
                rd = run_dir(root, regime, media, seed)
                mp = rd / "physics_only_metrics_by_seed.csv"
                dp = rd / "window_diagnostics.csv"
                if not mp.exists():
                    print(f"WARNING: missing {mp}")
                    continue
                mdf = pd.read_csv(mp)
                mdf["run_dir"] = str(rd)
                frames.append(mdf)

                if dp.exists():
                    ddf = pd.read_csv(dp)
                    ddf["regime"] = regime
                    ddf["media_arg"] = media
                    ddf["seed"] = seed
                    ddf["run_dir"] = str(rd)
                    diag_frames.append(ddf)

                npzs = sorted(rd.glob(f"prediction_{regime}_*_seed{seed}.npz"))
                if npzs:
                    data = np.load(npzs[0])
                    t = data["t"]
                    pinn = data["pinn"]
                    rk4 = data["rk4"]
                    for j, comp in [(1, "E"), (2, "I")]:
                        ip = int(np.argmax(pinn[:, j])); ir = int(np.argmax(rk4[:, j]))
                        ppk = float(pinn[ip, j]); rpk = float(rk4[ir, j])
                        peak_rows.append({
                            "regime": regime, "media_arg": media, "seed": seed,
                            "component": comp,
                            "pinn_peak": ppk, "rk4_peak": rpk,
                            "peak_rel_error": (ppk-rpk)/max(abs(rpk), 1.0),
                            "pinn_peak_time": float(t[ip]), "rk4_peak_time": float(t[ir]),
                            "peak_time_error": float(t[ip]-t[ir]),
                        })
                    for j, comp in enumerate(["S", "E", "I"]):
                        endpoint_rows.append({
                            "regime": regime, "media_arg": media, "seed": seed,
                            "component": comp,
                            "pinn_endpoint": float(pinn[-1,j]),
                            "rk4_endpoint": float(rk4[-1,j]),
                            "endpoint_abs_error": float(pinn[-1,j]-rk4[-1,j]),
                            "endpoint_rel_error": float((pinn[-1,j]-rk4[-1,j]) / max(abs(rk4[-1,j]),1.0)),
                        })

    metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    diagnostics = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    peaks = pd.DataFrame(peak_rows)
    endpoints = pd.DataFrame(endpoint_rows)
    return metrics, diagnostics, peaks, endpoints


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    group_cols = ["regime", "d", "R0", "m_label", "m", "component"]
    g = metrics.groupby(group_cols, dropna=False)
    pieces = []
    for name, sub in g:
        row = dict(zip(group_cols, name))
        row["n_seeds"] = len(sub)
        for col in METRIC_COLS:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_sd"] = sub[col].std(ddof=1) if len(sub) > 1 else 0.0
            row[f"{col}_min"] = sub[col].min()
            row[f"{col}_max"] = sub[col].max()
        pieces.append(row)
    return pd.DataFrame(pieces)


def aggregate_peaks(peaks: pd.DataFrame) -> pd.DataFrame:
    if peaks.empty:
        return pd.DataFrame()
    group_cols = ["regime", "media_arg", "component"]
    rows = []
    for name, sub in peaks.groupby(group_cols):
        row = dict(zip(group_cols, name))
        row["n_seeds"] = len(sub)
        for col in ["peak_rel_error", "peak_time_error", "pinn_peak", "rk4_peak"]:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_sd"] = sub[col].std(ddof=1) if len(sub) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_diagnostics(diag: pd.DataFrame) -> pd.DataFrame:
    if diag.empty:
        return pd.DataFrame()
    rows = []
    for (regime, media, seed), sub in diag.groupby(["regime", "media_arg", "seed"]):
        rows.append({
            "regime": regime, "media_arg": media, "seed": seed,
            "n_windows": len(sub),
            "training_seconds": sub["train_seconds"].sum(),
            "max_physics_loss": sub["physics"].max(),
            "max_abs_rms_res_S_people_per_time": sub["abs_rms_res_S_people_per_time"].max(),
            "max_abs_rms_res_E_people_per_time": sub["abs_rms_res_E_people_per_time"].max(),
            "max_abs_rms_res_I_people_per_time": sub["abs_rms_res_I_people_per_time"].max(),
            "final_physics_loss": sub["physics"].iloc[-1],
        })
    return pd.DataFrame(rows)


def rk4_refinement(core, regimes, media_values) -> pd.DataFrame:
    rows = []
    m_map = {"0": 0.0, "1e-6": 1e-6, "6e-6": 6e-6}
    for regime in regimes:
        d = core.REGIMES[regime]
        T = core.FIXED_HORIZONS[regime]
        for media in media_values:
            m = m_map[media]
            h = T / 50000.0
            print(f"RK4 refinement: {regime}, m={media}, h={h:g},{h/2:g},{h/4:g}")
            t1,x1 = core.rk4(m,d,T,h)
            t2,x2 = core.rk4(m,d,T,h/2)
            t3,x3 = core.rk4(m,d,T,h/4)
            te = np.linspace(0,T,2001)
            a = core.interp(t1,x1,te); b = core.interp(t2,x2,te); c = core.interp(t3,x3,te)
            for j,comp in enumerate(["S","E","I"]):
                denom2 = max(np.linalg.norm(b[:,j]), 1e-300)
                denom3 = max(np.linalg.norm(c[:,j]), 1e-300)
                rows.append({
                    "regime": regime, "media_arg": media, "component": comp,
                    "h": h,
                    "rel_diff_h_vs_h2": np.linalg.norm(a[:,j]-b[:,j])/denom2,
                    "rel_diff_h2_vs_h4": np.linalg.norm(b[:,j]-c[:,j])/denom3,
                    "max_abs_diff_h_vs_h2": np.max(np.abs(a[:,j]-b[:,j])),
                    "max_abs_diff_h2_vs_h4": np.max(np.abs(b[:,j]-c[:,j])),
                })
    return pd.DataFrame(rows)


def tex_sci(x: float, digits: int = 2) -> str:
    if not np.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / (10**exp)
    return f"${mant:.{digits}f}\\times10^{{{exp}}}$"


def write_metrics_tex(agg: pd.DataFrame, path: Path):
    if agg.empty:
        return
    lines = [
        "% Auto-generated by sei_publication_runner_v5_1.py",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Regime & $m$ & State & Rel. $L_2$ & $R^2$ & NRMSE \\\\",
        "\\midrule",
    ]
    for _,r in agg.sort_values(["regime","m","component"]).iterrows():
        mtxt = {0.0:"$0$",1e-6:"$10^{-6}$",6e-6:"$6\\times10^{-6}$"}.get(float(r["m"]), f"${r['m']}$")
        rel = f"{tex_sci(r['rel_l2_mean'])} $\\pm$ {tex_sci(r['rel_l2_sd'])}"
        r2 = f"${r['r2_mean']:.6f}\\pm{r['r2_sd']:.2e}$"
        nr = f"{tex_sci(r['nrmse_mean'])} $\\pm$ {tex_sci(r['nrmse_sd'])}"
        lines.append(f"{r['regime']} & {mtxt} & {r['component']} & {rel} & {r2} & {nr} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    core_script = Path(args.core_script).resolve() if args.core_script else here / "sei_windowed_physics_only_pinn_v5_1.py"
    if not core_script.exists():
        raise SystemExit(f"Core script not found: {core_script}")
    root = Path(args.output_root).resolve(); root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "frozen_core_script": str(core_script),
        "seeds": args.seeds, "regimes": args.regimes, "media": args.media,
        "device": args.device,
        "note": "PINN v5.1 frozen before multi-seed publication runs; RK4 is validation only.",
    }
    (root / "publication_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.aggregate_only:
        for regime in args.regimes:
            for media in args.media:
                for seed in args.seeds:
                    rd = run_dir(root, regime, media, seed)
                    done = rd / "physics_only_metrics_by_seed.csv"
                    if args.skip_existing and done.exists():
                        print("SKIP existing", rd)
                        continue
                    launch_one(core_script, rd, regime, media, seed, args.device)

    metrics, diag, peaks, endpoints = collect_metrics(root, args.regimes, args.media, args.seeds)
    if metrics.empty:
        raise SystemExit("No completed metrics found to aggregate.")

    metrics.to_csv(root / "all_metrics_by_seed.csv", index=False)
    if not diag.empty: diag.to_csv(root / "all_window_diagnostics.csv", index=False)
    if not peaks.empty: peaks.to_csv(root / "all_peak_diagnostics_by_seed.csv", index=False)
    if not endpoints.empty: endpoints.to_csv(root / "all_endpoint_diagnostics_by_seed.csv", index=False)

    agg = aggregate_metrics(metrics)
    agg.to_csv(root / "metrics_mean_sd.csv", index=False)
    write_metrics_tex(agg, root / "metrics_mean_sd_table.tex")

    pagg = aggregate_peaks(peaks)
    if not pagg.empty: pagg.to_csv(root / "peak_metrics_mean_sd.csv", index=False)

    dagg = aggregate_diagnostics(diag)
    if not dagg.empty:
        dagg.to_csv(root / "training_diagnostics_by_run.csv", index=False)
        runtime = dagg.groupby(["regime","media_arg"])["training_seconds"].agg(["mean","std","min","max"]).reset_index()
        runtime.to_csv(root / "runtime_summary.csv", index=False)

    if not args.no_rk4_refinement:
        core = load_core(core_script)
        ref = rk4_refinement(core, args.regimes, args.media)
        ref.to_csv(root / "rk4_step_refinement.csv", index=False)

    print("\nPublication aggregation complete:", root)
    print("  metrics:", root / "metrics_mean_sd.csv")
    print("  LaTeX table:", root / "metrics_mean_sd_table.tex")
    if len(args.seeds) < 3:
        print("NOTE: aggregates contain only the requested seeds", args.seeds,
              "and are not yet the final 3-seed publication statistics.")


if __name__ == "__main__":
    main()
