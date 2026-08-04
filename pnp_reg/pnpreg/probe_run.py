"""Experiment 2 driver: calibrate the probe, then diagnose PIRATE/PIRATE+.

    ~/miniforge3/envs/lpn_env/bin/python -m pnpreg.probe_run --smoke   (~1 min, CPU)
    ~/miniforge3/envs/lpn_env/bin/python -m pnpreg.probe_run --rows cal_mixture cal_quadrature cal_icnn cal_icnn32
    ~/miniforge3/envs/lpn_env/bin/python -m pnpreg.probe_run --device mps   (~2.5-3 h, the production table)

Protocol (task document Sections 5.2 and 6.2; DESIGN.md, Experiment 2):

1. CALIBRATION FIRST, ALWAYS. Whenever a PIRATE row is requested, the three
   calibration rows and the floor row run first and their gates must pass;
   otherwise the JSON records gates_passed = false, no PIRATE number is
   produced, and the exit code is 1 (kill criterion: "a number from an
   uncalibrated estimator is worthless"). --force overrides for debugging.
2. Per test point: `hutchinson_asymmetry` (rho with bootstrap SE and the
   identity_max arithmetic-noise diagnostic) and `lanczos_multistart`
   (lambda_min/lambda_max of the symmetric part, Ritz residual bounds).
3. The noise floor of the asymmetry column is max(floor-row rho, largest
   identity_max in the row); the table prints "<= floor" unless
   rho > floor + 2 SE (kill criterion: asymmetry at the level of the probe
   noise is reported as such, not as asymmetry).

Outputs (tracked): results/probe_metrics[<tag>].json, results/probe_table.md,
results/probe_table.tex. All seeds, budgets, and checkpoint names are in the
JSON config block; rerunning this script with the same flags reproduces every
number up to MPS convolution nondeterminism, which the reported Monte Carlo
standard errors and the CPU cross-check (--device cpu --tag cpu_check) bound.
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch

from . import paths
from . import probe
from . import probe_targets as pt

# eigenvalue tolerance floors by dtype: below these, a "violation" is
# arithmetic, not geometry (float32 network + float32 products ~ 1e-3;
# cf. tests/test_readout.py on the float32 precision budget)
DTYPE_TOL = {"torch.float64": 1e-8, "torch.float32": 1e-3}

SEEDS = {"cal_mixture": 11, "cal_quadrature": 12, "cal_icnn": 13,
         "pirate": 14, "pirate_plus": 14, "floor": 14, "probes": 1000,
         "lanczos": 2000}

CAL_ROWS = ("cal_mixture", "cal_quadrature", "cal_icnn", "cal_icnn32", "floor")
PIRATE_ROWS = ("pirate", "pirate_plus")
DEFAULT_ROWS = ("cal_mixture", "cal_quadrature", "cal_icnn", "cal_icnn32",
                "floor", "pirate", "pirate_plus")

ok_gates = True


def report(name, err, tol):
    """The tests' reporter, reused verbatim so gate output reads the same."""
    global ok_gates
    good = err < tol
    ok_gates &= good
    print(f"  [{'ok ' if good else 'FAIL'}] {name:58s} {err:.3e}")
    return {"name": name, "err": float(err), "tol": float(tol), "ok": bool(good)}


def build_row_cases(row, cfg):
    """Instantiate the PointCases for one table row."""
    if row == "cal_mixture":
        return pt.target_cal_mixture(n=64, sigma=0.5, n_points=cfg["cal_points"],
                                     seed=SEEDS["cal_mixture"])
    if row == "cal_quadrature":
        return pt.target_cal_quadrature(n_points=2 * cfg["cal_points"],
                                        seed=SEEDS["cal_quadrature"])
    if row == "cal_icnn":
        return pt.target_cal_icnn(n_points=cfg["cal_points"],
                                  seed=SEEDS["cal_icnn"],
                                  dtype=torch.float64, cg_tol=1e-12)
    if row == "cal_icnn32":
        return pt.target_cal_icnn(n_points=max(cfg["cal_points"] // 2, 1),
                                  seed=SEEDS["cal_icnn"],
                                  dtype=torch.float32, cg_tol=1e-6)
    if row == "floor":
        return pt.target_floor(n_points=cfg["floor_points"],
                               seed=SEEDS["floor"], device=cfg["device"],
                               crop=cfg["crop"])
    if row in PIRATE_ROWS:
        return pt.target_pirate(row, n_points=cfg["n_test"],
                                seed=SEEDS[row], device=cfg["device"],
                                crop=cfg["crop"])
    raise ValueError(f"unknown row {row}")


def fd_sanity(op, seed=99, h_rel=1e-2):
    """One-probe check that the jvp is a directional derivative of D at all:
    catches a silently wrong torch.func composition, not float noise."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(op.n, generator=g, dtype=torch.float64).to(dtype=op.dtype)
    if op.device != "cpu":
        v = v.to(op.device)
    Jv = op.jvp(v).detach().cpu().double()
    h = h_rel * float(op.z.norm()) / float(v.norm())
    vt = v.reshape(op.shape)
    fd = (op.apply_D(op.z + h * vt) - op.apply_D(op.z - h * vt)).reshape(-1)
    fd = fd.detach().cpu().double() / (2 * h)
    return float((Jv - fd).norm() / fd.norm())


def run_point(case, cfg, verbose=True):
    """All estimators on one PointCase; returns the JSON record."""
    t0 = time.time()
    op = case.get_op()
    hut = probe.hutchinson_asymmetry(op, m_probes=cfg["probes"],
                                     seed=SEEDS["probes"] + 37 * case.index)
    lan = probe.lanczos_multistart(op, k=cfg["k"], starts=cfg["starts"],
                                   seed0=SEEDS["lanczos"] + 37 * case.index)
    tol_eig = max(lan["res_lmin"], lan["res_lmax"],
                  DTYPE_TOL.get(str(op.dtype), 1e-3))
    rec = {
        "index": case.index,
        "n": op.n,
        "dtype": str(op.dtype),
        "rho": hut["rho"], "rho_se": hut["rho_se"],
        "F2": hut["F2"], "F2_se": hut["F2_se"],
        "K2": hut["K2"], "K2_se": hut["K2_se"],
        "identity_max": hut["identity_max"],
        "lmin": lan["lmin"], "lmax": lan["lmax"],
        "res_lmin": lan["res_lmin"], "res_lmax": lan["res_lmax"],
        "spread_lmin": lan["spread_lmin"], "spread_lmax": lan["spread_lmax"],
        "orth_err": lan["orth_err"],
        "lanczos_starts": lan["per_start"],
        "tol_eig": tol_eig,
        "viol2": max(0.0, -lan["lmin"]),
        "viol3": max(0.0, lan["lmax"] - 1.0),
        "exact": case.exact,
        "meta": case.meta,
        "seconds": None,
    }
    case.release()
    # the MPS caching allocator holds freed graphs across points; without an
    # explicit flush the process footprint grows by ~2 GB per point until
    # jetsam kills it (observed twice, 2026-07-30)
    if op.device == "mps":
        import gc
        del op
        gc.collect()
        torch.mps.empty_cache()
    rec["seconds"] = round(time.time() - t0, 2)
    if verbose:
        print(f"    {case.name}[{case.index}] n={rec['n']}: "
              f"rho {rec['rho']:.3e}±{rec['rho_se']:.1e} | "
              f"lmin {rec['lmin']:.4f} lmax {rec['lmax']:.4f} "
              f"(res {rec['res_lmin']:.1e}/{rec['res_lmax']:.1e}) | "
              f"{rec['seconds']:.0f}s", flush=True)
    return rec


def summarize_row(name, points):
    rhos = [p["rho"] for p in points]
    tol = max(p["tol_eig"] for p in points)
    v2 = [p for p in points if p["lmin"] < -tol]
    v3 = [p for p in points if p["lmax"] > 1.0 + tol]
    return {
        "n_points": len(points),
        "rho_mean": float(np.mean(rhos)),
        "rho_min": float(np.min(rhos)),
        "rho_max": float(np.max(rhos)),
        "rho_se_max": float(np.max([p["rho_se"] for p in points])),
        "identity_max": float(np.max([p["identity_max"] for p in points])),
        "lmin_min": float(np.min([p["lmin"] for p in points])),
        "lmax_max": float(np.max([p["lmax"] for p in points])),
        "tol_eig_max": tol,
        "frac_viol2": len(v2) / len(points),
        "frac_viol3": len(v3) / len(points),
        "max_viol2": float(np.max([p["viol2"] for p in v2])) if v2 else 0.0,
        "max_viol3": float(np.max([p["viol3"] for p in v3])) if v3 else 0.0,
        "max_mc_se": float(np.max([max(p["rho_se"], p["res_lmin"], p["res_lmax"])
                                   for p in points])),
    }


def check_gates(rows):
    """The calibration gates (DESIGN.md, Experiment 2). Returns gate records;
    sets the module-level ok_gates flag consumed by main()."""
    gates = []
    print("calibration gates:")
    for name, rec in rows.items():
        pts = rec["points"]
        if name == "cal_mixture":
            for p in pts:
                e = p["exact"]
                gates.append(report(f"mixture[{p['index']}] rho == 0", p["rho"], 1e-7))
                gates.append(report(f"mixture[{p['index']}] lmin vs exact",
                                    abs(p["lmin"] - e["lmin"]),
                                    max(1e-6, p["res_lmin"])))
                gates.append(report(f"mixture[{p['index']}] lmax vs exact",
                                    abs(p["lmax"] - e["lmax"]),
                                    max(1e-6, p["res_lmax"])))
                z = abs(p["F2"] - e["F2"]) / max(p["F2_se"], 1e-300)
                gates.append(report(f"mixture[{p['index']}] F2 z-score", z, 4.0))
            p0 = pts[0]
            gates.append(report("mixture[0] condition-3 failure detected",
                                1.0 if p0["lmax"] <= 1.0 + 1e-6 else 0.0, 0.5))
        elif name == "cal_quadrature":
            for p in pts:
                e = p["exact"]
                gates.append(report(f"quad[{p['index']}] rho vs exact",
                                    abs(p["rho"] - e["rho"]), 1e-6))
                gates.append(report(f"quad[{p['index']}] lmin vs exact",
                                    abs(p["lmin"] - e["lmin"]), 1e-10))
                gates.append(report(f"quad[{p['index']}] lmax vs exact",
                                    abs(p["lmax"] - e["lmax"]), 1e-10))
                gates.append(report(f"quad[{p['index']}] theorem lmin >= 0",
                                    max(0.0, -p["lmin"]), 1e-6))
                gates.append(report(f"quad[{p['index']}] theorem lmax <= 1",
                                    max(0.0, p["lmax"] - 1.0), 1e-6))
                gates.append(report(f"quad[{p['index']}] FD h vs 2h",
                                    p["meta"]["fd_h_vs_2h"], 1e-7))
        elif name in ("cal_icnn", "cal_icnn32"):
            f32 = name.endswith("32")
            tol_r = 1e-3 if f32 else 1e-8
            tol_e = 1e-3 if f32 else 1e-6
            for p in pts:
                e = p["exact"]
                # float32 L-BFGS bottoms out at ~1e-4 (gradient noise ~ eps32
                # * ||H||); 1e-3 is the protocol's float32 tolerance, and the
                # eigenvalue gates below verify the operator independently
                gates.append(report(f"{name}[{p['index']}] prox residual",
                                    p["meta"]["prox_resid"], 1e-3 if f32 else 1e-7))
                gates.append(report(f"{name}[{p['index']}] H PSD (eig_min)",
                                    max(0.0, -e["H_eig_min"]), 1e-6 if f32 else 1e-8))
                gates.append(report(f"{name}[{p['index']}] rho == 0 (CG vs dense)",
                                    p["rho"], tol_r))
                gates.append(report(f"{name}[{p['index']}] lmin vs exact",
                                    abs(p["lmin"] - e["lmin"]),
                                    max(tol_e, p["res_lmin"])))
                gates.append(report(f"{name}[{p['index']}] lmax vs exact",
                                    abs(p["lmax"] - e["lmax"]),
                                    max(tol_e, p["res_lmax"])))
        elif name == "floor":
            for p in pts:
                gates.append(report(f"floor[{p['index']}] rho == 0 (symmetric surrogate)",
                                    p["rho"], 1e-6))
    return gates


def effective_floor(rows):
    """The resolvable-asymmetry floor for the PIRATE rows: the floor row's
    largest measured rho (an end-to-end zero-test, 0 in practice) or the
    largest jvp-vs-vjp bilinear inconsistency, whichever is larger."""
    vals = [0.0]
    if "floor" in rows:
        vals.append(rows["floor"]["summary"]["rho_max"])
        vals.append(rows["floor"]["summary"]["identity_max"])
    for r in PIRATE_ROWS:
        if r in rows:
            vals.append(rows[r]["summary"]["identity_max"])
    return float(max(vals))


ROW_LABEL = {
    "cal_mixture": "CAL mixture (exact, d=64)",
    "cal_quadrature": "CAL quadrature u_PM (exact, d=2)",
    "cal_icnn": "CAL ICNN prox (float64, d=64)",
    "cal_icnn32": "CAL ICNN prox (float32, d=64)",
    "floor": "floor: symmetric surrogate",
    "pirate": "PIRATE (sigma=1)",
    "pirate_plus": "PIRATE+",
}


def format_rho(name, s, floor):
    if name in PIRATE_ROWS and s["rho_mean"] <= floor + 2 * s["rho_se_max"]:
        return f"<= floor ({floor:.1e})"
    cell = f"{s['rho_mean']:.3g} [{s['rho_min']:.3g}, {s['rho_max']:.3g}] ± {s['rho_se_max']:.1g}"
    return cell


def write_table(rows, floor, path_md, path_tex):
    hdr = ["row", "pts", "rho (mean [min,max] ± SE)", "lambda_min(S)",
           "lambda_max(S)", "cond2 viol frac(max)", "cond3 viol frac(max)"]
    lines_md = ["| " + " | ".join(hdr) + " |",
                "|" + "|".join("---" for _ in hdr) + "|"]
    lines_tex = []
    for name, rec in rows.items():
        s = rec["summary"]
        if name == "floor":
            # only rho is meaningful here: the surrogate's spectrum is its
            # own, not any denoiser's
            cells = [ROW_LABEL.get(name, name), str(s["n_points"]),
                     format_rho(name, s, floor), "--", "--", "--", "--"]
        else:
            cells = [ROW_LABEL.get(name, name), str(s["n_points"]),
                     format_rho(name, s, floor),
                     f"{s['lmin_min']:.4f}", f"{s['lmax_max']:.4f}",
                     f"{s['frac_viol2']:.2f} ({s['max_viol2']:.3g})",
                     f"{s['frac_viol3']:.2f} ({s['max_viol3']:.3g})"]
        lines_md.append("| " + " | ".join(cells) + " |")
        lines_tex.append(" & ".join(cells).replace("<=", r"$\le$")
                         .replace("±", r"$\pm$") + r" \\")
    md = "\n".join(lines_md) + f"\n\nasymmetry floor: {floor:.3e}\n"
    with open(path_md, "w") as f:
        f.write(md)
    with open(path_tex, "w") as f:
        f.write("% generated by pnpreg.probe_run; columns: " + "; ".join(hdr) + "\n")
        f.write("\n".join(lines_tex) + "\n")
    print("\n" + md)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="cropped field, tiny budgets, CPU; ~1 min end to end")
    ap.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    ap.add_argument("--rows", nargs="+", default=None, choices=DEFAULT_ROWS,
                    help="subset of rows (calibration is prepended whenever a "
                         "PIRATE row is requested)")
    ap.add_argument("--probes", type=int, default=16)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--cal-points", type=int, default=8)
    ap.add_argument("--floor-points", type=int, default=2)
    ap.add_argument("--force", action="store_true",
                    help="report PIRATE rows even if gates fail (debugging only)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    cfg = {"probes": a.probes, "k": a.k, "starts": a.starts,
           "n_test": a.n_test, "cal_points": a.cal_points,
           "floor_points": a.floor_points, "device": a.device, "crop": None}
    rows_req = list(a.rows) if a.rows else list(DEFAULT_ROWS)
    if a.smoke:
        cfg.update(probes=2, k=6, starts=1, n_test=1, cal_points=2,
                   floor_points=1, device="cpu", crop=(24, 24, 24))
        rows_req = ["cal_mixture", "floor", "pirate", "pirate_plus"]
        a.tag = a.tag or "smoke"

    # calibration always precedes a PIRATE row
    if any(r in PIRATE_ROWS for r in rows_req):
        need = [r for r in (CAL_ROWS if not a.smoke else ("cal_mixture", "floor"))
                if r not in rows_req]
        rows_req = need + rows_req
    rows_req = sorted(set(rows_req), key=DEFAULT_ROWS.index)

    print(f"probe_run: rows {rows_req}, device {cfg['device']}, "
          f"probes {cfg['probes']}, k {cfg['k']}, starts {cfg['starts']}, "
          f"n_test {cfg['n_test']}, crop {cfg['crop']}", flush=True)

    rows = {}
    for row in rows_req:
        if row in PIRATE_ROWS:
            continue  # after gating
        print(f"  row {row}:", flush=True)
        cases = build_row_cases(row, cfg)
        # calibration operators are tiny; extra probes are free and make the
        # statistical gates (F2 z-score) reliable even at smoke budgets
        cfg_row = cfg if row == "floor" else {**cfg, "probes": max(cfg["probes"], 64)}
        pts = [run_point(c, cfg_row) for c in cases]
        rows[row] = {"points": pts, "summary": summarize_row(row, pts)}

    gates = check_gates(rows)
    passed = ok_gates
    sanity = {}
    if passed or a.force:
        for row in [r for r in rows_req if r in PIRATE_ROWS]:
            print(f"  row {row}:", flush=True)
            cases = build_row_cases(row, cfg)
            s = fd_sanity(cases[0].get_op())
            sanity[row] = s
            print(f"    jvp vs FD directional derivative (rel): {s:.2e}", flush=True)
            if s > 2e-2:
                raise RuntimeError(f"{row}: jvp disagrees with finite differences "
                                   f"({s:.2e}); torch.func composition broken")
            cases[0].release()
            # per-point checkpointing: a killed run resumes instead of
            # recomputing ~12-minute points; the partial file is keyed to the
            # budget so a config change invalidates it
            key = {"probes": cfg["probes"], "k": cfg["k"],
                   "starts": cfg["starts"], "device": cfg["device"],
                   "crop": cfg["crop"]}
            ppath = os.path.join(paths.RESULTS, f".probe_partial_{row}{('_' + a.tag) if a.tag else ''}.json")
            done = {}
            if os.path.exists(ppath):
                with open(ppath) as f:
                    part = json.load(f)
                if part.get("key") == key:
                    done = {p["index"]: p for p in part["points"]}
                    print(f"    resuming: {sorted(done)} already recorded", flush=True)
            pts = []
            for c in cases:
                if c.index in done:
                    pts.append(done[c.index])
                    continue
                pts.append(run_point(c, cfg))
                with open(ppath, "w") as f:
                    json.dump({"key": key, "points": pts}, f)
            with open(ppath, "w") as f:
                json.dump({"key": key, "points": pts}, f)
            rows[row] = {"points": pts, "summary": summarize_row(row, pts)}
            # the partial file is KEPT after completion: a later invocation
            # (e.g. a combined run after chunked ones) reuses the points
            # instead of recomputing ~10-minute Jacobian probes
    elif any(r in PIRATE_ROWS for r in rows_req):
        print("GATES FAILED -- no PIRATE numbers produced (kill criterion 1). "
              "Fix the estimator or rerun with --force to debug.", flush=True)
    else:
        print("GATES FAILED (calibration-only run).", flush=True)

    floor = effective_floor(rows)
    paths.ensure_dirs()
    tag = f"_{a.tag}" if a.tag else ""
    out = {
        "config": {
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "torch": torch.__version__,
            "argv": sys.argv[1:] if argv is None else argv,
            **cfg,
            "seeds": SEEDS,
            "checkpoints": {"pirate": os.path.relpath(paths.PIRATE_CKPT_AWGN, paths.ROOT),
                            "pirate_plus": os.path.relpath(paths.PIRATE_CKPT_PLUS, paths.ROOT)},
            "field": os.path.relpath(paths.PIRATE_FIELD, paths.ROOT),
            "fd_sanity": sanity,
        },
        "gates_passed": bool(passed),
        "gates": gates,
        "asymmetry_floor": floor,
        "rows": rows,
    }
    jpath = os.path.join(paths.RESULTS, f"probe_metrics{tag}.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {jpath}", flush=True)
    if not a.smoke and not tag:
        write_table(rows, floor,
                    os.path.join(paths.RESULTS, "probe_table.md"),
                    os.path.join(paths.RESULTS, "probe_table.tex"))
    elif rows:
        write_table(rows, floor,
                    os.path.join(paths.RESULTS, f"probe_table{tag}.md"),
                    os.path.join(paths.RESULTS, f"probe_table{tag}.tex"))
    return 0 if (passed or a.force) else 1


if __name__ == "__main__":
    sys.exit(main())
