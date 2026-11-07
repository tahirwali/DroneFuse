"""
Step 3 driver: every method x protocol x seed, then the tables and figures of the paper.

Each run is written to <out>/runs/ as soon as it finishes:
  <protocol>_threshold.json            the point threshold chosen on validation (DroneFuse, first seed)
  <protocol>_s<seed>_<method>.json     training history and the metrics of the natural test windows
  <protocol>_s<seed>_<method>.npz      validation / test logits, and test logits with sensors removed
Runs already on disk are skipped, so an interrupted notebook continues where it stopped (attach the
previous output as an input and set RESUME_FROM). The test split is never used for any choice: the
threshold, the best epoch and the rejection thresholds all come from validation.
"""
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .fusion import (FUSED, GROUP_NAMES, METHODS, THRESHOLD_CANDIDATES, TrainSettings, all_subsets,
                     evaluate, group_presence, late_fusion, make_task, perturbed_masks, predict, prepare_inputs,
                     presence, single_logits, subset_mask, train, _predict_class)
from .windows import PROTOCOLS

SUBSET_PROTOCOLS = ("closed", "v1_to_v2", "v2_to_v1", "xrec_s1_to_s2", "xrec_s2_to_s1")
LABEL = {"single_camera": "Camera only", "single_audio": "Audio only", "single_avia": "Avia only",
         "single_mid360": "Mid-360 only", "single_radar": "Radar only", "late": "Late fusion",
         "concat": "Feature concatenation", "lmmuavc": "LMMUAVC-style", "dronefuse": "DroneFuse",
         "dronefuse_iid": "DroneFuse (i.i.d. dropping)", "dronefuse_mix": "DroneFuse (mixed dropping)"}


def device_name():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def runs_folder(camera="full"):
    """Runs with whole-frame camera features go to runs/, runs with Step 2b crops to runs_crop/."""
    return "runs" if camera == "full" else f"runs_{camera}"


def resume_from(previous, out_dir, log=print):
    """Copy finished runs of an earlier session (an attached output) into this session's output."""
    copied = 0
    for name in ("runs", "runs_crop"):
        runs = Path(out_dir) / name
        for path in Path(previous).rglob(f"{name}/*"):
            if path.suffix in (".json", ".npz") and not (runs / path.name).exists():
                runs.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, runs / path.name)
                copied += 1
    log(f"resumed {copied} files from {previous}")
    return copied


def _save(path, arrays, info):
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    path.with_suffix(".json").write_text(json.dumps(info, indent=1))


def choose_threshold(data, task, seed, settings, device, runs, log):
    path = runs / f"{task.protocol}_threshold.json"
    if path.exists():
        return json.loads(path.read_text())
    trials = {}
    for t in THRESHOLD_CANDIDATES:
        mask = presence(data.counts, t)
        _, history = train("dronefuse", prepare_inputs(data, task, mask), mask, task, seed, settings, device)
        trials[str(t)] = history.get("best_val_accuracy")
        log(f"  point threshold {t}: DroneFuse val accuracy {trials[str(t)]:.4f}")
    best = max(THRESHOLD_CANDIDATES, key=lambda t: (trials[str(t)], -t))      # ties -> the smaller threshold
    info = {"protocol": task.protocol, "chosen": best, "val_accuracy": trials, "seed": seed}
    path.write_text(json.dumps(info, indent=1))
    return info


def run_protocol(data, windows, protocol, seeds, out_dir, settings=None, device=None, log=print,
                 methods=METHODS, camera="full"):
    settings = settings or TrainSettings()
    device = device or device_name()
    runs = Path(out_dir) / runs_folder(camera)
    runs.mkdir(parents=True, exist_ok=True)
    task = make_task(data, windows, protocol)
    log(f"{protocol}: known {task.known}, unknown {task.unknown}, train {len(task.train)}, val {len(task.val)}, "
        f"test {len(task.test)} ({int((task.y[task.test] < 0).sum())} unknown)")
    threshold = choose_threshold(data, task, seeds[0], settings, device, runs, log)["chosen"]
    mask = presence(data.counts, threshold)
    gmask = group_presence(mask)
    n_groups = gmask.sum(1)
    inputs = prepare_inputs(data, task, mask)
    yv, yt = task.y[task.val], task.y[task.test]
    done = {}
    for seed in seeds:
        single_val, single_test = {}, {}
        for method in methods:
            if method == "late":
                continue
            path = runs / f"{protocol}_s{seed}_{method}"
            if path.with_suffix(".json").exists() and path.with_suffix(".npz").exists():
                saved = np.load(path.with_suffix(".npz"))
                val, test = saved["val"], saved["test"]
            else:
                began = time.time()
                model, history = train(method, inputs, mask, task, seed, settings, device)
                if model is None:
                    path.with_suffix(".json").write_text(json.dumps({"note": history["note"]}))
                    log(f"  {protocol} seed {seed} {method}: {history['note']}")
                    continue
                val = predict(model, inputs, mask, task.val, device)
                test = predict(model, inputs, mask, task.test, device)
                arrays = {}
                if method.startswith("single_"):
                    group = method[len("single_"):]
                    val = single_logits(val, mask, group, task.val)
                    test = single_logits(test, mask, group, task.test)
                elif protocol in SUBSET_PROTOCOLS:
                    for k, subset in enumerate(all_subsets()):
                        forced = mask.copy()
                        forced[task.test] = subset_mask(mask[task.test], subset)
                        arrays[f"subset_{k}"] = predict(model, inputs, forced, task.test, device).astype(np.float16)
                    for kind, m in perturbed_masks(mask, task.test, seed).items():
                        forced = mask.copy()
                        forced[task.test] = m
                        arrays[f"perturb_{kind}"] = predict(model, inputs, forced, task.test, device).astype(np.float16)
                metrics = evaluate(test, yt, n_groups[task.test], val, yv, n_groups[task.val])
                arrays.update(val=val.astype(np.float32), test=test.astype(np.float32))
                _save(path, arrays, {"protocol": protocol, "seed": seed, "method": method, "threshold": threshold,
                                     "history": history, "metrics": metrics,
                                     "seconds": round(time.time() - began, 1)})
                log(f"  {protocol} seed {seed} {method:14s} val {history['best_val_accuracy']:.3f} "
                    f"(epoch {history['best_epoch']}) | test known acc {metrics['known_accuracy']:.3f}"
                    + (f" | AUROC {metrics['energy']['auroc']:.3f}" if metrics['energy']['auroc'] is not None else "")
                    + f" | {time.time() - began:.0f} s")
            if method.startswith("single_"):
                single_val[method[len("single_"):]], single_test[method[len("single_"):]] = val, test
        if "late" in methods and single_val:
            val = late_fusion(single_val, gmask[task.val])
            test = late_fusion(single_test, gmask[task.test])
            metrics = evaluate(test, yt, n_groups[task.test], val, yv, n_groups[task.val])
            _save(runs / f"{protocol}_s{seed}_late", {"val": val, "test": test},
                  {"protocol": protocol, "seed": seed, "method": "late", "threshold": threshold, "metrics": metrics})
        done[seed] = True
    return {"protocol": protocol, "threshold": threshold, "seeds": list(done)}


# Aggregation
def _mean_std(values):
    v = [x for x in values if x is not None]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (None, None, 0)


def summarize(data, windows, out_dir, protocols=PROTOCOLS, seeds=(0, 1, 2), camera="full"):
    """Read every run on disk and compute the summary tables (means and standard deviations over seeds)."""
    runs = Path(out_dir) / runs_folder(camera)
    table, groups_table, subsets, perturb, thresholds = {}, {}, {}, {}, {}
    for protocol in protocols:
        th_path = runs / f"{protocol}_threshold.json"
        if not th_path.exists():
            continue
        thresholds[protocol] = json.loads(th_path.read_text())
        task = make_task(data, windows, protocol)
        mask = presence(data.counts, thresholds[protocol]["chosen"])
        gmask = group_presence(mask)
        yt = task.y[task.test]
        known = yt >= 0
        table[protocol] = {}
        for method in METHODS:
            per_seed = []
            for seed in seeds:
                p = runs / f"{protocol}_s{seed}_{method}.json"
                if p.exists():
                    info = json.loads(p.read_text())
                    if "metrics" in info:
                        per_seed.append(info["metrics"])
            if not per_seed:
                continue
            row = {"seeds": len(per_seed)}
            for key in ("known_accuracy", "known_macro_f1"):
                row[key] = _mean_std([m[key] for m in per_seed])
            for score in ("msp", "energy", "avail_energy"):
                for key in ("auroc", "fpr_at_95_tpr", "open_accuracy"):
                    row[f"{score}_{key}"] = _mean_std([m[score][key] for m in per_seed])
            table[protocol][method] = row
            if protocol == "closed":
                groups_table[method] = {b: _mean_std([m["known_accuracy_by_groups_present"].get(b, {}).get("accuracy")
                                                      for m in per_seed]) for b in map(str, range(1, 6))}
        if protocol not in SUBSET_PROTOCOLS:
            continue
        subsets[protocol], perturb[protocol] = {}, {}
        for method in FUSED + ("late",):
            accs = {k: [] for k in range(len(all_subsets()))}
            pert = {"natural": [], "iid_drop": [], "pattern_drop": []}
            for seed in seeds:
                if method == "late":
                    single = {}
                    for g in GROUP_NAMES:
                        p = runs / f"{protocol}_s{seed}_single_{g}.npz"
                        if p.exists():
                            single[g] = np.load(p)["test"]
                    if not single:
                        continue
                    gm = gmask[task.test]
                    for k, subset in enumerate(all_subsets()):
                        keep = np.array([g in subset for g in GROUP_NAMES])
                        z = late_fusion(single, gm & keep[None, :])
                        accs[k].append(float((_predict_class(z)[known] == yt[known]).mean()))
                    pert["natural"].append(float((_predict_class(late_fusion(single, gm))[known] == yt[known]).mean()))
                    for kind, m in perturbed_masks(mask, task.test, seed).items():
                        z = late_fusion(single, group_presence(m))
                        pert[kind].append(float((_predict_class(z)[known] == yt[known]).mean()))
                    continue
                p = runs / f"{protocol}_s{seed}_{method}.npz"
                if not p.exists():
                    continue
                saved = np.load(p)
                pert["natural"].append(float((_predict_class(saved["test"])[known] == yt[known]).mean()))
                for k in range(len(all_subsets())):
                    if f"subset_{k}" in saved.files:
                        accs[k].append(float((_predict_class(saved[f"subset_{k}"].astype(np.float32))[known] == yt[known]).mean()))
                for kind in ("iid_drop", "pattern_drop"):
                    if f"perturb_{kind}" in saved.files:
                        pert[kind].append(float((_predict_class(saved[f"perturb_{kind}"].astype(np.float32))[known] == yt[known]).mean()))
            subsets[protocol][method] = {"+".join(s): _mean_std(accs[k]) for k, s in enumerate(all_subsets())}
            perturb[protocol][method] = {k: _mean_std(v) for k, v in pert.items()}
    return {"main": table, "closed_by_groups_present": groups_table, "subsets": subsets, "perturbation": perturb,
            "thresholds": thresholds}


# Report, tables, figures
def _pct(ms, digits=1):
    mean, std, n = ms
    if mean is None:
        return "--"
    return f"{100 * mean:.{digits}f}" + (f" $\\pm$ {100 * std:.{digits}f}" if n > 1 else "")


def _txt(ms):
    mean, std, n = ms
    return "  --  " if mean is None else f"{100 * mean:5.1f}+-{100 * std:3.1f}"


def format_step3_report(summary, data, environment=None):
    lines = []
    if environment:
        lines.append(f"MACHINE  CPUs {environment.get('cpu_count')} | GPU {environment.get('gpu')} | "
                     f"run type {environment.get('kaggle_run_type')}")
    lines.append(f"WINDOWS  {len(data)} covered windows; skipped recordings: {data.skipped or 'none'}")
    lines.append("POINT THRESHOLD chosen on validation: " +
                 ", ".join(f"{p} {t['chosen']}" for p, t in summary["thresholds"].items()))
    for protocol, rows in summary["main"].items():
        lines.append(f"\n{protocol}  (mean+-std over seeds, %)")
        lines.append(f"  {'method':28s} {'known acc':>10s} {'macro F1':>10s} {'AUROC MSP':>10s} {'AUROC E':>10s} "
                     f"{'AUROC aE':>10s} {'FPR95 aE':>10s} {'open acc aE':>11s}")
        for method, r in rows.items():
            lines.append(f"  {LABEL[method]:28s} {_txt(r['known_accuracy']):>10s} {_txt(r['known_macro_f1']):>10s} "
                         f"{_txt(r['msp_auroc']):>10s} {_txt(r['energy_auroc']):>10s} "
                         f"{_txt(r['avail_energy_auroc']):>10s} {_txt(r['avail_energy_fpr_at_95_tpr']):>10s} "
                         f"{_txt(r['avail_energy_open_accuracy']):>11s}")
    lines.append("\nCLOSED: known accuracy by number of sensor groups present (1..5)")
    for method, r in summary["closed_by_groups_present"].items():
        lines.append(f"  {LABEL[method]:28s} " + " ".join(f"{_txt(r[b]):>10s}" for b in map(str, range(1, 6))))
    for protocol, rows in summary["perturbation"].items():
        lines.append(f"\n{protocol}: known accuracy with extra sensors removed at test time (natural | i.i.d. | patterns)")
        for method, r in rows.items():
            lines.append(f"  {LABEL[method]:28s} {_txt(r['natural']):>10s} {_txt(r['iid_drop']):>10s} "
                         f"{_txt(r['pattern_drop']):>10s}")
    for protocol, rows in summary["subsets"].items():
        lines.append(f"\n{protocol}: known accuracy when only these sensor groups are kept")
        names = list(next(iter(rows.values())).keys())
        lines.append("  " + " | ".join(f"{LABEL[m][:14]}" for m in rows) + "   subset")
        for name in names:
            lines.append("  " + " | ".join(f"{_txt(rows[m][name]):>14s}" for m in rows) + f"   {name}")
    return "\n".join(lines)


def write_tables(summary, folder):
    """LaTeX tables for the paper (numbers come only from the runs)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    main = summary["main"]
    loo = [p for p in main if p.startswith("loo_")]

    def loo_mean(method, key):
        vals = [main[p][method][key][0] for p in loo if method in main[p] and main[p][method][key][0] is not None]
        return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (None, None, 0)

    lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r" & Closed & \multicolumn{2}{c}{V1$\rightarrow$V2/V3} & \multicolumn{2}{c}{V2/V3$\rightarrow$V1} & "
             r"\multicolumn{2}{c}{Leave-one-type-out (mean)} \\",
             r"Method & Acc. & Acc. & AUROC & Acc. & AUROC & Acc. & AUROC \\", r"\midrule"]
    for method in METHODS:
        if method not in main.get("closed", {}):
            continue
        cells = [_pct(main["closed"][method]["known_accuracy"])]
        for p in ("v1_to_v2", "v2_to_v1"):
            r = main.get(p, {}).get(method)
            cells += [_pct(r["known_accuracy"]), _pct(r["avail_energy_auroc"])] if r else ["--", "--"]
        cells += [_pct(loo_mean(method, "known_accuracy")), _pct(loo_mean(method, "avail_energy_auroc"))]
        lines.append(f"{LABEL[method]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (folder / "tab_main.tex").write_text("\n".join(lines))

    xrec = [p for p in ("xrec_s1_to_s2", "xrec_s2_to_s1") if p in main]
    if xrec:
        heads = {"xrec_s1_to_s2": r"Flight 1$\rightarrow$2", "xrec_s2_to_s1": r"Flight 2$\rightarrow$1"}
        lines = [r"\begin{tabular}{l" + "ccc" * len(xrec) + "}", r"\toprule",
                 " & " + " & ".join(r"\multicolumn{3}{c}{" + heads[p] + "}" for p in xrec) + r" \\",
                 "Method" + " & Acc. & AUROC & FPR@95" * len(xrec) + r" \\", r"\midrule"]
        for method in METHODS:
            cells = []
            for p in xrec:
                r = main[p].get(method)
                cells += ([_pct(r["known_accuracy"]), _pct(r["avail_energy_auroc"]),
                           _pct(r["avail_energy_fpr_at_95_tpr"])] if r else ["--"] * 3)
            if any(c != "--" for c in cells):
                lines.append(f"{LABEL[method]} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (folder / "tab_xrec.tex").write_text("\n".join(lines))

    open_protocols = [p for p in main if p != "closed"]
    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Method & MSP & Energy & Availability-normalised energy \\",
             r"\midrule"]
    for method in METHODS:
        cells = []
        for score in ("msp", "energy", "avail_energy"):
            vals = [main[p][method][f"{score}_auroc"][0] for p in open_protocols
                    if method in main[p] and main[p][method][f"{score}_auroc"][0] is not None]
            cells.append(_pct((float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (None, None, 0)))
        if any(c != "--" for c in cells):
            lines.append(f"{LABEL[method]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (folder / "tab_scores.tex").write_text("\n".join(lines))

    lines = [r"\begin{tabular}{lccccc}", r"\toprule", r"Method & 1 & 2 & 3 & 4 & 5 \\", r"\midrule"]
    for method, r in summary["closed_by_groups_present"].items():
        lines.append(f"{LABEL[method]} & " + " & ".join(_pct(r[b]) for b in map(str, range(1, 6))) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (folder / "tab_groups_present.tex").write_text("\n".join(lines))

    lines = [r"\begin{tabular}{llccc}", r"\toprule", r"Protocol & Method & Natural & i.i.d. removal & "
             r"Pattern removal \\", r"\midrule"]
    for protocol, rows in summary["perturbation"].items():
        for method, r in rows.items():
            lines.append(f"{protocol.replace('_', ' ')} & {LABEL[method]} & {_pct(r['natural'])} & "
                         f"{_pct(r['iid_drop'])} & {_pct(r['pattern_drop'])} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (folder / "tab_perturbation.tex").write_text("\n".join(lines))
    return sorted(p.name for p in folder.glob("*.tex"))


def write_figures(summary, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    written = []
    for protocol, rows in summary["subsets"].items():
        methods = list(rows)
        names = list(next(iter(rows.values())).keys())
        grid = np.array([[rows[m][n][0] if rows[m][n][0] is not None else np.nan for n in names] for m in methods])
        fig, ax = plt.subplots(figsize=(14, 0.5 * len(methods) + 2))
        im = ax.imshow(100 * grid, aspect="auto", cmap="viridis", vmin=0, vmax=100)
        ax.set_yticks(range(len(methods)), [LABEL[m] for m in methods])
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
        fig.colorbar(im, ax=ax, label="known accuracy (%)")
        ax.set_title(f"Sensor groups kept at test time ({protocol})")
        fig.tight_layout()
        path = folder / f"fig_subsets_{protocol}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path.name)
    r = summary["closed_by_groups_present"]
    if r:
        fig, ax = plt.subplots(figsize=(6, 4))
        for method in ("single_camera", "late", "concat", "lmmuavc", "dronefuse"):
            if method in r:
                ax.plot(range(1, 6), [100 * r[method][b][0] if r[method][b][0] is not None else np.nan
                                      for b in map(str, range(1, 6))], marker="o", label=LABEL[method])
        ax.set_xlabel("sensor groups present in the window")
        ax.set_ylabel("known accuracy (%)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / "fig_groups_present.png", dpi=200)
        plt.close(fig)
        written.append("fig_groups_present.png")
    return written


def availability_table(data, point_threshold=1):
    """Fraction of covered windows in which each sensor group is present, per site (for the paper)."""
    gm = group_presence(presence(data.counts, point_threshold))
    out = {}
    for site in ("V1", "V2_V3"):
        sel = data.version == site
        out[site] = {g: round(float(gm[sel, j].mean()), 3) for j, g in enumerate(GROUP_NAMES)} if sel.any() else {}
        out[site]["windows"] = int(sel.sum())
    return out
