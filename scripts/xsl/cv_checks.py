"""Three checks on the XSL leaderboard's scoring, without refitting the full corpus.

  1. wSSE: `evaluate` used to compare a median-ability person's accuracy with
     the human mean under the random-ability objective. Recompute full-fit
     and held-out wSSE for every leaderboard theory from its stored full and
     fold parameters with the population-mean prediction.
  2. Cold-start CV: the leaderboard's folds were warm-started from the
     full-corpus fit, which has seen the held-out conditions. Refit every
     fold of `--labels` from scratch (same folds, no `x0`).
  3. Grouped CV: the same, cold, with folds that keep each experiment family
     (`fitting.experiment_family`) whole.

`run` fits the (theory, split, fold) jobs in parallel, one process each,
saving every job to `--out/jobs/` so it can resume; `report` writes
`--out/comparison.md` from whatever jobs are done.

    .venv/bin/python scripts/xsl/cv_checks.py run --workers 12
    .venv/bin/python scripts/xsl/cv_checks.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xsl.corpus import load_corpus  # noqa: E402
from src.xsl.fitting import (  # noqa: E402
    ABILITY_SD, LAPSE, FitBudget, cv_folds, evaluate, experiment_family, fit_theory, n_choices_per_item,
)
from src.xsl.scoring import predict_accuracy, sse  # noqa: E402
from src.xsl.theory import XSLTheory  # noqa: E402

SOURCE = ROOT / "results/xsl/debate_xsl_12_vs_inc_fazly_gated"
OUT = ROOT / "results/xsl/cv_checks"
LABELS = ("xsl_14", "xsl_09", "kachergis", "inc_fazly")
SPLITS = ("random", "grouped")
OBJECTIVE = "subject"


def folds_for(split: str, conditions, k: int, seed: int) -> list[np.ndarray]:
    groups = [experiment_family(c.label) for c in conditions] if split == "grouped" else None
    return cv_folds(len(conditions), k, seed, groups)


def wsse(per_condition_sse, conditions) -> float:
    w = np.array([c.n_subj for c in conditions], dtype=float)
    return float((np.array(per_condition_sse) * w).sum() / w.sum())


# --- run -------------------------------------------------------------------------

def run_job(theory_dump: dict, split: str, fold: int, k: int, seed: int, budget: FitBudget) -> dict:
    conditions = load_corpus()
    held = folds_for(split, conditions, k, seed)[fold]
    held_set = set(held.tolist())
    theory = XSLTheory(**theory_dump)
    train = [c for i, c in enumerate(conditions) if i not in held_set]
    fit = fit_theory(theory, train, budget=budget)  # cold: no x0
    ev = evaluate(theory.learner(), fit.params, [conditions[i] for i in held], budget.objective)
    return {
        "split": split, "fold": fold, "held": held.tolist(), "params": fit.params,
        "train_nll": fit.nll, "heldout_nll": ev.nll,
        "per_condition_nll": ev.per_condition_nll, "per_condition_sse": ev.per_condition_sse,
        "n_evals": fit.n_evals, "seconds": fit.seconds, "at_bound": fit.at_bound,
        "budget": {"popsize": budget.popsize, "maxiter": budget.maxiter, "restarts": budget.restarts,
                   "seed": budget.seed, "objective": budget.objective},
    }


def run(args: argparse.Namespace) -> None:
    source = {d["label"]: d for d in json.loads((args.source / "leaderboard.json").read_text())}
    jobs_dir = args.out / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    budget = FitBudget(popsize=args.popsize, maxiter=args.maxiter, restarts=args.restarts, objective=OBJECTIVE)
    # slowest theories first, so the pool's tail is the cheap jobs
    labels = sorted(args.labels, key=lambda lab: -source[lab]["fit"]["n_free"])
    todo = [(lab, split, f) for lab in labels for split in args.splits for f in range(args.cv_folds)
            if not (jobs_dir / f"{lab}_{split}_{f}.json").exists()]
    print(f"{len(todo)} jobs to run on {args.workers} processes", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, source[lab]["theory"], split, f, args.cv_folds, args.seed, budget):
                   (lab, split, f) for lab, split, f in todo}
        for fut in as_completed(futures):
            lab, split, f = futures[fut]
            res = fut.result()
            (jobs_dir / f"{lab}_{split}_{f}.json").write_text(json.dumps({"label": lab, **res}, indent=2))
            print(f"[{(time.perf_counter() - t0) / 60:6.1f} min] {lab} {split} fold {f}: train NLL "
                  f"{res['train_nll']:.1f}, held-out NLL {res['heldout_nll']:.1f} "
                  f"({res['seconds'] / 60:.0f} min, {res['n_evals']} evals)", flush=True)


# --- report ----------------------------------------------------------------------

def median_person_sse(learner, params, c) -> float:
    """The old `evaluate` SSE: a median-ability person's accuracy vs the human mean."""
    m = learner(params, c.positional_trials, len(c.words), len(c.objects))
    lapse = params.get(LAPSE, 0.0)
    perf = (1 - lapse) * predict_accuracy(m, c, d=params.get("ch_dec")) + lapse / n_choices_per_item(c)
    return sse(perf, c)


def wsse_rows(source: list[dict], conditions, k: int, seed: int) -> list[dict]:
    folds = cv_folds(len(conditions), k, seed)
    rows = []
    for d in source:
        learner = XSLTheory(**d["theory"]).learner()
        full = evaluate(learner, d["fit"]["params"], conditions, OBJECTIVE)
        old_full = wsse([median_person_sse(learner, d["fit"]["params"], c) for c in conditions], conditions)
        per_nll, per_new, per_old = ([np.nan] * len(conditions) for _ in range(3))
        for held, params in zip(folds, d["cv"]["fold_params"]):
            ev = evaluate(learner, params, [conditions[i] for i in held], OBJECTIVE)
            for i, nll_i, sse_i in zip(held, ev.per_condition_nll, ev.per_condition_sse):
                per_nll[i], per_new[i] = nll_i, sse_i
                per_old[i] = median_person_sse(learner, params, conditions[i])
        rows.append({
            "label": d["label"], "k": d["fit"]["n_free"], "ability_sd": d["fit"]["params"].get(ABILITY_SD),
            "heldout_nll": d["cv"]["heldout_nll"],
            "full_before": d["fit"]["weighted_sse"], "full_after": full.weighted_sse,
            "cv_before": d["cv"]["heldout_weighted_sse"], "cv_after": wsse(per_new, conditions),
            # the stored values must be reproducible from the stored parameters
            "full_before_recomputed": old_full, "cv_before_recomputed": wsse(per_old, conditions),
            "heldout_nll_recomputed": float(sum(per_nll)),
        })
    return sorted(rows, key=lambda r: r["heldout_nll"])


def rank(rows: list[dict], key: str) -> dict[str, int]:
    return {r["label"]: i for i, r in enumerate(sorted(rows, key=lambda r: r[key]), 1)}


def load_jobs(out: Path, label: str, split: str, k: int) -> list[dict] | None:
    paths = [out / "jobs" / f"{label}_{split}_{f}.json" for f in range(k)]
    return [json.loads(p.read_text()) for p in paths] if all(p.exists() for p in paths) else None


def assemble(jobs: list[dict], conditions) -> dict:
    n = len(conditions)
    per_nll, per_sse = [np.nan] * n, [np.nan] * n
    for j in jobs:
        for i, a, b in zip(j["held"], j["per_condition_nll"], j["per_condition_sse"]):
            per_nll[i], per_sse[i] = a, b
    return {"heldout_nll": float(sum(per_nll)), "heldout_wsse": wsse(per_sse, conditions),
            "per_condition_nll": per_nll, "train_nll": [j["train_nll"] for j in jobs],
            "fold_heldout_nll": [j["heldout_nll"] for j in jobs], "fold_params": [j["params"] for j in jobs],
            "minutes": sum(j["seconds"] for j in jobs) / 60}


def report(args: argparse.Namespace) -> None:
    conditions = load_corpus()
    source = json.loads((args.source / "leaderboard.json").read_text())
    by_label = {d["label"]: d for d in source}
    k, seed = args.cv_folds, args.seed
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    L: list[str] = ["# XSL scoring checks\n",
                    f"Source leaderboard: `{args.source.relative_to(ROOT)}`. Written by "
                    "`scripts/xsl/cv_checks.py`; see `note.md` for the reading.\n"]

    # 1. wSSE
    rows = wsse_rows(source, conditions, k, seed)
    for r in rows:
        assert abs(r["full_before_recomputed"] - r["full_before"]) < 1e-9, r
        assert abs(r["cv_before_recomputed"] - r["cv_before"]) < 1e-9, r
        assert abs(r["heldout_nll_recomputed"] - r["heldout_nll"]) < 1e-6, r
    ranks = {key: rank(rows, key) for key in ("full_before", "full_after", "cv_before", "cv_after")}
    L += ["## 1. Weighted SSE: median-person vs population-mean prediction\n",
          "Recomputed from the stored full-fit and fold parameters (no refitting). \"before\" compares the "
          "accuracy of a median-ability person with the human mean; \"after\" compares the population-mean "
          "accuracy (`population_accuracy`) with it. The \"before\" columns and the held-out NLL were also "
          "recomputed from the stored parameters and match the stored values. NLL is unaffected. Rows are in "
          "leaderboard (held-out NLL) order; (n) is the rank on that column.\n",
          "| label | k | ability_sd | held-out NLL | full wSSE before | full wSSE after | held-out wSSE before "
          "| held-out wSSE after |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cells = " | ".join(f"{r[key]:.4f} ({ranks[key][r['label']]})"
                           for key in ("full_before", "full_after", "cv_before", "cv_after"))
        L.append(f"| {r['label']} | {r['k']} | {r['ability_sd']:.3f} | {r['heldout_nll']:.1f} | {cells} |")
    (out / "wsse.json").write_text(json.dumps(rows, indent=2))

    # 2 and 3. cold-start CV on the stored (random) folds and on grouped folds
    random_folds = cv_folds(len(conditions), k, seed)
    cv_out = {}
    for label in args.labels:
        d = by_label[label]
        learner = XSLTheory(**d["theory"]).learner()
        warm_train = []
        for held, params in zip(random_folds, d["cv"]["fold_params"]):
            held_set = set(held.tolist())
            train = [c for i, c in enumerate(conditions) if i not in held_set]
            warm_train.append(evaluate(learner, params, train, OBJECTIVE).nll)
        cv_out[label] = {"warm": {"heldout_nll": d["cv"]["heldout_nll"], "train_nll": warm_train,
                                  "per_condition_nll": d["cv"]["per_condition_nll"]}}
        for split in SPLITS:
            jobs = load_jobs(out, label, split, k)
            if jobs is not None:
                cv_out[label][split] = assemble(jobs, conditions)
    (out / "cv.json").write_text(json.dumps(cv_out, indent=2))

    def gap(a: dict, b: dict, key: str) -> str:
        return f"{b[key]['heldout_nll'] - a[key]['heldout_nll']:+.1f}" if key in a and key in b else "pending"

    top = cv_out[args.labels[0]]
    done = sorted((out / "jobs").glob("*.json"))
    b = json.loads(done[0].read_text())["budget"] if done else None
    budget_note = (f"popsize {b['popsize']}, maxiter {b['maxiter']}, {b['restarts']} restarts, no `x0`"
                   if b else "no `x0`")
    L += ["\n## 2. Cold-start CV on the leaderboard's folds\n",
          f"Same 5 folds (`cv_folds(47, 5, seed=0)`), each fold fitted from scratch ({budget_note}). \"warm\" is "
          "the stored leaderboard value, every fold warm-started from the full-corpus fit. Train NLL is summed "
          "over the 5 folds' training sets (each condition counted 4 times): a cold train NLL above the warm "
          "one means the cold optimiser stopped short of the warm solution, so its held-out NLL also reflects "
          "optimiser error, not only the removal of leakage.\n",
          "| label | k | held-out NLL warm (stored) | held-out NLL cold | cold - warm | train NLL warm | "
          "train NLL cold | cold - warm | held-out wSSE cold | max fold-parameter shift, cold vs warm "
          "(% of range) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for label in args.labels:
        c, d = cv_out[label], by_label[label]
        if "random" not in c:
            L.append(f"| {label} | {d['fit']['n_free']} | {c['warm']['heldout_nll']:.1f} | pending | | "
                     f"{sum(c['warm']['train_nll']):.1f} | | | | |")
            continue
        r = c["random"]
        theory = XSLTheory(**d["theory"])
        shifts = [abs(pc[n] - pw[n]) / (hi - lo) for pc, pw in zip(r["fold_params"], d["cv"]["fold_params"])
                  for n, (lo, hi) in theory.free_parameters.items()]
        L.append(f"| {label} | {d['fit']['n_free']} | {c['warm']['heldout_nll']:.1f} | {r['heldout_nll']:.1f} "
                 f"| {r['heldout_nll'] - c['warm']['heldout_nll']:+.1f} | {sum(c['warm']['train_nll']):.1f} "
                 f"| {sum(r['train_nll']):.1f} | {sum(r['train_nll']) - sum(c['warm']['train_nll']):+.1f} "
                 f"| {r['heldout_wsse']:.4f} | {100 * max(shifts):.0f} |")
    L += ["", f"Gap to `{args.labels[0]}` in held-out NLL (positive: worse than `{args.labels[0]}`):\n",
          "| label | warm (stored) | cold |", "|---|---|---|"]
    L += [f"| {lab} | {gap(top, cv_out[lab], 'warm')} | {gap(top, cv_out[lab], 'random')} |"
          for lab in args.labels[1:]]

    grouped_folds = folds_for("grouped", conditions, k, seed)
    L += ["\n## 3. Grouped CV: experiment families kept whole\n",
          "Folds from `cv_folds(47, 5, seed=0, groups=[experiment_family(c.label) ...])`, fitted cold with "
          "the same budget, so compare with the cold column of table 2. Held-out conditions per fold:\n"]
    L += [f"- fold {i}: " + ", ".join(conditions[j].label for j in f) for i, f in enumerate(grouped_folds)]
    L += ["", "| label | k | held-out NLL, random folds (cold) | held-out NLL, grouped folds (cold) | "
              "grouped - random | held-out wSSE grouped | held-out NLL per grouped fold (0-4) |",
          "|---|---|---|---|---|---|---|"]
    for label in args.labels:
        c = cv_out[label]
        rnd = f"{c['random']['heldout_nll']:.1f}" if "random" in c else "pending"
        if "grouped" not in c:
            L.append(f"| {label} | {by_label[label]['fit']['n_free']} | {rnd} | pending | | | |")
            continue
        g = c["grouped"]
        diff = f"{g['heldout_nll'] - c['random']['heldout_nll']:+.1f}" if "random" in c else ""
        L.append(f"| {label} | {by_label[label]['fit']['n_free']} | {rnd} | {g['heldout_nll']:.1f} | {diff} "
                 f"| {g['heldout_wsse']:.4f} | " + ", ".join(f"{x:.1f}" for x in g["fold_heldout_nll"]) + " |")
    L += ["", f"Gap to `{args.labels[0]}` in held-out NLL:\n",
          "| label | random folds (cold) | grouped folds (cold) |", "|---|---|---|"]
    L += [f"| {lab} | {gap(top, cv_out[lab], 'random')} | {gap(top, cv_out[lab], 'grouped')} |"
          for lab in args.labels[1:]]
    if all("grouped" in cv_out[lab] for lab in args.labels):
        fams = {}
        for i, c in enumerate(conditions):
            fam = experiment_family(c.label)
            fams.setdefault(fam if fam != c.label else "single-condition families", []).append(i)
        L += ["", "Held-out NLL under grouped folds, summed by family:\n",
              "| family | n conditions | " + " | ".join(args.labels) + " |",
              "|---|---|" + "---|" * len(args.labels)]
        for fam, idx in fams.items():
            L.append(f"| {fam} | {len(idx)} | " + " | ".join(
                f"{sum(cv_out[lab]['grouped']['per_condition_nll'][i] for i in idx):.1f}"
                for lab in args.labels) + " |")
    (out / "comparison.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("run", "report"))
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--labels", nargs="+", default=list(LABELS))
    p.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    p.add_argument("--popsize", type=int, default=FitBudget.popsize)
    p.add_argument("--maxiter", type=int, default=FitBudget.maxiter)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0, help="fold seed")
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args(argv)
    (run if args.command == "run" else report)(args)


if __name__ == "__main__":
    main()
