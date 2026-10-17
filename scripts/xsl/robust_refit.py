"""Refit selected leaderboard theories with a larger, multi-start budget.

The loop's fits use one differential-evolution run with a small budget, which
proved unreliable for 8-14 parameter theories (e.g. xsl_05, which contains
xsl_04 as a special case, fitted ~610 BIC worse). For each `--labels` theory
of the `--source` run this:

  1. applies any `--bounds` overrides (e.g. widening a parameter that fitted
     at its bound),
  2. refits it to the full corpus with `--restarts` runs of a larger budget,
     the first warm-started from its existing fit (so it cannot end worse,
     unless bounds or `--lapse_max` changed),
  3. cross-validates it on the same folds, every fold warm-started from the
     new full-corpus fit,

and writes a leaderboard (`main_xsl_offline` format) plus `comparison.md`
(before vs after) to `--out`, saving after every theory so a run can resume.
With `--lapse_max 0.1` it is the guessing-rate sensitivity check.

    python scripts/xsl/robust_refit.py --source results/xsl/binomial_e70c10c \\
        --out results/xsl/binomial_e70c10c_robust --labels inc_fazly xsl_09 \\
        --bounds "xsl_09:s=[0.01, 4]" "xsl_09:kc=[0, 40]"
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main_xsl_offline as runner  # noqa: E402
from src.xsl.corpus import load_corpus  # noqa: E402
from src.xsl.fitting import LAPSE, OBJECTIVES, FitBudget, cross_validate_theory, fit_theory  # noqa: E402
from src.xsl.prompts import Entry  # noqa: E402
from src.xsl.theory import XSLTheory  # noqa: E402


def parse_bounds(specs: list[str]) -> dict[str, dict[str, str]]:
    """["xsl_09:s=[0.01, 4]", ...] -> {"xsl_09": {"s": "[0.01, 4]"}}."""
    out: dict[str, dict[str, str]] = {}
    for spec in specs:
        label, _, assignment = spec.partition(":")
        name, _, interval = assignment.partition("=")
        if not (label and name and interval):
            raise ValueError(f"--bounds expects LABEL:PARAM=[lo, hi], got {spec!r}")
        out.setdefault(label.strip(), {})[name.strip()] = interval.strip()
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True, help="run directory holding leaderboard.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--objective", choices=OBJECTIVES, default="subject")
    p.add_argument("--source_objective", choices=OBJECTIVES, default=None,
                   help="objective the --source run was scored with (default: --objective); its fits "
                        "only seed the warm start")
    p.add_argument("--bounds", nargs="*", default=[], help='overrides, e.g. "xsl_09:s=[0.01, 4]"')
    p.add_argument("--popsize", type=int, default=12)
    p.add_argument("--maxiter", type=int, default=60)
    p.add_argument("--restarts", type=int, default=3, help="DE runs for the full-corpus fit")
    p.add_argument("--fold_restarts", type=int, default=2, help="DE runs for each CV fold")
    p.add_argument("--workers", type=int, default=4, help="processes scoring each DE population")
    p.add_argument("--lapse_max", type=float, default=FitBudget.lapse_max)
    p.add_argument("--cv_folds", type=int, default=5)
    return p.parse_args(argv)


def write_comparison(out: Path, rows: list[dict], before: str, after: str) -> None:
    (out / "comparison.json").write_text(json.dumps(rows, indent=2))
    note = "" if before == after else (
        f"\n\n\"before\" is the {before} objective and \"after\" the {after} objective, so their NLLs are not "
        "comparable; compare theories within a column.\n")
    lines = [
        "| label | k | full NLL before | full NLL after | held-out NLL before | held-out NLL after "
        "| held-out wSSE after | lapse after | at bound after | full-fit minutes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['k']} | {r['nll_before']:.1f} | {r['nll_after']:.1f} "
            f"| {r['cv_before']:.1f} | {r['cv_after']:.1f} | {r['cv_wsse_after']:.4f} "
            f"| {r['lapse_after']:.3f} | {', '.join(r['at_bound_after']) or '—'} | {r['minutes']:.0f} |"
        )
    (out / "comparison.md").write_text("\n".join(lines) + "\n" + note)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    conditions = load_corpus()
    overrides = parse_bounds(args.bounds)
    source_objective = args.source_objective or args.objective
    source = {e.label: e for e in runner.load(args.source, conditions, source_objective)}
    missing = [label for label in args.labels if label not in source]
    if missing:
        raise SystemExit(f"not in {args.source}/leaderboard.json: {missing}")

    budget = FitBudget(popsize=args.popsize, maxiter=args.maxiter, restarts=args.restarts,
                       workers=args.workers, lapse_max=args.lapse_max, objective=args.objective)
    fold_budget = replace(budget, restarts=args.fold_restarts)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    entries = runner.load(out, conditions, args.objective)
    done = {e.label for e in entries}
    rows = json.loads((out / "comparison.json").read_text()) if (out / "comparison.json").exists() else []

    for label in args.labels:
        if label in done:
            print(f"[skip] {label}: already refitted in {out}")
            continue
        old = source[label]
        theory = old.theory
        if label in overrides:
            theory = XSLTheory(**{**theory.model_dump(), "parameters": {**theory.parameters, **overrides[label]}})
        x0 = dict(old.fit.params)
        x0[LAPSE] = min(x0.get(LAPSE, args.lapse_max), args.lapse_max)

        fit = fit_theory(theory, conditions, budget=budget, x0=x0)
        print(f"[fit] {label}: NLL {old.fit.nll:.1f} -> {fit.nll:.1f} ({fit.seconds / 60:.0f} min, "
              f"{fit.n_evals} evals) params {fit.params}")
        cv = cross_validate_theory(theory, conditions, k=args.cv_folds, budget=fold_budget, x0=fit.params)
        print(f"[cv] {label}: held-out NLL {old.cv.heldout_nll:.1f} -> {cv.heldout_nll:.1f}, "
              f"wSSE {cv.heldout_weighted_sse:.4f}")

        entries.append(Entry(label=label, theory=theory, fit=fit, cv=cv))
        runner.save(out, entries)
        rows.append({
            "label": label, "k": fit.n_free,
            "nll_before": old.fit.nll, "nll_after": fit.nll,
            "cv_before": old.cv.heldout_nll, "cv_after": cv.heldout_nll,
            "cv_wsse_after": cv.heldout_weighted_sse,
            "lapse_after": fit.params.get(LAPSE, float("nan")),
            "at_bound_after": fit.at_bound,
            "bounds_override": overrides.get(label, {}),
            "minutes": fit.seconds / 60,
        })
        write_comparison(out, rows, source_objective, args.objective)


if __name__ == "__main__":
    main()
