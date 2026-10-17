"""Offline AutoCog for cross-situational word learning — Phase 1, mode B.

No new data is collected and there is no debate yet: each round, an LLM
theory generator proposes a new XSL theory (prose + `learn` function), every
candidate is fitted to the XSLmodels corpus by likelihood (by default the
per-subject random-ability likelihood, which models individual differences;
--objective binomial / multinomial for item-level counts) and scored by
5-fold cross-validation over conditions (held-out NLL), and a critic steers
the next revision. The best candidate of each round joins the leaderboard,
which starts from the fitted seed theories.

    uv run python main_xsl_offline.py --llm_provider openrouter \
        --llm_model anthropic/claude-opus-5.5 --n_rounds 1 --max_iters 3

API keys are read from `.env` at the repo root (see README §3), e.g.
OPENROUTER_API_KEY for --llm_provider openrouter.

Outputs (resumable; re-running continues the rounds) under --out:
    leaderboard.json / leaderboard.md   every theory: full-corpus fit + CV
    cv.json / cv.md                      the CV results alone, ranked
    theories/<label>.yaml                each theory in seed-YAML format
    round_NN/prompts/*.md                every generator / critic exchange

See docs/xsl_offline_proposal.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xsl.corpus import Condition, load_corpus  # noqa: E402
from src.xsl.fitting import (  # noqa: E402
    OBJECTIVES, CVResult, FitBudget, FitResult, cross_validate_theory, evaluate, fit_theory, total_choices,
)
from src.xsl.generator import XSLTheoryGenerator  # noqa: E402
from src.xsl.prompts import Entry, leaderboard_table  # noqa: E402
from src.xsl.theory import XSLTheory  # noqa: E402

SEEDS = (ROOT / "theories/xsl/baseline.yaml", ROOT / "theories/xsl/kachergis.yaml")
INCUMBENTS = tuple(sorted((ROOT / "theories/xsl/incumbents").glob("*.yaml")))
DEFAULT_GUIDE = (
    "Propose a theory of cross-situational word learning that achieves a lower held-out NLL "
    "than every theory on the leaderboard. Study which conditions the best theory predicts worst "
    "and what those designs have in common, and target that mechanism."
)


# --- persistence ---------------------------------------------------------------

def _entry_json(e: Entry) -> dict:
    f = e.fit
    return {
        "label": e.label,
        "objective": f.evaluation.objective,
        "theory": e.theory.model_dump(),
        "fit": {
            "params": f.params, "nll": f.nll, "bic": f.bic, "aic": f.aic,
            "weighted_sse": f.evaluation.weighted_sse, "n_free": f.n_free,
            "n_choices": f.n_choices, "n_evals": f.n_evals, "seconds": f.seconds,
            "at_bound": f.at_bound,
            "per_condition_nll": f.evaluation.per_condition_nll,
            "per_condition_sse": f.evaluation.per_condition_sse,
            "other_nll": f.evaluation.other_nll,
        },
        "cv": None if e.cv is None else _cv_json(e.cv),
    }


def _cv_json(cv: CVResult) -> dict:
    return {"heldout_nll": cv.heldout_nll, "heldout_weighted_sse": cv.heldout_weighted_sse,
            "per_condition_nll": cv.per_condition_nll, "fold_params": cv.fold_params}


def _cv_from_json(d: dict) -> CVResult:
    return CVResult(heldout_nll=d["heldout_nll"], heldout_weighted_sse=d["heldout_weighted_sse"],
                    per_condition_nll=list(d["per_condition_nll"]), fold_params=d["fold_params"])


def _entry_from_json(d: dict, conditions: list[Condition], objective: str) -> Entry:
    # entries written before the objective was recorded were all multinomial
    saved = d.get("objective", "multinomial")
    if saved != objective:
        raise ValueError(
            f"{d['label']} was scored with the {saved} objective but this run uses {objective}; "
            f"pass --objective {saved}, or --import_from it into a fresh --out to refit"
        )
    theory = XSLTheory(**d["theory"])
    f = d["fit"]
    return Entry(
        label=d["label"],
        theory=theory,
        fit=FitResult(
            params=f["params"],
            evaluation=evaluate(theory.learner(), f["params"], conditions, objective),
            n_free=f["n_free"], n_choices=f["n_choices"], n_evals=f["n_evals"],
            seconds=f["seconds"], at_bound=f["at_bound"],
        ),
        cv=_cv_from_json(d["cv"]) if d.get("cv") else None,
    )


def next_generated_number(entries: list[Entry]) -> int:
    """1 + the highest NN among `xsl_NN` labels. A run seeded with only some
    generated theories (e.g. xsl_03-09) must not reuse a number: counting
    them instead once labelled a new theory xsl_08 next to the old one."""
    numbers = [int(e.label[4:]) for e in entries if e.label.startswith("xsl_") and e.label[4:].isdigit()]
    return max(numbers, default=0) + 1


def save(run_dir: Path, entries: list[Entry]) -> None:
    labels = [e.label for e in entries]
    duplicates = sorted({x for x in labels if labels.count(x) > 1})
    if duplicates:
        raise ValueError(f"duplicate leaderboard labels {duplicates}; refusing to overwrite their theory files")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "leaderboard.json").write_text(json.dumps([_entry_json(e) for e in entries], indent=2))
    (run_dir / "leaderboard.md").write_text(f"# XSL leaderboard\n\n{leaderboard_table(entries)}\n")
    (run_dir / "theories").mkdir(exist_ok=True)
    for e in entries:
        e.theory.to_yaml(run_dir / "theories" / f"{e.label}.yaml")
    rows = sorted((e for e in entries if e.cv is not None), key=lambda e: e.score)
    (run_dir / "cv.json").write_text(json.dumps(
        [{"label": e.label, "name": e.theory.name, **_cv_json(e.cv)} for e in rows], indent=2))
    lines = ["# 5-fold CV over conditions\n", "| rank | label | name | held-out NLL | held-out wSSE |",
             "|---|---|---|---|---|"]
    lines += [f"| {i} | {e.label} | {e.theory.name} | {e.cv.heldout_nll:.1f} | {e.cv.heldout_weighted_sse:.4f} |"
              for i, e in enumerate(rows, 1)]
    (run_dir / "cv.md").write_text("\n".join(lines) + "\n")


def load(run_dir: Path, conditions: list[Condition], objective: str = "binomial") -> list[Entry]:
    path = run_dir / "leaderboard.json"
    if not path.exists():
        return []
    entries = [_entry_from_json(d, conditions, objective) for d in json.loads(path.read_text())]
    # runs from before CV was stored per entry kept it only in cv.json (same
    # folds, seed and budget as now), with per-condition NLL keyed by label
    legacy = run_dir / "cv.json"
    if legacy.exists():
        by_label = {r["label"]: r for r in json.loads(legacy.read_text())}
        for e in entries:
            r = by_label.get(e.label)
            if e.cv is None and r is not None and isinstance(r.get("per_condition_nll"), dict):
                e.cv = _cv_from_json({**r, "per_condition_nll": [r["per_condition_nll"][c.label] for c in conditions]})
    return entries


# --- run -----------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "results/xsl/mode_b")
    p.add_argument("--n_rounds", type=int, default=1, help="generator rounds to add on top of the saved run")
    p.add_argument("--max_iters", type=int, default=5, help="propose/fit/critique iterations per round")
    p.add_argument("--llm_provider", default=None, help="override configs/default.yaml's provider")
    p.add_argument("--llm_model", default=None)
    p.add_argument("--max_output_tokens", type=int, default=32_000,
                   help="cap on output (incl. reasoning) tokens per LLM call; bounds the cost of a runaway reply")
    p.add_argument("--reasoning_effort", default="high",
                   help="OpenRouter reasoning effort; reasoning tokens dominate cost ('max' can use 100k+ per call)")
    p.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    p.add_argument("--guide", default=DEFAULT_GUIDE)
    p.add_argument("--popsize", type=int, default=FitBudget.popsize)
    p.add_argument("--maxiter", type=int, default=FitBudget.maxiter)
    p.add_argument("--restarts", type=int, default=FitBudget.restarts,
                   help="DE runs per fit; the first is warm-started from the closest existing fit")
    p.add_argument("--objective", choices=OBJECTIVES, default="subject",
                   help="likelihood to fit and score by: subject (per-person responses with random "
                        "abilities; default), binomial (item-level accuracy), multinomial")
    p.add_argument("--lapse_max", type=float, default=FitBudget.lapse_max,
                   help="upper bound of the fitted guessing rate (lower it for sensitivity checks)")
    p.add_argument("--incumbents", action="store_true",
                   help="add the xsl-bakeoff incumbent ports (theories/xsl/incumbents/, labelled inc_*)")
    p.add_argument("--import_from", type=Path, default=None,
                   help="leaderboard.json of another run: refit and cross-validate its theories here")
    p.add_argument("--import_labels", nargs="*", default=None, help="with --import_from: only these labels")
    p.add_argument("--cv", action="store_true", help="recompute the CV of every leaderboard theory at the end")
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--cv_workers", type=int, default=3, help="processes for fitting CV folds in parallel")
    return p.parse_args(argv)


# Closed-model vendors on OpenRouter. src.llm's routing was chosen for
# open-weight models and matches none of their endpoints: they report
# quantization "unknown" (so the fp8-only filter fails), and some reject
# `temperature` (e.g. every Claude Sonnet 5.5 endpoint), which
# `require_parameters` turns into a hard routing failure. Neither filter is
# needed here: the XSL loop parses free text, never `response_format`, so
# OpenRouter may simply drop a parameter a backend does not support.
CLOSED_MODEL_VENDORS = ("anthropic/", "google/", "openai/", "x-ai/")


def make_llm_client(args: argparse.Namespace):
    # imported here so the module imports without any LLM SDK or API key
    from src.config import load_config
    from src.llm import make_client

    llm = load_config(args.config).llm
    if args.llm_provider:
        llm = llm.model_copy(update={"provider": args.llm_provider})
    if args.llm_model:
        llm = llm.model_copy(update={"model": args.llm_model})
    if llm.provider == "openrouter":
        llm = llm.model_copy(update={"reasoning_effort": args.reasoning_effort})
    client = make_client(llm)
    if llm.provider == "openrouter" and llm.model.startswith(CLOSED_MODEL_VENDORS):
        client.extra_body["provider"].pop("quantizations", None)
        client.extra_body["provider"]["require_parameters"] = False
    client.max_tokens = min(client.max_tokens, args.max_output_tokens)
    return client


def add_theories(entries: list[Entry], theories: list[tuple[str, XSLTheory]], conditions: list[Condition],
                 budget: FitBudget) -> None:
    """Fit and append each (label, theory) not already on the leaderboard (CV comes from ensure_cv)."""
    have = {e.label for e in entries}
    for label, theory in theories:
        if label in have:
            continue
        have.add(label)  # the seeds also appear in imported leaderboards
        fit = fit_theory(theory, conditions, budget=budget)
        print(f"[add] {label}: BIC {fit.bic:.1f} ({fit.seconds:.0f}s) params {fit.params}")
        entries.append(Entry(label=label, theory=theory, fit=fit))


def theories_to_add(args: argparse.Namespace) -> list[tuple[str, XSLTheory]]:
    out = [(p.stem, XSLTheory.from_yaml(p)) for p in SEEDS]
    if args.incumbents:
        out += [(f"inc_{p.stem}", XSLTheory.from_yaml(p)) for p in INCUMBENTS]
    if args.import_from is not None:
        for d in json.loads(Path(args.import_from).read_text()):
            if args.import_labels is None or d["label"] in args.import_labels:
                out.append((d["label"], XSLTheory(**d["theory"])))
    return out


def ensure_cv(entries: list[Entry], conditions: list[Condition], *, k: int, budget: FitBudget,
              workers: int, recompute: bool = False, on_done=None) -> None:
    """Cross-validate every entry that has no CV yet (or all, with `recompute`);
    `on_done()` runs after each one (e.g. to save progress)."""
    for e in entries:
        if e.cv is None or recompute:
            e.cv = cross_validate_theory(e.theory, conditions, k=k, budget=budget, workers=workers)
            print(f"[cv] {e.label}: held-out NLL {e.cv.heldout_nll:.1f}, wSSE {e.cv.heldout_weighted_sse:.4f}")
            if on_done is not None:
                on_done()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    budget = FitBudget(popsize=args.popsize, maxiter=args.maxiter, lapse_max=args.lapse_max,
                       objective=args.objective, restarts=args.restarts)
    conditions = load_corpus()
    run_dir: Path = args.out
    entries = load(run_dir, conditions, args.objective)
    add_theories(entries, theories_to_add(args), conditions, budget)
    save(run_dir, entries)
    ensure_cv(entries, conditions, k=args.cv_folds, budget=budget, workers=args.cv_workers, recompute=args.cv,
              on_done=lambda: save(run_dir, entries))
    print(f"{len(conditions)} conditions, {total_choices(conditions)} choices; leaderboard:\n"
          f"{leaderboard_table(entries)}")

    if args.n_rounds > 0:
        generator = XSLTheoryGenerator(llm_client=make_llm_client(args), conditions=conditions, budget=budget,
                                       cv_folds=args.cv_folds, cv_workers=args.cv_workers)
        first = next_generated_number(entries)
        for n in range(first, first + args.n_rounds):
            label, r = f"xsl_{n:02d}", n - 1
            result = generator.propose_theory(
                guide=args.guide, leaderboard=entries, max_iters=args.max_iters,
                workspace=run_dir / f"round_{r:02d}",
            )
            best = result.best
            entries.append(Entry(label=label, theory=best.candidate, fit=best.fit, cv=best.cv))
            save(run_dir, entries)
            print(f"[round {r}] {label} = {best.candidate.name}: held-out NLL {best.score:.1f}, "
                  f"BIC {best.fit.bic:.1f}\n{leaderboard_table(entries)}")


if __name__ == "__main__":
    main()
