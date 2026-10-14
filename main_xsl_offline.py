"""Offline AutoCog for cross-situational word learning — Phase 1, mode B.

No new data is collected and there is no debate yet: each round, an LLM
theory generator proposes a new XSL theory (prose + `learn` function), every
candidate is fitted to the whole XSLmodels corpus (group NLL, scored by BIC),
and a critic steers the next revision. The best candidate of each round joins
the leaderboard, which starts from the fitted seed theories.

    uv run python main_xsl_offline.py --llm_provider openrouter \
        --llm_model anthropic/claude-opus-5.5 --n_rounds 1 --max_iters 3

API keys are read from `.env` at the repo root (see README §3), e.g.
OPENROUTER_API_KEY for --llm_provider openrouter.

Outputs (resumable; re-running continues the rounds) under --out:
    leaderboard.json / leaderboard.md   every theory with its corpus fit
    theories/<label>.yaml                each theory in seed-YAML format
    round_NN/prompts/*.md                every generator / critic exchange
    cv.json / cv.md                      with --cv: 5-fold CV over conditions

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
from src.xsl.fitting import FitBudget, FitResult, cross_validate, evaluate, fit_theory, total_choices  # noqa: E402
from src.xsl.generator import XSLTheoryGenerator  # noqa: E402
from src.xsl.prompts import Entry, leaderboard_table  # noqa: E402
from src.xsl.theory import XSLTheory  # noqa: E402

SEEDS = (ROOT / "theories/xsl/baseline.yaml", ROOT / "theories/xsl/kachergis.yaml")
DEFAULT_GUIDE = (
    "Propose a theory of cross-situational word learning that achieves a lower BIC than "
    "every theory on the leaderboard. Study which conditions the best theory fits worst "
    "and what those designs have in common, and target that mechanism."
)


# --- persistence ---------------------------------------------------------------

def _entry_json(e: Entry) -> dict:
    f = e.fit
    return {
        "label": e.label,
        "theory": e.theory.model_dump(),
        "fit": {
            "params": f.params, "nll": f.nll, "bic": f.bic, "aic": f.aic,
            "weighted_sse": f.evaluation.weighted_sse, "n_free": f.n_free,
            "n_choices": f.n_choices, "n_evals": f.n_evals, "seconds": f.seconds,
            "at_bound": f.at_bound,
            "per_condition_nll": f.evaluation.per_condition_nll,
            "per_condition_sse": f.evaluation.per_condition_sse,
        },
    }


def _entry_from_json(d: dict, conditions: list[Condition]) -> Entry:
    theory = XSLTheory(**d["theory"])
    f = d["fit"]
    return Entry(
        label=d["label"],
        theory=theory,
        fit=FitResult(
            params=f["params"],
            evaluation=evaluate(theory.learner(), f["params"], conditions),
            n_free=f["n_free"], n_choices=f["n_choices"], n_evals=f["n_evals"],
            seconds=f["seconds"], at_bound=f["at_bound"],
        ),
    )


def save(run_dir: Path, entries: list[Entry]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "leaderboard.json").write_text(json.dumps([_entry_json(e) for e in entries], indent=2))
    (run_dir / "leaderboard.md").write_text(f"# XSL leaderboard\n\n{leaderboard_table(entries)}\n")
    (run_dir / "theories").mkdir(exist_ok=True)
    for e in entries:
        e.theory.to_yaml(run_dir / "theories" / f"{e.label}.yaml")


def load(run_dir: Path, conditions: list[Condition]) -> list[Entry]:
    path = run_dir / "leaderboard.json"
    if not path.exists():
        return []
    return [_entry_from_json(d, conditions) for d in json.loads(path.read_text())]


# --- run -----------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "results/xsl/mode_b")
    p.add_argument("--n_rounds", type=int, default=1, help="generator rounds to add on top of the saved run")
    p.add_argument("--max_iters", type=int, default=5, help="propose/fit/critique iterations per round")
    p.add_argument("--llm_provider", default=None, help="override configs/default.yaml's provider")
    p.add_argument("--llm_model", default=None)
    p.add_argument("--reasoning_effort", default="high",
                   help="OpenRouter reasoning effort; reasoning tokens dominate cost ('max' can use 100k+ per call)")
    p.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    p.add_argument("--guide", default=DEFAULT_GUIDE)
    p.add_argument("--popsize", type=int, default=FitBudget.popsize)
    p.add_argument("--maxiter", type=int, default=FitBudget.maxiter)
    p.add_argument("--cv", action="store_true", help="cross-validate every leaderboard theory at the end")
    p.add_argument("--cv_folds", type=int, default=5)
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
    return client


def seed_leaderboard(conditions: list[Condition], budget: FitBudget) -> list[Entry]:
    entries = []
    for path in SEEDS:
        theory = XSLTheory.from_yaml(path)
        fit = fit_theory(theory, conditions, budget=budget)
        print(f"[seed] {path.stem}: BIC {fit.bic:.1f} ({fit.seconds:.0f}s) params {fit.params}")
        entries.append(Entry(label=path.stem, theory=theory, fit=fit))
    return entries


def run_cv(run_dir: Path, entries: list[Entry], conditions: list[Condition], k: int, budget: FitBudget) -> None:
    rows = []
    for e in entries:
        cv = cross_validate(e.theory.learner(), e.theory.free_parameters, conditions,
                            fixed=e.theory.fixed_parameters, k=k, budget=budget)
        rows.append({"label": e.label, "name": e.theory.name, "heldout_nll": cv.heldout_nll,
                     "heldout_weighted_sse": cv.heldout_weighted_sse, "fold_params": cv.fold_params})
        print(f"[cv] {e.label}: held-out NLL {cv.heldout_nll:.1f}, wSSE {cv.heldout_weighted_sse:.4f}")
    rows.sort(key=lambda r: r["heldout_nll"])
    (run_dir / "cv.json").write_text(json.dumps(rows, indent=2))
    lines = [f"# {k}-fold CV over conditions\n", "| rank | label | name | held-out NLL | held-out wSSE |",
             "|---|---|---|---|---|"]
    lines += [f"| {i} | {r['label']} | {r['name']} | {r['heldout_nll']:.1f} | {r['heldout_weighted_sse']:.4f} |"
              for i, r in enumerate(rows, 1)]
    (run_dir / "cv.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    budget = FitBudget(popsize=args.popsize, maxiter=args.maxiter)
    conditions = load_corpus()
    run_dir: Path = args.out
    entries = load(run_dir, conditions)
    if not entries:
        entries = seed_leaderboard(conditions, budget)
        save(run_dir, entries)
    print(f"{len(conditions)} conditions, {total_choices(conditions)} choices; leaderboard:\n"
          f"{leaderboard_table(entries)}")

    if args.n_rounds > 0:
        generator = XSLTheoryGenerator(llm_client=make_llm_client(args), conditions=conditions, budget=budget)
        n_generated = sum(e.label.startswith("xsl_") for e in entries)
        for r in range(n_generated, n_generated + args.n_rounds):
            label = f"xsl_{r + 1:02d}"
            result = generator.propose_theory(
                guide=args.guide, leaderboard=entries, max_iters=args.max_iters,
                workspace=run_dir / f"round_{r:02d}",
            )
            entries.append(Entry(label=label, theory=result.best.candidate, fit=result.best.fit))
            save(run_dir, entries)
            print(f"[round {r}] {label} = {result.best.candidate.name}: BIC {result.best.fit.bic:.1f}\n"
                  f"{leaderboard_table(entries)}")

    if args.cv:
        run_cv(run_dir, entries, conditions, args.cv_folds, budget)


if __name__ == "__main__":
    main()
