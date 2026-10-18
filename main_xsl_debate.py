"""Offline AutoCog for cross-situational word learning — Phase 2 debate.

Two theories from an existing leaderboard debate over the corpus (mode B: all
conditions visible). Each round:

  1. each side's advocate proposes a corpus query — conditions plus a
     statistic of their per-item accuracies — on which its theory should match
     people better; a query counts only if the two theories' predictions differ
     by >= 2 human standard errors (src/xsl/debate.py);
  2. an arbiter reads both outcomes, the ledger and both theories' held-out
     NLL, and picks a theory to revise ('new_model' or 'new_theory');
  3. the theory generator revises it, starting from the target as the bar to
     beat; a revision replaces the target in its slot (and joins the
     leaderboard as a new xsl_NN) only if its cross-validated held-out NLL is
     lower.

    uv run python main_xsl_debate.py --a xsl_12 --b inc_fazly --rounds 3 \
        --llm_provider openrouter --llm_model anthropic/claude-sonnet-5.5

Outputs under --out (resumable): debate.json (every round: queries,
outcomes, verdict, revision), debate.md, leaderboard.json/.md (incl. any
revised theories), round_NN/ prompt logs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main_xsl_offline as runner  # noqa: E402
from src.xsl.corpus import Condition, load_corpus  # noqa: E402
from src.xsl.debate import (  # noqa: E402
    CorpusQuery, DebateVerdict, QueryOutcome, Side, evaluate_query, ledger_block,
    render_arbitration, render_query,
)
from src.xsl.fitting import FitBudget, population_accuracy  # noqa: E402
from src.xsl.generator import XSLTheoryGenerator, parse_reply  # noqa: E402
from src.xsl.prompts import OUTPUT_RULES, Entry  # noqa: E402

DEFAULT_SOURCES = (ROOT / "results/xsl/mode_b_subject", ROOT / "results/xsl/mode_b_subject_unpinned")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", default="xsl_12", help="label of theory 1")
    p.add_argument("--b", default="inc_fazly", help="label of theory 2")
    p.add_argument("--sources", type=Path, nargs="+", default=list(DEFAULT_SOURCES),
                   help="run directories to take theories from (later ones override earlier labels)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--rounds", type=int, default=3, help="debate rounds to add on top of the saved state")
    p.add_argument("--max_iters", type=int, default=3, help="revision iterations per round")
    p.add_argument("--query_attempts", type=int, default=3)
    p.add_argument("--llm_provider", default=None)
    p.add_argument("--llm_model", default=None)
    p.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    p.add_argument("--reasoning_effort", default="high")
    p.add_argument("--max_output_tokens", type=int, default=32_000)
    p.add_argument("--popsize", type=int, default=FitBudget.popsize)
    p.add_argument("--maxiter", type=int, default=FitBudget.maxiter)
    p.add_argument("--restarts", type=int, default=FitBudget.restarts)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--cv_workers", type=int, default=3)
    args = p.parse_args(argv)
    if args.out is None:
        args.out = ROOT / f"results/xsl/debate_{args.a}_vs_{args.b}"
    return args


def build_leaderboard(sources: list[Path], conditions: list[Condition]) -> list[Entry]:
    by_label: dict[str, Entry] = {}
    for src in sources:
        for e in runner.load(src, conditions, "subject"):
            by_label[e.label] = e
    return list(by_label.values())


def predict(entry: Entry, conditions: list[Condition]) -> dict:
    learn = entry.theory.learner()
    return {c.label: population_accuracy(learn, entry.fit.params, c) for c in conditions}


def side(entry: Entry) -> Side:
    return Side(label=entry.label, theory=entry.theory, fit=entry.fit, cv=entry.cv)


def propose_query(gen: XSLTheoryGenerator, advocate: Side, opponent: Side, conditions, predicted, ledger,
                  workspace: Path, attempts: int) -> tuple[QueryOutcome | None, list[str]]:
    failures: list[str] = []
    for k in range(attempts):
        system, user = render_query(advocate=advocate, opponent=opponent, conditions=conditions,
                                    predicted=predicted, ledger=ledger, failures=failures,
                                    output_rules=OUTPUT_RULES)
        text = gen._chat(system, user, workspace, f"query_{advocate.label}_attempt_{k:02d}")
        try:
            query = parse_reply(CorpusQuery, text)
            outcome = evaluate_query(query, advocate.label, conditions, predicted)
        except Exception as e:  # noqa: BLE001 - any failure of an LLM-written query is a rejection
            failures.append(f"invalid query ({type(e).__name__}: {e})")
            continue
        if outcome.separable:
            return outcome, failures
        failures.append(f"not separable: {outcome.summary()}")
    return None, failures


def arbitrate(gen: XSLTheoryGenerator, sides, conditions, predicted, round_outcomes, ledger,
              workspace: Path) -> DebateVerdict:
    system, user = render_arbitration(sides=sides, conditions=conditions, predicted=predicted,
                                      round_outcomes=round_outcomes, ledger=ledger, output_rules=OUTPUT_RULES)
    error = None
    for k in range(3):
        try:
            return parse_reply(DebateVerdict, gen._chat(system, user, workspace, f"arbiter_attempt_{k:02d}"))
        except Exception as e:  # noqa: BLE001
            error = e
    raise RuntimeError(f"arbiter gave no valid verdict: {error}")


def revision_guide(target: Side, verdict: DebateVerdict) -> str:
    keep = ("Keep its central claim; change its mechanism or implementation."
            if verdict.verdict == "new_model" else
            "Replace it with a different account that fixes these failures.")
    return (f"The debate arbiter asks you to revise {target.label} ({verdict.verdict}). {keep}\n"
            f"Arbiter's interpretation: {verdict.interpretation}\n"
            f"Recommendation: {verdict.recommendation}\n"
            f"The revision must reach a lower held-out NLL than the CURRENT BEST CANDIDATE, which is "
            f"{target.label} itself.")


def debate_round(gen: XSLTheoryGenerator, state: dict, entries: list[Entry], by_label: dict[str, Entry],
                 conditions: list[Condition], ledger: list[QueryOutcome], out: Path, *,
                 query_attempts: int, max_iters: int) -> dict:
    """One round: two queries, a verdict, a revision attempt. Updates `state`
    (slots, rounds), `entries`, `by_label` and `ledger` in place."""
    r_idx = len(state["rounds"])
    workspace = out / f"round_{r_idx:02d}"
    a, b = (side(by_label[label]) for label in state["slots"])
    predicted = {s.label: predict(by_label[s.label], conditions) for s in (a, b)}
    print(f"[debate round {r_idx}] {a.label} vs {b.label}")

    record = {"slots": [a.label, b.label], "outcomes": [], "failures": {}}
    for advocate, opponent in ((a, b), (b, a)):
        outcome, failures = propose_query(gen, advocate, opponent, conditions, predicted, ledger,
                                          workspace, query_attempts)
        record["failures"][advocate.label] = failures
        if outcome is not None:
            record["outcomes"].append(outcome.to_json())
            ledger.append(outcome)
            print(f"[debate round {r_idx}] {outcome.summary()}")
        else:
            print(f"[debate round {r_idx}] {advocate.label}: no separable query")
    round_outcomes = [o for o in ledger[len(ledger) - len(record["outcomes"]):]]

    verdict = arbitrate(gen, [a, b], conditions, predicted, round_outcomes, ledger, workspace)
    target = (a, b)[verdict.target - 1]
    record["verdict"] = verdict.model_dump()
    record["target_label"] = target.label
    print(f"[debate round {r_idx}] verdict: revise {target.label} ({verdict.verdict})")

    result = gen.propose_theory(guide=revision_guide(target, verdict), leaderboard=entries,
                                max_iters=max_iters, workspace=workspace / "revision",
                                start=by_label[target.label])
    record["revised_label"] = None
    record["revised_score"] = None
    if result.best.candidate is not target.theory:
        label = f"xsl_{runner.next_generated_number(entries):02d}"
        best = result.best
        new = Entry(label=label, theory=best.candidate, fit=best.fit, cv=best.cv)
        entries.append(new)
        by_label[label] = new
        state["slots"][verdict.target - 1] = label
        record["revised_label"], record["revised_score"] = label, best.score
        print(f"[debate round {r_idx}] {label} replaces {target.label}: held-out NLL "
              f"{target.cv.heldout_nll:.1f} -> {best.score:.1f}")
    else:
        print(f"[debate round {r_idx}] no revision beat {target.label}")
    state["rounds"].append(record)
    return record


def write_summary(out: Path, state: dict) -> None:
    lines = [f"# Debate: {state['initial'][0]} vs {state['initial'][1]}\n"]
    for i, r in enumerate(state["rounds"]):
        lines.append(f"## Round {i + 1}: {r['slots'][0]} vs {r['slots'][1]}\n")
        for o in r["outcomes"]:
            lines.append(f"- {QueryOutcome.from_json(o).summary()}")
        for label, fails in r["failures"].items():
            if fails and not any(o["proposer"] == label for o in r["outcomes"]):
                lines.append(f"- {label}: no separable query in {len(fails)} attempts")
        v = r["verdict"]
        lines.append(f"\n**Verdict:** revise theory {v['target']} ({r['target_label']}), {v['verdict']}. "
                     f"{v['recommendation']}\n")
        lines.append(f"**Revision:** {r['revised_label'] + ' accepted' if r['revised_label'] else 'no candidate beat ' + r['target_label']}"
                     f"{' (held-out NLL ' + format(r['revised_score'], '.1f') + ')' if r['revised_label'] else ''}\n")
    (out / "debate.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    conditions = load_corpus()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    entries = runner.load(out, conditions, "subject") or build_leaderboard(args.sources, conditions)
    by_label = {e.label: e for e in entries}
    for label in (args.a, args.b):
        if label not in by_label:
            raise SystemExit(f"{label} is not on the leaderboard")
    runner.save(out, entries)

    state_path = out / "debate.json"
    state = (json.loads(state_path.read_text()) if state_path.exists()
             else {"initial": [args.a, args.b], "slots": [args.a, args.b], "rounds": []})
    ledger = [QueryOutcome.from_json(o) for r in state["rounds"] for o in r["outcomes"]]

    budget = FitBudget(popsize=args.popsize, maxiter=args.maxiter, objective="subject", restarts=args.restarts)
    gen = XSLTheoryGenerator(llm_client=runner.make_llm_client(args), conditions=conditions, budget=budget,
                             cv_folds=args.cv_folds, cv_workers=args.cv_workers)

    for _ in range(args.rounds):
        debate_round(gen, state, entries, by_label, conditions, ledger, out,
                     query_attempts=args.query_attempts, max_iters=args.max_iters)
        runner.save(out, entries)
        state_path.write_text(json.dumps(state, indent=2))
        write_summary(out, state)
    print(f"\nLedger:\n{ledger_block(ledger)}")


if __name__ == "__main__":
    main()
