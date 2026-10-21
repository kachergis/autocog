"""Offline AutoCog for cross-situational word learning — Phase 3 prospective designs.

Designs a new experiment on which two fitted theories make clearly different
predictions (src/xsl/design.py). For each design slot:

  1. the designer (an LLM) writes a design generator with a few knobs and a
     pre-stated contrast over per-item test accuracy, seeing both theories,
     where they differ on the existing corpus and what the corpus debate
     found;
  2. the search builds the design at every knob setting (several random
     orders each) and simulates experiments with --n_subj people per arm from
     each theory's random-ability fit, including parameter uncertainty from
     the CV folds; the best setting's power is re-estimated independently;
  3. if power >= .8 (the pre-registered midpoint rule picks the generating
     theory at least 80% of the time under both), the design is frozen into
     the queue with every leaderboard theory's predictions; otherwise the
     designer sees the result and revises (up to --max_iters).

    uv run python main_xsl_design.py --a xsl_14 --b inc_fazly --n_designs 3 \
        --llm_provider openrouter --llm_model anthropic/claude-sonnet-5.5

Outputs under --out (resumable; --n_designs adds designs to the queue):
queue.json / queue.md (accepted designs, frozen predictions), attempts.json
(every proposal and its search), design_NN/prompts/ (LLM logs).
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
from main_xsl_debate import predict, side  # noqa: E402
from src.xsl.corpus import load_corpus  # noqa: E402
from src.xsl.debate import QueryOutcome, _condition_table, ledger_block  # noqa: E402
from src.xsl.design import (  # noqa: E402
    FINAL_SIMS, MAX_COMBOS, PLANNED_N, SEARCH_SIMS, SEEDS_PER_COMBO, DesignProposal, Generator,
    SearchResult, chat_logged, freeze, render_design, search,
)
from src.xsl.generator import parse_reply  # noqa: E402
from src.xsl.prompts import OUTPUT_RULES  # noqa: E402

DEFAULT_LEADERBOARD = ROOT / "results/xsl/debate_xsl_12_vs_inc_fazly_gated"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", default="xsl_14", help="label of theory 1")
    p.add_argument("--b", default="inc_fazly", help="label of theory 2")
    p.add_argument("--leaderboard", type=Path, default=DEFAULT_LEADERBOARD,
                   help="run directory whose leaderboard.json holds the fitted theories")
    p.add_argument("--debate", type=Path, default=DEFAULT_LEADERBOARD,
                   help="debate run directory whose debate.json ledger the designer sees (optional)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--n_designs", type=int, default=3, help="design slots to add to the queue")
    p.add_argument("--max_iters", type=int, default=4, help="designer revisions per slot")
    p.add_argument("--proposal_attempts", type=int, default=3, help="retries for an unparseable proposal")
    p.add_argument("--n_subj", type=int, default=PLANNED_N, help="planned people per arm")
    p.add_argument("--max_combos", type=int, default=MAX_COMBOS)
    p.add_argument("--seeds", type=int, default=SEEDS_PER_COMBO)
    p.add_argument("--search_sims", type=int, default=SEARCH_SIMS)
    p.add_argument("--final_sims", type=int, default=FINAL_SIMS)
    p.add_argument("--llm_provider", default=None)
    p.add_argument("--llm_model", default=None)
    p.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    p.add_argument("--reasoning_effort", default="high")
    p.add_argument("--max_output_tokens", type=int, default=32_000)
    args = p.parse_args(argv)
    if args.out is None:
        args.out = ROOT / f"results/xsl/designs_{args.a}_vs_{args.b}"
    return args


def debate_ledger(path: Path) -> str:
    state = path / "debate.json"
    if not state.exists():
        return "(no debate run)"
    rounds = json.loads(state.read_text())["rounds"]
    return ledger_block([QueryOutcome.from_json(o) for r in rounds for o in r["outcomes"]])


def propose(llm, system: str, user: str, workspace: Path, label: str, attempts: int) -> DesignProposal | None:
    error = None
    for k in range(attempts):
        prompt = user if error is None else (
            f"{user}\n\n## YOUR PREVIOUS REPLY FAILED\n{error}\nFix it and return the full JSON object again.")
        text = chat_logged(llm, system, prompt, workspace, f"{label}_attempt_{k:02d}")
        try:
            return parse_reply(DesignProposal, text)
        except Exception as e:  # noqa: BLE001 - any failure of LLM-written code is a rejection
            error = f"{type(e).__name__}: {e}"
            print(f"[xsl design] {label} attempt {k} rejected ({error})")
    return None


def attempt_record(slot: int, it: int, result: SearchResult | None, error: str | None = None) -> dict:
    rec = {"slot": slot, "iter": it, "error": error}
    if result is not None:
        rec.update({
            "proposal": result.proposal.model_dump(by_alias=False),
            "accepted": result.accepted,
            "summary": result.summary(),
            "search_power": result.search_power,
            "power": result.power.to_json() if result.power else None,
            "design": result.design.to_json() if result.design else None,
            "trials": [{"knobs": t.knobs, "seed": t.seed, "power": t.power, "error": t.error}
                       for t in result.trials],
            "seconds": result.seconds,
        })
    return rec


def design_slot(llm, slot: int, a, b, gens, entries, *, corpus_table: str, ledger: str, queue: list,
                attempts: list, out: Path, args) -> dict | None:
    """Propose and revise one design until it is accepted or `max_iters` runs out."""
    workspace = out / f"design_{slot:02d}"
    history: list[str] = []
    for it in range(args.max_iters):
        system, user = render_design(a=a, b=b, corpus_table=corpus_table, debate_ledger=ledger, queue=queue,
                                     history=history, n_subj=args.n_subj, output_rules=OUTPUT_RULES)
        proposal = propose(llm, system, user, workspace, f"design_iter_{it:02d}", args.proposal_attempts)
        if proposal is None:
            attempts.append(attempt_record(slot, it, None, "no valid proposal"))
            history.append(f"iteration {it}: no parseable proposal")
            continue
        result = search(proposal, *gens, n_subj=args.n_subj, max_combos=args.max_combos, seeds=args.seeds,
                        search_sims=args.search_sims, final_sims=args.final_sims, seed=slot * 100 + it)
        attempts.append(attempt_record(slot, it, result))
        print(f"[xsl design {slot} iter {it}] {result.summary()} -> "
              f"{'ACCEPTED' if result.accepted else 'not accepted'} ({result.seconds:.1f}s)")
        if result.accepted:
            return freeze(result, entries, pair=(a.label, b.label),
                          leaderboard_path=args.leaderboard / "leaderboard.json", root=ROOT)
        history.append(f"iteration {it}: {result.summary()}")
    return None


def write_queue_md(out: Path, queue: list, attempts: list) -> None:
    lines = [f"# Prospective design queue\n",
             f"{len(queue)} accepted design(s); {len(attempts)} proposals tried.\n"]
    for q in queue:
        pw = q["power"]
        a, b = q["pair"]
        lines += [
            f"## {q['id']}\n",
            f"Frozen {q['frozen_at']}, sha256 `{q['sha256'][:16]}…`, commit `{(q['git_commit'] or '?')[:8]}`.\n",
            f"**Rationale.** {q['proposal']['rationale']}\n",
            f"**Prediction.** {q['proposal']['prediction']}\n",
            f"**Design.** " + "; ".join(
                f"arm `{arm['name']}`: {arm['n_words']} words × {arm['n_objects']} objects, "
                f"{arm['n_trials']} trials, {len(arm['test'])} test items" for arm in q["design"]["arms"])
            + f" (knobs {q['design']['knobs']}, seed {q['design']['seed']}); "
              f"{q['planned_n_per_arm']} people per arm.\n",
            f"**Decision rule.** Contrast above {q['decision_rule']['threshold']:+.4f} favours "
            f"`{q['decision_rule']['favours_above']}`. Predicted: {a} {pw['predicted'][a]:+.4f}, "
            f"{b} {pw['predicted'][b]:+.4f}; power {pw['power']:.3f} "
            f"(P(correct | {a}) {pw['hit_rate'][a]:.3f}, P(correct | {b}) {pw['hit_rate'][b]:.3f}; "
            f"{pw['n_sims']} simulated experiments each).\n",
            "```python\n" + q["proposal"]["contrast_source"].strip() + "\n```\n",
            "| theory | predicted contrast | mean accuracy per arm |", "|---|---|---|",
        ]
        for label, p in q["predictions"].items():
            if "error" in p:
                lines.append(f"| {label} | error: {p['error'][:60]} | |")
            else:
                accs = ", ".join(f"{k} {sum(v) / len(v):.3f}" for k, v in p["items"].items())
                lines.append(f"| {label} | {p['contrast']:+.4f} | {accs} |")
        lines.append("")
    (out / "queue.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    conditions = load_corpus()
    entries = runner.load(args.leaderboard, conditions, "subject")
    by_label = {e.label: e for e in entries}
    for label in (args.a, args.b):
        if label not in by_label:
            raise SystemExit(f"{label} is not on the leaderboard in {args.leaderboard}")
    a, b = by_label[args.a], by_label[args.b]
    gens = (Generator.from_entry(a), Generator.from_entry(b))
    predicted = {e.label: predict(e, conditions) for e in (a, b)}
    corpus_table = _condition_table(conditions, [side(a), side(b)], predicted)
    ledger = debate_ledger(args.debate)

    queue_path, attempts_path = out / "queue.json", out / "attempts.json"
    queue = json.loads(queue_path.read_text()) if queue_path.exists() else []
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    llm = runner.make_llm_client(args)

    start = max([r["slot"] for r in attempts], default=-1) + 1
    for slot in range(start, start + args.n_designs):
        record = design_slot(llm, slot, a, b, gens, entries, corpus_table=corpus_table, ledger=ledger,
                             queue=queue, attempts=attempts, out=out, args=args)
        if record is not None:
            queue.append(record)
            print(f"[xsl design {slot}] queued {record['id']} (power {record['power']['power']:.3f})")
        else:
            print(f"[xsl design {slot}] no design reached the power target")
        queue_path.write_text(json.dumps(queue, indent=2))
        attempts_path.write_text(json.dumps(attempts, indent=2, default=str))
        write_queue_md(out, queue, attempts)


if __name__ == "__main__":
    main()
