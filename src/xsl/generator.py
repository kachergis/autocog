"""XSL theory generator: propose -> fit -> critique, with an accept gate.

The XSL counterpart of `src.theory_generator.TheoryGenerator`, with the same
loop contract:

  * the proposer always builds on the running-best ACCEPTED candidate;
  * by default a candidate is ACCEPTED iff its held-out NLL (k-fold CV over
    conditions, same split for every candidate) is strictly lower than the
    running best's (iteration 0's first valid candidate auto-accepts); a
    caller can pass its own gate (`propose_theory(accept=...)`);
  * a critic's "continue" ends the loop only if the candidate was accepted;
  * the loop returns the running-best accepted candidate.

What differs is the evidence: candidates are fitted to the whole human corpus
(`src.xsl.fitting`) instead of being simulated on observations. The loss is
cross-validated NLL rather than BIC: in the first runs BIC on the full-corpus
group likelihood admitted candidates that fit better and generalised worse.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.xsl.corpus import Condition
from src.xsl.fitting import FitBudget, cross_validate_theory, fit_theory, total_choices
from src.xsl.prompts import CritiqueVerdict, Entry, Iteration, render_critique, render_generation
from src.xsl.theory import XSLTheory


def parse_reply(cls, text: str):
    """Parse an LLM reply into Promptable `cls`.

    A bare JSON object is decoded directly; anything else (a ```json fence,
    prose around it) goes through the shared `Promptable.from_llm_text`
    extractor, which lives in `src.llm` and so needs the LLM SDKs installed.
    """
    try:
        return cls.model_validate(json.loads(text))
    except json.JSONDecodeError:
        return cls.from_llm_text(text)


def unpin_fitted_names(theory: XSLTheory, leaderboard: Sequence[Entry]) -> tuple[XSLTheory, list[str]]:
    """Free every parameter `theory` pins to a constant whose name a leaderboard
    theory fits, giving it that theory's range (best-ranked first).

    Leaderboard theories are fitted to the whole corpus, and the generator sees
    their fitted values. A candidate that hard-codes them (xsl_10-12 pinned six
    of xsl_09's) carries information from every cross-validation fold's
    held-out conditions into its CV score, and understates its k. Constants
    with new names stay pinned.
    """
    ranges: dict[str, str] = {}
    for e in sorted(leaderboard, key=lambda e: e.score):
        for name, spec in e.theory.parameters.items():
            if name in e.theory.free_parameters and name not in ranges:
                ranges[name] = spec
    unpinned = [n for n in theory.fixed_parameters if n in ranges]
    if not unpinned:
        return theory, []
    params = {**theory.parameters, **{n: ranges[n] for n in unpinned}}
    return XSLTheory(**{**theory.model_dump(), "parameters": params}), unpinned


def warm_start(theory: XSLTheory, sources: Sequence[dict[str, float]]) -> dict[str, float]:
    """Starting values for `theory`'s fit: each free parameter (and the reserved
    `lapse` / `ability_sd`) from the first source fit that has a parameter of
    that name. Candidates are usually edits of the running best or of the top
    leaderboard theory, so shared names usually mean shared roles; anything
    unmatched starts at its range midpoint."""
    names = [*theory.free_parameters, "lapse", "ability_sd"]
    out: dict[str, float] = {}
    for name in names:
        for src in sources:
            if name in src:
                out[name] = float(src[name])
                break
    return out


def lower_heldout_nll(it: Iteration, best: Iteration | None) -> bool:
    """The default accept gate: strictly lower held-out NLL than the running best."""
    return best is None or it.score < best.score


@dataclass
class GeneratorResult:
    best: Iteration
    history: list[Iteration] = field(default_factory=list)


class XSLTheoryGenerator:
    def __init__(
        self,
        *,
        llm_client: Any,
        conditions: Sequence[Condition],
        budget: FitBudget = FitBudget(),
        max_attempts: int = 3,
        max_seconds_per_pass: float = 2.0,
        cv_folds: int = 5,
        cv_workers: int = 1,
    ):
        self.llm_client = llm_client
        self.conditions = list(conditions)
        self.budget = budget
        self.max_attempts = max_attempts
        self.max_seconds_per_pass = max_seconds_per_pass
        self.cv_folds = cv_folds
        self.cv_workers = cv_workers
        self.n_choices = total_choices(self.conditions)

    # --- llm ---------------------------------------------------------------------

    def _chat(self, system: str, user: str, workspace: Path | None, log_label: str) -> str:
        result = self.llm_client.chat(
            messages=[{"role": "user", "content": user}], system=system, response_schema=None
        )
        if workspace is not None:
            prompts = workspace / "prompts"
            prompts.mkdir(parents=True, exist_ok=True)
            (prompts / f"{log_label}.md").write_text(
                f"# {log_label}\n\n## System Prompt\n\n{system}\n\n## User Prompt\n\n{user}\n\n"
                f"## Response\n\n{result.text}\n\n## Usage\n\n```json\n"
                f"{json.dumps(result.usage, indent=2)}\n```\n"
            )
        return result.text

    def _propose(self, system: str, user: str, workspace: Path | None, crit: int) -> tuple[XSLTheory | None, str | None]:
        """Ask for a theory until one parses, compiles and passes `XSLTheory.check`."""
        error = None
        for attempt in range(self.max_attempts):
            text = self._chat(system, user, workspace, f"generation_iter_{crit:02d}_attempt_{attempt:02d}")
            try:
                theory = parse_reply(XSLTheory, text)
                theory.check(self.conditions, max_seconds=self.max_seconds_per_pass)
                return theory, None
            except Exception as e:  # noqa: BLE001 - any failure of LLM-written code is a rejection
                error = f"{type(e).__name__}: {e}"
                print(f"[xsl generator iter {crit} attempt {attempt}] rejected ({error})")
                user = (
                    f"{user}\n\n## YOUR PREVIOUS ATTEMPT FAILED\n{error}\n"
                    "Fix the problem and return the full JSON object again."
                )
        return None, error

    # --- loop ------------------------------------------------------------------

    def propose_theory(
        self,
        *,
        guide: str | Callable[[], str],
        leaderboard: Sequence[Entry],
        max_iters: int = 5,
        workspace: Path | None = None,
        start: Entry | None = None,
        accept: Callable[[Iteration, Iteration | None], bool] = lower_heldout_nll,
    ) -> GeneratorResult:
        """Run the propose -> fit -> critique loop.

        With `start` (a fitted, cross-validated theory, e.g. the one a debate
        arbiter wants revised) the loop begins with it as the running best, so
        a candidate is accepted only if it beats it; if none does, the result's
        best is `start` itself.

        `accept(candidate, running_best)` is the accept gate, called on every
        fitted, cross-validated candidate (`running_best` is None only before
        anything is accepted). `guide` may be a callable, re-read before each
        proposal, so a caller's gate can report its verdicts back to the
        proposer.
        """
        history: list[Iteration] = []
        best: Iteration | None = None
        if start is not None:
            best = Iteration(candidate=start.theory, fit=start.fit, accepted=True, cv=start.cv)

        for crit in range(max_iters):
            system, user = render_generation(
                conditions=self.conditions, leaderboard=leaderboard,
                guide=guide() if callable(guide) else guide,
                best=best, history=history, n_choices=self.n_choices,
                objective=self.budget.objective,
            )
            theory, error = self._propose(system, user, workspace, crit)
            if theory is not None:
                theory, unpinned = unpin_fitted_names(theory, leaderboard)
                if unpinned:
                    print(f"[xsl generator iter {crit}] refitting pinned parameters that leaderboard "
                          f"theories fit: {unpinned}")
            if theory is None:
                history.append(Iteration(candidate=None, fit=None, accepted=False, error=error))
                if best is not None:
                    break
                continue

            # warm-start from the running best, then the leaderboard (best first):
            # single cold starts left several 8-10 parameter theories hundreds of
            # NLL from their optimum
            sources = ([best.fit.params] if best is not None else []) + [
                e.fit.params for e in sorted(leaderboard, key=lambda e: e.score)]
            x0 = warm_start(theory, sources)
            fit = fit_theory(theory, self.conditions, budget=self.budget, x0=x0)
            cv = cross_validate_theory(theory, self.conditions, k=self.cv_folds,
                                       budget=self.budget, workers=self.cv_workers, x0=fit.params)
            it = Iteration(candidate=theory, fit=fit, accepted=False, cv=cv)
            it.accepted = accept(it, best)
            if it.accepted:
                best = it
            print(
                f"[xsl generator iter {crit}] {theory.name}: held-out NLL {it.score:.1f}, "
                f"BIC {fit.bic:.1f} ({fit.n_free} params) -> "
                f"{'ACCEPTED' if it.accepted else 'REJECTED'} (best held-out NLL {best.score:.1f})"
            )

            c_system, c_user = render_critique(
                conditions=self.conditions, leaderboard=leaderboard, candidate=it,
                history=history, n_choices=self.n_choices,
                objective=self.budget.objective,
            )
            try:
                it.critique = parse_reply(
                    CritiqueVerdict, self._chat(c_system, c_user, workspace, f"critique_iter_{crit:02d}")
                )
            except Exception as e:  # noqa: BLE001 - an unparseable critique just means "keep going"
                print(f"[xsl generator iter {crit}] critique unparseable ({type(e).__name__}: {e})")
            history.append(it)
            if it.critique is not None and it.critique.verdict == "continue" and it.accepted:
                break

        if best is None:
            raise RuntimeError(f"xsl generator: no valid theory in {max_iters} iterations: {history[-1].error}")
        return GeneratorResult(best=best, history=history)
