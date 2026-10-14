"""XSL theory generator: propose -> fit -> critique, with an accept gate.

The XSL counterpart of `src.theory_generator.TheoryGenerator`, with the same
loop contract:

  * the proposer always builds on the running-best ACCEPTED candidate;
  * a candidate is ACCEPTED iff its BIC is strictly lower than the running
    best (iteration 0's first fitted candidate auto-accepts);
  * a critic's "continue" ends the loop only if the candidate was accepted;
  * the loop returns the running-best accepted candidate.

What differs is the evidence: candidates are fitted to the whole human corpus
(`src.xsl.fitting`) instead of being simulated on observations, so the loss
is BIC on the group likelihood.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.xsl.corpus import Condition
from src.xsl.fitting import FitBudget, fit_theory, total_choices
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
    ):
        self.llm_client = llm_client
        self.conditions = list(conditions)
        self.budget = budget
        self.max_attempts = max_attempts
        self.max_seconds_per_pass = max_seconds_per_pass
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
        guide: str,
        leaderboard: Sequence[Entry],
        max_iters: int = 5,
        workspace: Path | None = None,
    ) -> GeneratorResult:
        history: list[Iteration] = []
        best: Iteration | None = None

        for crit in range(max_iters):
            system, user = render_generation(
                conditions=self.conditions, leaderboard=leaderboard, guide=guide,
                best=best, history=history, n_choices=self.n_choices,
            )
            theory, error = self._propose(system, user, workspace, crit)
            if theory is None:
                history.append(Iteration(candidate=None, fit=None, accepted=False, error=error))
                if best is not None:
                    break
                continue

            fit = fit_theory(theory, self.conditions, budget=self.budget)
            accepted = best is None or fit.bic < best.fit.bic
            it = Iteration(candidate=theory, fit=fit, accepted=accepted)
            if accepted:
                best = it
            print(
                f"[xsl generator iter {crit}] {theory.name}: BIC {fit.bic:.1f} "
                f"({fit.n_free} params, {fit.seconds:.0f}s fit) -> "
                f"{'ACCEPTED' if accepted else 'REJECTED'} (best BIC {best.fit.bic:.1f})"
            )

            c_system, c_user = render_critique(
                conditions=self.conditions, leaderboard=leaderboard, candidate=it,
                history=history, n_choices=self.n_choices,
            )
            try:
                it.critique = parse_reply(
                    CritiqueVerdict, self._chat(c_system, c_user, workspace, f"critique_iter_{crit:02d}")
                )
            except Exception as e:  # noqa: BLE001 - an unparseable critique just means "keep going"
                print(f"[xsl generator iter {crit}] critique unparseable ({type(e).__name__}: {e})")
            history.append(it)
            if it.critique is not None and it.critique.verdict == "continue" and accepted:
                break

        if best is None:
            raise RuntimeError(f"xsl generator: no valid theory in {max_iters} iterations: {history[-1].error}")
        return GeneratorResult(best=best, history=history)
