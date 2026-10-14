"""The incumbent panel: previous best-fitting XSL models, run through R.

Incumbents are the xsl-bakeoff v05 group fits (exported to
`data/xsl/incumbents.json`). Their predictions on new designs come from the
pinned XSLmodels build itself — `scripts/xsl/run_incumbents.R` via a
subprocess — rather than from Python ports, so they are exactly the published
models. R is only touched when `predict` is called; importing this module
needs nothing beyond the standard library and NumPy.

Incumbent predictions are a *reference* for prospective designs (where would a
new theory disagree with the field's best models?), never a loss target.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.xsl.corpus import DATA_DIR, Condition

REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_SCRIPT = REPO_ROOT / "scripts" / "xsl" / "run_incumbents.R"

# Interim panel, from v05 *group* fits until v05 cross-validation exists (see
# docs/xsl_offline_proposal.md §4). kachergis is left out because
# uncfam_gamma contains it (gamma = 1) and fits better.
INTERIM_PANEL: tuple[str, ...] = (
    "fazly",
    "uncfam_gamma",
    "uncfam_attention",
    "tilles",
    "Bayesian_decay",
)


@dataclass(frozen=True)
class Incumbent:
    name: str
    constructor: str
    params: dict[str, float]
    fixed: dict[str, Any] = field(default_factory=dict)
    stochastic: bool = False
    sse_v05: float | None = None
    n_params: int = 0

    def request(self) -> dict[str, Any]:
        return {"name": self.name, "constructor": self.constructor,
                "params": self.params, "fixed": self.fixed}


def load_incumbents(data_dir: Path | None = None) -> dict[str, Incumbent]:
    raw = json.loads(((data_dir or DATA_DIR) / "incumbents.json").read_text())
    return {
        name: Incumbent(
            name=name,
            constructor=d["constructor"],
            params=d["params"],
            fixed=d["fixed"],
            stochastic=d["stochastic"],
            sse_v05=d["sse_v05"],
            n_params=d["n_params"],
        )
        for name, d in raw.items()
    }


def load_panel(names: Sequence[str] = INTERIM_PANEL, data_dir: Path | None = None) -> list[Incumbent]:
    available = load_incumbents(data_dir)
    missing = [n for n in names if n not in available]
    if missing:
        raise KeyError(f"no exported v05 fit for {missing}")
    return [available[n] for n in names]


def default_rlib() -> Path:
    """The pinned XSLmodels library: `$XSL_RLIB`, else xsl-bakeoff next to this repo."""
    env = os.environ.get("XSL_RLIB")
    return Path(env) if env else REPO_ROOT.parent / "xsl-bakeoff" / "fits" / "v05" / "Rlib"


def _numpy_scalar(o: Any) -> Any:
    """Designs built with NumPy (e.g. by a generated design function) carry
    numpy scalars as labels; write them as plain JSON numbers."""
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def _design_json(design_id: str, c: Condition) -> dict[str, Any]:
    test = None
    if c.test is not None:
        test = {"words": [[w] for w in c.test.words],
                "objects": [list(os) for os in c.test.objects]}
        if c.test.correct is not None:
            test["correct"] = [[o] for o in c.test.correct]
    return {
        "id": design_id,
        "train": {"words": [list(t) for t in c.train_words],
                  "objects": [list(t) for t in c.train_objects]},
        "test": test,
    }


def predict(
    incumbents: Sequence[Incumbent],
    designs: Sequence[Condition],
    *,
    n_sim: int = 500,
    seed: int = 1,
    rlib: Path | None = None,
    rscript: str = "Rscript",
    timeout: float | None = 600,
) -> dict[str, list[np.ndarray]]:
    """Per-item predicted accuracy of each incumbent on each design, from R.

    Returns `{incumbent name: [perf for designs[0], perf for designs[1], ...]}`.
    `n_sim` and `seed` only matter for stochastic incumbents.
    """
    rlib = rlib or default_rlib()
    if not (rlib / "XSLmodels").is_dir():
        raise FileNotFoundError(f"pinned XSLmodels library not found at {rlib}; set XSL_RLIB")
    ids = [str(i) for i in range(len(designs))]
    request = {
        "models": [inc.request() for inc in incumbents],
        "designs": [_design_json(i, c) for i, c in zip(ids, designs)],
        "n_sim": n_sim,
        "seed": seed,
    }
    with tempfile.TemporaryDirectory() as tmp:
        req_path, resp_path = Path(tmp) / "request.json", Path(tmp) / "response.json"
        req_path.write_text(json.dumps(request, default=_numpy_scalar))
        proc = subprocess.run(
            [rscript, str(BRIDGE_SCRIPT), str(req_path), str(resp_path), str(rlib)],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"run_incumbents.R failed ({proc.returncode}):\n{proc.stderr}")
        predictions = json.loads(resp_path.read_text())["predictions"]
    return {
        inc.name: [np.array(predictions[inc.name][i], dtype=float) for i in ids]
        for inc in incumbents
    }
