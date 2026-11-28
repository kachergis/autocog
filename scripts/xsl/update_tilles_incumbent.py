"""Copy the corrected tilles refit into data/xsl/incumbents.json.

`scripts/xsl/export_tilles_reference.R` writes the entry (refit parameters,
SSE, v05's per-item ModelPerf, and `rlib` / `xslmodels_commit`: tilles runs
in xsl-bakeoff's fits/v05/tilles/Rlib, XSLmodels b44b9fa, not the pinned
0.4.0 build). This replaces the old-model entry, leaves every other entry
byte-for-byte as it was, and records the change in manifest.json.

    Rscript scripts/xsl/export_tilles_reference.R
    .venv/bin/python scripts/xsl/update_tilles_incumbent.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "xsl"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    ref = json.loads((DATA / "reference_tilles.json").read_text())
    path = DATA / "incumbents.json"
    incumbents = json.loads(path.read_text())
    entry = ref["incumbent"]
    assert list(entry)[:7] == list(incumbents["tilles"])[:7], "entry layout changed"
    old = incumbents["tilles"]
    incumbents["tilles"] = entry  # dict order is kept, so the file's model order is unchanged
    path.write_text(json.dumps(incumbents, separators=(",", ":")) + "\n")

    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["incumbents.json"] = md5(path)
    manifest["files"]["reference_tilles.json"] = md5(DATA / "reference_tilles.json")
    manifest["tilles"] = {
        "note": "tilles is the corrected model, refit by xsl-bakeoff; it runs in its own library",
        "xslmodels_commit": ref["xslmodels_commit"],
        "rlib": entry["rlib"],
        "group_fits_md5": ref["group_fits_md5"],
        "replaced": {"params": old["params"], "sse_v05": old["sse_v05"]},
    }
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    print(f"tilles: {old['params']} (SSE {old['sse_v05']:.4f}) -> {entry['params']} (SSE {entry['sse_v05']:.4f})")


if __name__ == "__main__":
    main()
