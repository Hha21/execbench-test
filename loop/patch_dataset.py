"""Local dataset fixes that let NVIDIA's harness load every problem on our test bench (never sent to the portal).

FlashInfer-Bench definitions leave the metadata field hf_id empty, which the harness's Definition model rejects
(NonEmptyString), so `sol-execbench` cannot load them. We fill it with a placeholder in our local copy of the data.
Idempotent; run after cloning or updating SOL-ExecBench.

  python3 loop/patch_dataset.py
"""

import glob
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "SOL-ExecBench" / "data" / "benchmark"

n = 0
for f in sorted(glob.glob(str(DATA / "*" / "*" / "definition.json"))):
    d = json.loads(Path(f).read_text())
    if not str(d.get("hf_id", "x")).strip():
        d["hf_id"] = "unknown"
        Path(f).write_text(json.dumps(d, indent=2) + "\n")
        n += 1
print(f"filled hf_id in {n} definition(s)")
