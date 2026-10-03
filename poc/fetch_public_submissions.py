"""Download the public B200 submissions for one SOL-ExecBench kernel, with their portal results.

Purpose: test whether our cheap sources (other-GPU timings, compiled-code features,
simulator) predict the *real* B200 ranking of real submissions. The code belongs to its
authors; it is only run locally to measure it and is never resubmitted. public_subs/ is
gitignored so it does not end up in our public repo.

Only unauthenticated GET requests to the portal's public API, one per second. Nothing is
executed: files are saved as downloaded.

  python3 poc/fetch_public_submissions.py [--kernel 038_flux_multi_head_rmsnorm_qk]

Output: poc/public_subs/<submission id>/<file> and poc/public_subs/index.csv
(id, username, submitted_at, latency_ms, sol_score, file, bytes).
"""

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://research.nvidia.com/benchmarks/sol-execbench/api"
HERE = Path(__file__).resolve().parent


def get(path, raw=False):
    req = urllib.request.Request(API + path, headers={"User-Agent": "solx-poc (research; contact via portal)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
        return (body, r.headers) if raw else json.loads(body)


def public_submissions(kernel_id, version):
    subs, offset = [], 0
    while True:
        q = urllib.parse.urlencode(dict(kernel_id=kernel_id, gpu_type="B200", evaluation_stack_version=version,
                                        status="COMPLETED", offset=offset, limit=50))
        page = get(f"/submissions?{q}")["data"]
        subs += page["submissions"]
        offset += 50
        time.sleep(1)
        if offset >= page.get("total", 0) or not page["submissions"]:
            return subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="038_flux_multi_head_rmsnorm_qk")
    ap.add_argument("--version", default="v1.1")
    ap.add_argument("--out", type=Path, default=HERE / "public_subs")
    args = ap.parse_args()

    kernel = next(k for k in get("/kernels")["data"]["kernels"] if k["name"] == args.kernel)
    subs = [s for s in public_submissions(kernel["id"], args.version)
            if s["submission_mode"] == "release" and s["is_correct"] and not s["is_disqualified"]]
    print(f"{args.kernel}: {len(subs)} public, correct, non-disqualified {args.version} submissions")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in subs:
        d = args.out / str(s["id"])
        if d.exists() and any(d.iterdir()):
            f = next(d.iterdir())
        else:
            try:
                body, headers = get(f"/submissions/{s['id']}/download", raw=True)
            except Exception as e:  # not every submission may be downloadable
                print(f"  {s['id']}: download failed ({e})")
                time.sleep(1)
                continue
            m = re.search(r'filename="?([^";]+)', headers.get("Content-Disposition", ""))
            name = Path(m[1]).name if m else "download.bin"
            d.mkdir(exist_ok=True)
            f = d / name
            f.write_bytes(body)
            time.sleep(1)
        rows.append(dict(id=s["id"], username=s["username"], submitted_at=s["submitted_at"],
                         latency_ms=s["latency_ms"], sol_score=s["sol_score"], file=f.name, bytes=f.stat().st_size))
        print(f"  {s['id']} {s['username'][:24]:24s} {s['latency_ms']:.6f} ms -> {f.name}")

    with open(args.out / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["id"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out / 'index.csv'}")


if __name__ == "__main__":
    main()
