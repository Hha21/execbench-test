"""Run one round of the kernel loop: propose candidates with the inner LLM, test them on CSF3, collect results.

  python3 loop/round.py propose --round r1 --operation structural_mutation --parents g2-os-r8w4 \
          --band L --instructions "..." [--calls 1] [--n 1] [--model anthropic/claude-opus-5.5] [--dry-run]
  python3 loop/round.py test    --round r1 --gpus A100,H200
  python3 loop/round.py status  --round r1
  python3 loop/round.py collect --round r1
  python3 loop/round.py table

Candidates live in loop/rounds/<round>/candidates/ (solution JSON + design card); lint failures go to rejected/.
Portal results arrive through poc/ingest_portal.py and are picked up by `collect` and `table`.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import archive  # noqa: E402
import prompts  # noqa: E402
import reply  # noqa: E402

ROOT = archive.ROOT
REMOTE = "csf3"
RSOLX = "/scratch/t95317ha/solx"
SLURM = {"A100": ("gpu-sk01", "gpuA"), "L40S": ("gpu-sk01", "gpuL"), "H200": ("gpu-cdt-dmcs", "gpuH_short")}


def rdir(r):
    return ROOT / "loop" / "rounds" / r


def ssh(cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE, cmd], capture_output=True, text=True, check=True).stdout


def get_archive():
    return archive.load() or archive.seed()


def cmd_propose(a):
    import llm
    arc = get_archive()
    d = rdir(a.round)
    for sub in ("candidates", "rejected", "replies"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    static = prompts.static_prompt()
    for i in range(a.calls):
        dyn = prompts.dynamic_prompt(arc, a.round, a.operation, a.parents, n=a.n, band=a.band, niche=a.niche,
                                     extra=a.instructions)
        (d / "replies" / f"prompt_{i}.txt").write_text(dyn)
        if a.dry_run:
            print(f"dry run: static prompt {len(static):,} chars (~{len(static) // 4:,} tokens), "
                  f"dynamic {len(dyn):,} chars (~{len(dyn) // 4:,} tokens); saved replies/prompt_{i}.txt")
            continue
        text, info = llm.chat(static, dyn, model=a.model, max_tokens=a.max_tokens, effort=a.effort,
                              tag=f"{a.round}/{i}")
        (d / "replies" / f"reply_{i}.md").write_text(text)
        print(f"call {i}: {info['prompt_tokens']} in ({info['cached_tokens']} cached), {info['completion_tokens']} out "
              f"({info['reasoning_tokens']} reasoning), ${info['cost_usd']}, {info['seconds']} s, "
              f"finish={info['finish']}")
        cands = reply.candidates(text)
        if not cands:
            print("  no parseable candidate in the reply")
        for c in cands:
            cid, sol, card, problems = reply.build(c, a.round, set(arc))
            if sol is None:
                print(f"  rejected: {problems}")
                continue
            hits = reply.lint(sol)
            ok = not (problems or hits)
            where = d / ("candidates" if ok else "rejected")
            (where / f"{cid}.json").write_text(json.dumps(sol, indent=1))
            (where / f"{cid}.card.yaml").write_text(card)
            pc = archive.parse_card(card)
            arc[cid] = dict(id=cid, round=a.round, parents=pc.get("parents") or a.parents,
                            operation=pc.get("operation") or a.operation, language=pc.get("language", "triton"),
                            niche=pc.get("niche") or {}, card=card, solution=str((where / f"{cid}.json").relative_to(ROOT)),
                            status="proposed" if ok else "rejected: " + "; ".join(problems + hits), timings={})
            print(f"  {cid}: {arc[cid]['status']}")
        archive.save(arc)


def cmd_test(a):
    subprocess.run(["rsync", "-a", "--exclude", "__pycache__", f"{ROOT}/loop/", f"{REMOTE}:{RSOLX}/loop/"], check=True)
    subprocess.run(["rsync", "-a", f"{ROOT}/poc/run_timing.py", f"{REMOTE}:{RSOLX}/poc/"], check=True)
    jobs = {}
    for gpu in a.gpus.split(","):
        acct, part = SLURM[gpu]
        out = ssh(f"cd {RSOLX} && sbatch --parsable -A {acct} -p {part} --export=ALL,ROUND={a.round} "
                  f"loop/jobs/test_round.sbatch")
        jobs[gpu] = out.strip()
        print(f"{gpu}: job {jobs[gpu]}")
    f = rdir(a.round) / "jobs.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    f.write_text(json.dumps({**old, **jobs}, indent=1))


def cmd_status(a):
    jobs = json.loads((rdir(a.round) / "jobs.json").read_text())
    print(ssh(f"sacct -X -n -P -j {','.join(jobs.values())} -o JobID,Partition,State,Start,Elapsed"))


def cmd_collect(a):
    d = rdir(a.round)
    (d / "results").mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", f"{REMOTE}:{RSOLX}/loop/rounds/{a.round}/results/", f"{d}/results/"], check=True)
    arc = get_archive()
    for gdir in sorted((d / "results" / "timing").glob("*")) if (d / "results" / "timing").exists() else []:
        for tr in gdir.glob("*.jsonl"):
            cid = tr.stem
            if cid in arc:
                times, statuses = archive.read_traces(tr)
                arc[cid].setdefault("timings", {})[gdir.name] = times
                arc[cid]["status"] = "passed" if statuses == {"PASSED"} else f"failed: {sorted(statuses)}"
                log = gdir / f"{cid}.log"
                if log.exists() and statuses != {"PASSED"}:
                    arc[cid]["failure_log"] = log.read_text()[-3000:]
    for sf in sorted((d / "results").glob("static_*.jsonl")):
        for line in open(sf):
            s = json.loads(line)
            if s["id"] in arc and s.get("kernels"):
                arc[s["id"]]["static_sm100"] = [{k: v for k, v in kk.items() if k != "target"} for kk in s["kernels"]]
    archive.ingest_portal(arc)
    archive.save(arc)
    print(archive.table({k: v for k, v in arc.items() if v.get("round") == a.round or v.get("b200")}))


def cmd_table(a):
    arc = get_archive()
    archive.ingest_portal(arc)
    archive.save(arc)
    print(archive.table(arc))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose")
    p.add_argument("--round", required=True)
    p.add_argument("--operation", required=True)
    p.add_argument("--parents", nargs="*", default=[])
    p.add_argument("--band", default="all")
    p.add_argument("--niche", default="any empty niche")
    p.add_argument("--instructions", default="")
    p.add_argument("--calls", type=int, default=1)
    p.add_argument("--n", type=int, default=1, help="candidates per call")
    p.add_argument("--model", default="anthropic/claude-opus-5.5")
    p.add_argument("--effort", default="high")
    p.add_argument("--max-tokens", type=int, default=32000)
    p.add_argument("--dry-run", action="store_true")
    for name in ("test", "status", "collect"):
        q = sub.add_parser(name)
        q.add_argument("--round", required=True)
        if name == "test":
            q.add_argument("--gpus", default="A100")
    sub.add_parser("table")
    a = ap.parse_args()
    if a.cmd == "propose" and not re.fullmatch(r"[a-z0-9-]+", a.round):
        sys.exit("--round must be lowercase letters, digits and dashes")
    dict(propose=cmd_propose, test=cmd_test, status=cmd_status, collect=cmd_collect, table=cmd_table)[a.cmd](a)


if __name__ == "__main__":
    main()
