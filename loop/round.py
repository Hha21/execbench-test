"""Run the kernel loop: plan a round, propose candidates with the inner LLM, test them on CSF3, collect results,
and shortlist the best for the B200 portal.

  python3 loop/round.py auto      --round r3 [--max-tasks 3] [--gpus A100,H200]   plan + propose + test
                                  [--interactive [--tool-gpu A100,L40S] [--turns 12] [--gpu-calls 8]]
  python3 loop/round.py plan      --round r3                                    show what the planner would do
  python3 loop/round.py propose   --round r3 --operation structural_mutation --parents g2-os-r16w8 \
                                  --band L --instructions "..." [--dry-run]      one hand-written task
  python3 loop/round.py test      --round r3 --gpus A100,H200
  python3 loop/round.py status    --round r3
  python3 loop/round.py collect   --round r3
  python3 loop/round.py shortlist --round r3                                    -> rounds/r3/portal/
  python3 loop/round.py table

With --interactive, each task is a design session (loop/designer.py): the LLM can compile its drafts for sm_100a,
test and time them on a CSF3 GPU against the current best, and query the score model before it answers.

Candidates live in loop/rounds/<round>/candidates/ (solution JSON + design card); lint failures go to rejected/.
Portal results arrive through poc/ingest_portal.py and are picked up by collect, shortlist and table.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import archive  # noqa: E402
import planner  # noqa: E402
import prompts  # noqa: E402
import reply  # noqa: E402

ROOT = archive.ROOT
REMOTE = "csf3"
RSOLX = "/scratch/t95317ha/solx"
SLURM = {"A100": ("gpu-sk01", "gpuA"), "L40S": ("gpu-sk01", "gpuL"), "H200": ("gpu-cdt-dmcs", "gpuH_short")}
MODEL = "anthropic/claude-opus-5.5"


def rdir(r):
    return ROOT / "loop" / "rounds" / r


def ssh(cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE, cmd], capture_output=True, text=True, check=True).stdout


def get_archive():
    arc = archive.load() or archive.seed()
    archive.ingest_portal(arc)
    return arc


def slug(task, i):
    return f"{i:02d}-{task['operation']}-{re.sub(r'[^a-z0-9]+', '', task.get('band', 'all').lower())}"


def call_llm(static, dyn, task, round_id, i, a):
    """One task -> (text, info); retries once at medium effort if the answer ran out of tokens while thinking."""
    import llm
    tag = f"{round_id}/{slug(task, i)}"
    text, info = llm.chat(static, dyn, model=a.model, max_tokens=a.max_tokens, effort=a.effort, tag=tag)
    if info["finish"] == "length" and not reply.candidates(text):
        print(f"  {tag}: ran out of tokens without an answer; retrying at medium effort", flush=True)
        text2, info2 = llm.chat(static, dyn, model=a.model, max_tokens=a.max_tokens, effort="medium", tag=tag + "/retry")
        info2["cost_usd"] = (info2.get("cost_usd") or 0) + (info.get("cost_usd") or 0)
        return text2, info2
    return text, info


def propose_tasks(round_id, tasks, a):
    arc = get_archive()
    d = rdir(round_id)
    for sub in ("candidates", "rejected", "replies"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    static = prompts.static_prompt()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    jobs = []
    for i, t in enumerate(tasks):
        dyn = prompts.dynamic_prompt(arc, round_id, t["operation"], t["parents"], n=1, band=t.get("band", "all"),
                                     niche=t.get("niche", "any empty niche"), extra=t.get("instructions", ""),
                                     feedback=t.get("feedback", "none"))
        name = f"{stamp}_{slug(t, i)}"
        (d / "replies" / f"{name}.prompt.txt").write_text(dyn)
        jobs.append((i, t, dyn, name))
    if a.dry_run:
        for i, t, dyn, name in jobs:
            print(f"dry run {name}: static ~{len(static) // 4:,} tokens (cached), dynamic ~{len(dyn) // 4:,} tokens")
        return
    if a.interactive:
        import designer
        sync_remote()
        results = designer.run_sessions(round_id, jobs, static, a, SLURM, d / "sessions")
    else:
        results = {}
        first, rest = jobs[0], jobs[1:]
        results[first[0]] = call_llm(static, first[2], first[1], round_id, first[0], a)   # warms the prompt cache
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {j[0]: pool.submit(call_llm, static, j[2], j[1], round_id, j[0], a) for j in rest}
            for i, f in futs.items():
                results[i] = f.result()
    total = 0.0
    for i, t, dyn, name in jobs:
        text, info = results[i]
        (d / "replies" / f"{name}.reply.md").write_text(text)
        total += info.get("cost_usd") or 0
        extra = f", {info['turns']} turns, {info['gpu_calls']} GPU calls" if "turns" in info else ""
        print(f"{name}: {info['prompt_tokens']} in ({info['cached_tokens']} cached), {info['completion_tokens']} out "
              f"({info['reasoning_tokens']} reasoning), ${info['cost_usd']}, {info['seconds']} s, "
              f"finish={info.get('finish')}{extra}")
        cands = reply.candidates(text)
        if not cands:
            print("  no parseable candidate in the reply")
        for c in cands:
            cid, sol, card, problems = reply.build(c, round_id, set(arc))
            if sol is None:
                print(f"  rejected: {problems}")
                continue
            hits = reply.lint(sol)
            ok = not (problems or hits)
            where = d / ("candidates" if ok else "rejected")
            (where / f"{cid}.json").write_text(json.dumps(sol, indent=1))
            (where / f"{cid}.card.yaml").write_text(card)
            pc = archive.parse_card(card)
            arc[cid] = dict(id=cid, round=round_id, parents=pc.get("parents") or t["parents"],
                            operation=pc.get("operation") or t["operation"], language=pc.get("language", "triton"),
                            niche=pc.get("niche") or {}, card=card, task=t,
                            solution=str((where / f"{cid}.json").relative_to(ROOT)),
                            status="proposed" if ok else "rejected: " + "; ".join(problems + hits), timings={})
            print(f"  {cid}: {arc[cid]['status']}")
    archive.save(arc)
    print(f"round {round_id}: {len(jobs)} calls, ${total:.2f}")


def cmd_propose(a):
    task = dict(operation=a.operation, parents=a.parents, band=a.band, niche=a.niche, instructions=a.instructions)
    propose_tasks(a.round, [task], a)


def make_plan(a):
    arc = get_archive()
    p = planner.plan(arc, max_tasks=a.max_tasks, last_round=a.last_round)
    d = rdir(a.round)
    d.mkdir(parents=True, exist_ok=True)
    (d / "plan.json").write_text(json.dumps(p, indent=1, default=str))
    g = p["gains"]
    print(f"best on B200: {p['best']} (score {p['best_score']:.4f}); fit {g['fixed']:.1f} us fixed + {g['bw']:.1f} TB/s; "
          f"M {g['tbs_M']:.2f} TB/s, L {g['tbs_L']:.2f} TB/s")
    print(f"estimated score gain: small-input fixed cost -> {planner.FIXED_TARGET_US} us: {g['S']:+.3f}; "
          f"medium/large a further 5%: {g['ML']:+.3f}")
    for i, t in enumerate(p["tasks"]):
        print(f"  task {i}: {t['operation']} parents={t['parents']} band={t.get('band')} niche={t.get('niche', '-')}")
    return p


def cmd_plan(a):
    make_plan(a)


def cmd_auto(a):
    p = make_plan(a)
    propose_tasks(a.round, p["tasks"], a)
    if not a.dry_run and any((rdir(a.round) / "candidates").glob("*.json")):
        a.gpus = a.gpus or ("B200" if a.tool_gpu == "B200" else "A100,H200")
        cmd_test(a)
        print(f"next: python3 loop/round.py collect --round {a.round} (after any CSF3 jobs finish); then shortlist")


def sync_remote():
    subprocess.run(["rsync", "-a", "--exclude", "__pycache__", f"{ROOT}/loop/", f"{REMOTE}:{RSOLX}/loop/"], check=True)
    subprocess.run(["rsync", "-a", f"{ROOT}/poc/run_timing.py", f"{REMOTE}:{RSOLX}/poc/"], check=True)


def b200_test(round_id):
    """Test every candidate of a round on the rented B200 (Modal): timings -> results/timing/B200r, compile stats ->
    results/static_B200r.jsonl. Runs in this process (a few minutes); candidates already timed are skipped."""
    import b200_modal
    b200_modal.load_token()
    d = rdir(round_id)
    out = d / "results" / "timing" / "B200r"
    todo = [p for p in sorted((d / "candidates").glob("*.json")) if not (out / f"{p.stem}.jsonl").exists()]
    statics = []
    t0 = time.time()
    with b200_modal.app.run():
        for sol_path in todo:
            sol = json.loads(sol_path.read_text())
            res = b200_modal.handle.remote(dict(id=sol_path.stem, kind="test", solution=sol))
            b200_modal.write_trace(out / f"{sol_path.stem}.jsonl", res)
            wl = res.get("workloads") or []
            comp = b200_modal.handle.remote(dict(id=f"{sol_path.stem}-c", kind="compile", solution=sol))
            statics.append(dict(id=sol_path.stem, kernels=comp.get("kernels") or [], error=comp.get("error") or ""))
            print(f"B200r {sol_path.stem}: {sum(w['status'] == 'PASSED' for w in wl)}/{len(wl)} passed", flush=True)
    if statics:
        with open(d / "results" / "static_B200r.jsonl", "a") as f:
            f.write("".join(json.dumps(x) + "\n" for x in statics))
    print(f"rented B200: {len(todo)} candidate(s) in {(time.time() - t0) / 60:.1f} min", flush=True)


def cmd_test(a):
    gpus = a.gpus.split(",")
    if "B200" in gpus:
        b200_test(a.round)
        gpus = [g for g in gpus if g != "B200"]
    if not gpus:
        return
    sync_remote()
    jobs = {}
    for gpu in gpus:
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
    r = subprocess.run(["rsync", "-a", f"{REMOTE}:{RSOLX}/loop/rounds/{a.round}/results/", f"{d}/results/"])
    if r.returncode:
        print("(no CSF3 results for this round; using local results only)")
    arc = get_archive()            # archive.load() re-reads every result file, including the ones just pulled
    archive.save(arc)
    print(archive.table({k: v for k, v in arc.items() if v.get("round") == a.round or v.get("b200")}))


def cmd_shortlist(a):
    arc = get_archive()
    s = planner.shortlist(arc, a.round)
    out = rdir(a.round) / "portal"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    lines = [f"# Portal shortlist for round {a.round}", "",
             f"Reference: {s['best']} (B200 score {s['best_score']:.4f}). Predicted scores project each candidate's "
             f"speed relative to the reference, per workload, on the rented B200 when measured there (else its most "
             f"representative cheap GPU), onto the reference's "
             f"B200 per-workload times.", "",
             "| file | basis | change vs reference S/M/L | predicted B200 score |", "|---|---|---|---|"]
    for w in s.get("awaiting", []):
        lines.append(f"| (not copied) `{w['id']}` | waiting for {w['gpu']} timings | - | run collect again when they land |")
    for p in s["picks"]:
        shutil.copy(ROOT / arc[p["id"]]["solution"], out / f"{p['id']}.json")
        pred = f"{p['predicted_score']:.4f}" if p["predicted_score"] else "unknown (exploration)"
        lines.append(f"| `{p['id']}.json` | {p['gpu'] or '-'} | {p['rel']} | {pred} |")
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_table(a):
    arc = get_archive()
    archive.save(arc)
    print(archive.table(arc))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    llm_args = argparse.ArgumentParser(add_help=False)
    llm_args.add_argument("--model", default=MODEL)
    llm_args.add_argument("--effort", default="high")
    llm_args.add_argument("--max-tokens", type=int, default=64000)
    llm_args.add_argument("--dry-run", action="store_true")
    llm_args.add_argument("--interactive", action="store_true",
                          help="design sessions with tools (compile, test, score model) on a CSF3 GPU tool server")
    llm_args.add_argument("--tool-gpu", default="A100,L40S", help="GPU type(s) for the session tool server: CSF3 types "
                          "(several with one Slurm account are queued together; the first free one is used), or B200 "
                          "for a rented B200 on Modal (run round.py with .venv/bin/python)")
    llm_args.add_argument("--b200-minutes", type=int, default=60, help="interactive with --tool-gpu B200: cap on "
                          "Modal B200 minutes per round (about $6.25 an hour)")
    llm_args.add_argument("--gpu-wait", type=int, default=180, help="interactive: minutes to wait for the tool server")
    llm_args.add_argument("--turns", type=int, default=12, help="interactive: max LLM turns per session")
    llm_args.add_argument("--gpu-calls", type=int, default=8, help="interactive: max compile/test calls per session")
    llm_args.add_argument("--turn-effort", default="medium", help="interactive: reasoning effort after the first turn")
    llm_args.add_argument("--session-budget", type=float, default=5.0, help="interactive: USD per session before "
                          "the model is told to finish")
    p = sub.add_parser("propose", parents=[llm_args])
    p.add_argument("--round", required=True)
    p.add_argument("--operation", required=True)
    p.add_argument("--parents", nargs="*", default=[])
    p.add_argument("--band", default="all")
    p.add_argument("--niche", default="any empty niche")
    p.add_argument("--instructions", default="")
    for name in ("plan", "auto"):
        q = sub.add_parser(name, parents=[llm_args])
        q.add_argument("--round", required=True)
        q.add_argument("--max-tasks", type=int, default=3)
        q.add_argument("--last-round", help="round whose failures get repair tasks")
        q.add_argument("--gpus", help="where to test the final candidates: B200 (rented, Modal) and/or CSF3 GPUs; "
                       "default B200 with --tool-gpu B200, else A100,H200")
    for name in ("test", "status", "collect", "shortlist"):
        q = sub.add_parser(name)
        q.add_argument("--round", required=True)
        if name == "test":
            q.add_argument("--gpus", default="A100,H200", help="B200 (rented, Modal) and/or CSF3 GPU types")
    sub.add_parser("table")
    a = ap.parse_args()
    if getattr(a, "round", None) and not re.fullmatch(r"[a-z0-9-]+", a.round):
        sys.exit("--round must be lowercase letters, digits and dashes")
    dict(propose=cmd_propose, plan=cmd_plan, auto=cmd_auto, test=cmd_test, status=cmd_status, collect=cmd_collect,
         shortlist=cmd_shortlist, table=cmd_table)[a.cmd](a)


if __name__ == "__main__":
    main()
