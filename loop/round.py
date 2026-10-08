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


CLAUDE_NOTE = """=== RUNNING AS A HEADLESS CLAUDE CODE SESSION ===
The design tools are MCP tools named mcp__solx__<tool> (compile_b200, run_tests, probe_b200, predict_score,
get_kernel, read_example). No other tools are available. When you are done, your final message must contain the
candidate in the OUTPUT CONTRACT format (fenced blocks), with its design card including paths and findings."""


def run_claude(prompt, system_file, model, effort, mcp_args, sdir, name, timeout_s, resume=None):
    """One headless `claude -p` run with our MCP tools only. Returns the parsed JSON result (or an error dict)."""
    cfg = sdir / f"{name}.mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"solx": {"command": str(ROOT / ".venv" / "bin" / "python"),
                                                       "args": [str(ROOT / "loop" / "mcp_tools.py"), *mcp_args]}}}))
    cmd = ["claude", "-p", "--model", model, "--effort", effort, "--mcp-config", str(cfg), "--strict-mcp-config",
           "--tools", "", "--allowedTools", "mcp__solx", "--output-format", "json"]
    if system_file:
        cmd += ["--append-system-prompt-file", str(system_file)]
    if resume:
        cmd += ["--resume", resume]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout_s, cwd=sdir)
        out = json.loads(p.stdout) if p.stdout.strip().startswith("{") else dict(is_error=True, result=p.stdout[-3000:],
                                                                                   stderr=p.stderr[-3000:])
    except subprocess.TimeoutExpired:
        out = dict(is_error=True, result="", subtype="timeout")
    out["seconds"] = round(time.time() - t0, 1)
    return out


def claude_sessions(round_id, jobs, static, a, sdir):
    """Design sessions as headless Claude Code runs on the user's plan, in parallel; each has its own MCP tool server
    (and its own Modal B200 container, opened on its first GPU call)."""
    import designer
    sdir.mkdir(parents=True, exist_ok=True)
    system_file = sdir / "briefing.txt"
    protocol = designer.PROTOCOL.format(gpu="a rented NVIDIA B200 (Modal; unlocked clocks)", gpu_calls=a.gpu_calls,
                                        turns="about 15")
    system_file.write_text(static + "\n\n" + protocol + "\n\n" + CLAUDE_NOTE)
    minutes = max(3.0, a.b200_minutes / max(1, len(jobs)))

    def one(job):
        i, t, dyn, name = job
        mcp_args = ["--name", name, "--round", round_id, "--dir", str(sdir), "--gpu-calls", str(a.gpu_calls),
                    "--probes", str(a.probes), "--b200-minutes", str(minutes)]
        model = t.get("model") or a.session_model
        out = run_claude(dyn, system_file, model, a.effort, mcp_args, sdir, name, a.session_timeout * 60)
        text = out.get("result") or ""
        if not reply.candidates(text) and out.get("session_id"):          # one nudge for the contract
            more = run_claude("Reply now with exactly one candidate in the OUTPUT CONTRACT format.", None,
                              model, a.effort, mcp_args, sdir, name, 900, resume=out["session_id"])
            text = more.get("result") or text
            out["total_cost_usd"] = (out.get("total_cost_usd") or 0) + (more.get("total_cost_usd") or 0)
        (sdir / f"{name}.claude.json").write_text(json.dumps({k: v for k, v in out.items() if k != "result"}, indent=1))
        u = out.get("usage") or {}
        info = dict(model=f"claude:{model}", finish=out.get("subtype") or ("error" if out.get("is_error") else
                    "stop"), seconds=out.get("seconds"), turns=out.get("num_turns"), gpu_calls="-",
                    prompt_tokens=(u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0)
                    + (u.get("cache_creation_input_tokens") or 0), cached_tokens=u.get("cache_read_input_tokens"),
                    completion_tokens=u.get("output_tokens"), reasoning_tokens=None,
                    cost_usd=round(out.get("total_cost_usd") or 0, 4))
        with open(ROOT / "loop" / "runs" / "usage.jsonl", "a") as f:
            f.write(json.dumps(dict(info, time=time.strftime("%Y-%m-%dT%H:%M:%S"), tag=f"{round_id}/{name}",
                                    billing="claude-plan (cost is the API-equivalent estimate)")) + "\n")
        return i, (text, info)

    with ThreadPoolExecutor(max_workers=max(1, min(a.parallel, len(jobs)))) as pool:    # each session = claude + MCP
        return dict(pool.map(one, jobs))


LEDGER = ROOT / "loop" / "ledger.yaml"
LEAD_PROTOCOL = """=== RESEARCH LEAD PROTOCOL ===
You are the research lead of this project. You do not write kernels. Each round you:
1. Read the evidence below: the lab notebook (portal vs bench outcomes and findings recorded by design sessions),
   the archive, the hypothesis ledger, and the emulator's learned portal-vs-bench effects. Use mcp__solx__get_kernel
   to read any kernel's code and results, mcp__solx__read_example for public SOTA kernels, and
   mcp__solx__predict_score to see what a change per size band is worth.
2. Update the ledger: change a status only on evidence; add hypotheses that findings suggest; cite evidence (kernel
   ids, portal numbers, probe findings); keep each entry short. Keep the constraints.
3. Design the next round: {n} experiments for design sessions. Each session is a Claude agent with a rented B200
   (probes take seconds, full tests 10-20 s, and run_tests predicts the portal score with about +-0.01 error;
   bench noise is about 1% per size, so small effects need repeated, interleaved probes). Each experiment tests one
   hypothesis (or a new idea you add to the ledger) and states what to build or measure, the success criterion, and
   what result would refute it. Prefer experiments that could move the portal score by 0.005 or more, or that
   settle an open hypothesis blocking a large gain. Avoid tweaks worth about 1%. Exploration is encouraged.
4. Give each experiment a model: opus for open-ended design and debugging, sonnet for well-specified work.
Output, in this order:
### Assessment
(at most 250 words: where the remaining score is, what we now know, what this round tests and why)
```yaml ledger
(the complete updated ledger, same schema)
```
```json tasks
[{{"id": "E1", "hypothesis": "H8", "title": "...", "operation": "new_design", "parents": ["<archive id>"],
  "band": "S|M|L|all", "instructions": "what to build or measure, step by step", "success": "...",
  "refuted_if": "...", "model": "opus|sonnet"}}]
```"""


def cmd_lead(a):
    """Run the research lead: update the hypothesis ledger and write the round's experiments to plan.json."""
    import designer
    import emulator
    arc = get_archive()
    d = rdir(a.round)
    sdir = d / "lead"
    sdir.mkdir(parents=True, exist_ok=True)
    system_file = sdir / "briefing.txt"
    system_file.write_text(prompts.static_prompt() + "\n\n" + LEAD_PROTOCOL.format(n=a.max_tasks))
    measured = [r for r in arc.values() if r.get("b200")]
    calib = "\n".join(f"- {r['id']}: portal {r['b200']['score']:.4f}, S/M/L {archive.fmt(archive.band_geomeans(r['b200'].get('timings', {})))} µs"
                       for r in sorted(measured, key=lambda r: -r["b200"]["score"]))
    try:
        effects = emulator.effects_markdown(emulator.Emulator(arc))
    except Exception as e:
        effects = f"(unavailable: {e})"
    prompt = "\n\n".join([
        f"=== TASK ===\nRound {a.round}: update the hypothesis ledger and design {a.max_tasks} experiments."
        + (f"\n\nBRIEF FROM THE OPERATOR FOR THIS ROUND:\n{a.brief}" if getattr(a, "brief", None) else ""),
        f"=== HYPOTHESIS LEDGER (loop/ledger.yaml) ===\n{LEDGER.read_text()}",
        f"=== LAB NOTEBOOK ===\n{prompts.lab_notebook(arc, limit=80)}",
        f"=== PORTAL RESULTS (best first) ===\n{calib}",
        f"=== EMULATOR: portal minus bench, by design feature ===\n{effects}",
        f"=== ARCHIVE ===\n{archive.table(arc, limit=60)}",
    ])
    (sdir / "prompt.txt").write_text(prompt)
    out = run_claude(prompt, system_file, a.lead_model, a.effort,
                     ["--name", "lead", "--round", a.round, "--dir", str(sdir), "--no-gpu"], sdir, "lead",
                     a.session_timeout * 60)
    text = out.get("result") or ""
    (sdir / "reply.md").write_text(text)
    (sdir / "lead.claude.json").write_text(json.dumps({k: v for k, v in out.items() if k != "result"}, indent=1))
    blocks = {m.group(1).strip(): m.group(2) for m in reply.FENCE.finditer(text)}
    ledger = next((v for k, v in blocks.items() if k.startswith("yaml") and "ledger" in k), None)
    tasks_js = next((v for k, v in blocks.items() if k.startswith("json") and "tasks" in k), None)
    if not (ledger and tasks_js):
        raise SystemExit(f"lead reply had no ledger or tasks block; see {sdir / 'reply.md'}")
    import yaml
    yaml.safe_load(ledger)                                  # must parse before it replaces the ledger
    LEDGER.write_text(ledger)
    tasks = []
    for e in json.loads(tasks_js)[:a.max_tasks]:
        parents = [p for p in (e.get("parents") or []) if p in arc] or [planner.best_kernel(arc)["id"]]
        tasks.append(dict(operation=e.get("operation") or "new_design", parents=parents, band=e.get("band") or "all",
                          niche=f"exp:{e.get('id', '?')}:{e.get('hypothesis', '-')}", model=e.get("model"),
                          instructions=(f"Experiment {e.get('id')} (tests {e.get('hypothesis')}): {e.get('title', '')}. "
                                        f"{e.get('instructions', '')} Success: {e.get('success', '')} Refuted if: "
                                        f"{e.get('refuted_if', '')}" + planner.EXPLORE_COMMON +
                                        " Set the card's tests: field to the hypothesis id(s).")))
    best = planner.best_kernel(arc)
    (d / "plan.json").write_text(json.dumps(dict(best=best["id"], best_score=best["b200"]["score"], tasks=tasks,
                                                 lead=dict(model=a.lead_model, cost_usd=out.get("total_cost_usd"))),
                                            indent=1))
    print(text.split("```")[0].strip()[:3000])
    for i, t in enumerate(tasks):
        print(f"  task {i}: {t['niche']} parents={t['parents']} band={t['band']} model={t.get('model')}")
    u = out.get("usage") or {}
    print(f"lead: {out.get('num_turns')} turns, {out.get('seconds')} s, API-equivalent ${out.get('total_cost_usd') or 0:.2f}")
    return dict(tasks=tasks)


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
    if a.backend == "claude":
        results = claude_sessions(round_id, jobs, static, a, d / "sessions")
    elif a.interactive:
        import designer
        if a.tool_gpu != "B200":
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
    p = planner.plan(arc, max_tasks=a.max_tasks, last_round=a.last_round, mode=a.mode)
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
    if a.use_plan:                                 # run a plan already written (e.g. by `round.py lead`)
        p = json.loads((rdir(a.round) / "plan.json").read_text())
        print(f"using the saved plan: {len(p['tasks'])} task(s)")
    else:
        p = cmd_lead(a) if a.lead else make_plan(a)
    propose_tasks(a.round, p["tasks"], a)
    if not a.dry_run and any((rdir(a.round) / "candidates").glob("*.json")):
        a.gpus = a.gpus or ("B200" if a.tool_gpu == "B200" or a.backend == "claude" else "A100,H200")
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
    import designer
    measured = {}                                     # code hash -> {test, compile} from this round's sessions
    for f in (d / "sessions").glob("*.b200.json"):
        for key, v in json.loads(f.read_text()).items():
            measured.setdefault(key, {}).update(v)
    statics, reused, ran = [], 0, []
    t0 = time.time()
    for sol_path in todo:
        sol = json.loads(sol_path.read_text())
        m = measured.get(designer.code_key(sol), {})
        test_ok = m.get("test") and all(w["status"] == "PASSED" for w in m["test"].get("workloads") or [{}])
        if test_ok:
            b200_modal.write_trace(out / f"{sol_path.stem}.jsonl", m["test"])
            reused += 1
            print(f"B200r {sol_path.stem}: reused the session's B200 timing (same code)", flush=True)
            if m.get("compile"):
                statics.append(dict(id=sol_path.stem, kernels=m["compile"].get("kernels") or [],
                                    error=m["compile"].get("error") or ""))
                continue
        ran.append((sol_path, sol, m, not test_ok))
    if ran:
        with b200_modal.app.run():
            for sol_path, sol, m, need_test in ran:
                if not need_test:                     # timing reused; only the compile statistics are missing
                    comp = b200_modal.handle.remote(dict(id=f"{sol_path.stem}-c", kind="compile", solution=sol))
                    statics.append(dict(id=sol_path.stem, kernels=comp.get("kernels") or [],
                                        error=comp.get("error") or ""))
                    continue
                res = b200_modal.handle.remote(dict(id=sol_path.stem, kind="test", solution=sol))
                b200_modal.write_trace(out / f"{sol_path.stem}.jsonl", res)
                wl = res.get("workloads") or []
                comp = m.get("compile") or b200_modal.handle.remote(dict(id=f"{sol_path.stem}-c", kind="compile",
                                                                         solution=sol))
                statics.append(dict(id=sol_path.stem, kernels=comp.get("kernels") or [], error=comp.get("error") or ""))
                print(f"B200r {sol_path.stem}: {sum(w['status'] == 'PASSED' for w in wl)}/{len(wl)} passed", flush=True)
    if statics:
        with open(d / "results" / "static_B200r.jsonl", "a") as f:
            f.write("".join(json.dumps(x) + "\n" for x in statics))
    print(f"rented B200: {sum(1 for r in ran if r[3])} candidate(s) tested in {(time.time() - t0) / 60:.1f} min, "
          f"{reused} timing(s) reused from the sessions", flush=True)


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
             "| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |",
             "|---|---|---|---|---|"]
    for w in s.get("awaiting", []):
        lines.append(f"| (not copied) `{w['id']}` | waiting for {w['gpu']} timings | - | run collect again when they land | - |")
    for p in s["picks"]:
        shutil.copy(ROOT / arc[p["id"]]["solution"], out / f"{p['id']}.json")
        pred = f"{p['predicted_score']:.4f}" if p["predicted_score"] else "unknown (exploration)"
        if p.get("sd"):
            pred += f" ± {p['sd']:.4f}"
        pb = f"{p['p_better']:.0%}" if p.get("p_better") is not None else "-"
        lines.append(f"| `{p['id']}.json` | {p['gpu'] or '-'} | {p['rel']} | {pred} | {pb} |")
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
    llm_args.add_argument("--backend", choices=("openrouter", "claude"), default="openrouter",
                          help="claude: run design sessions as headless Claude Code on your Claude plan (implies "
                          "interactive tools on the rented B200)")
    llm_args.add_argument("--session-model", default="opus", help="claude backend: model for design sessions")
    llm_args.add_argument("--lead-model", default="fable", help="claude backend: model for the research lead")
    llm_args.add_argument("--session-timeout", type=int, default=60, help="claude backend: minutes per session")
    llm_args.add_argument("--parallel", type=int, default=4, help="claude backend: sessions at once (each runs a "
                          "claude process and an MCP server; keep low on a laptop)")
    llm_args.add_argument("--interactive", action="store_true",
                          help="design sessions with tools (compile, test, score model) on a CSF3 GPU tool server")
    llm_args.add_argument("--tool-gpu", default="A100,L40S", help="GPU type(s) for the session tool server: CSF3 types "
                          "(several with one Slurm account are queued together; the first free one is used), or B200 "
                          "for a rented B200 on Modal (run round.py with .venv/bin/python)")
    llm_args.add_argument("--b200-minutes", type=int, default=20, help="interactive with --tool-gpu B200: cap on "
                          "Modal B200 minutes per round (about $6.25 an hour)")
    llm_args.add_argument("--gpu-wait", type=int, default=180, help="interactive: minutes to wait for the tool server")
    llm_args.add_argument("--turns", type=int, default=12, help="interactive: max LLM turns per session")
    llm_args.add_argument("--probes", type=int, default=10, help="interactive with --tool-gpu B200: max probe_b200 "
                          "experiments per session")
    llm_args.add_argument("--gpu-calls", type=int, default=6, help="interactive: max compile/test calls per session")
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
    for name in ("plan", "auto", "lead"):
        q = sub.add_parser(name, parents=[llm_args])
        q.add_argument("--round", required=True)
        q.add_argument("--max-tasks", type=int, default=3)
        q.add_argument("--last-round", help="round whose failures get repair tasks")
        q.add_argument("--use-plan", action="store_true", help="auto: run rounds/<round>/plan.json as it is")
        q.add_argument("--brief", help="extra guidance for the research lead this round")
        q.add_argument("--lead", action="store_true", help="auto: let the research lead (claude backend) update "
                       "the hypothesis ledger and choose the experiments instead of the planner")
        q.add_argument("--mode", choices=("exploit", "explore"), default="exploit",
                       help="exploit: improve the best kernel band by band; explore: one big idea per session")
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
    dict(propose=cmd_propose, plan=cmd_plan, auto=cmd_auto, lead=cmd_lead, test=cmd_test, status=cmd_status, collect=cmd_collect,
         shortlist=cmd_shortlist, table=cmd_table)[a.cmd](a)


if __name__ == "__main__":
    main()
