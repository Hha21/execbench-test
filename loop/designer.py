"""Interactive design sessions: the inner LLM drafts a kernel, checks it with tools, revises it, then submits.

Tools offered to the model:
  compile_b200  compile a draft for sm_100a (B200): registers, shared memory, spills, CTAs per SM, load/store widths
  run_tests     NVIDIA's harness on a cheap GPU: correctness on all 16 workloads, timing against the current best
                kernel on the same GPU, and the B200 score that timing projects to (the emulator)
  predict_score the planner's maths model: per-band latency changes -> B200 score
  get_kernel    any archive kernel's source, design card and measurements
The two GPU tools go to loop/toolserver.py, a Slurm job on CSF3 that serves every session of a round through a
file queue on /scratch. Transcripts are saved to loop/rounds/<round>/sessions/<name>.{jsonl,md}.
"""

import json
import math
import subprocess
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import archive
import llm
import planner
import prompts
import reply

REMOTE, RSOLX = "csf3", "/scratch/t95317ha/solx"
CACHE = {"type": "ephemeral", "ttl": "1h"}
EXAMPLES = archive.ROOT / "loop" / "context" / "examples"

PROTOCOL = """=== DESIGN SESSION PROTOCOL (interactive) ===
This is an interactive design session. Use the tools to check and improve your design before you commit to it:
1. Draft. 2. compile_b200: fix compile errors; check registers, CTAs per SM, spills, and that the B200 SASS uses the
memory instructions you intended (e.g. LDG.256 / STG.256). 3. run_tests: correctness on all 16 workloads, timing on a
cheap GPU against the current best kernel, and the projected B200 score. 4. Revise while it pays. 5. Final answer.
- Local timing runs on {gpu}. Unless that is a B200, it cannot run B200-only paths (256-bit loads, sm_100-only
  instructions) or show B200's 126 MB L2; those run a fallback there, so judge them from compile_b200 output and
  reasoning, not from local speed. A rented B200 runs unlocked clocks (the portal locks SM 1500 / DRAM 3996 MHz), so
  trust its relative timings more than its absolute ones. Timing noise is
  about 1% on medium/large sizes and up to 3% on the smallest; re-test before trusting a change that small.
- predict_score turns per-band latency changes into a B200 score, so you can see which band is worth the effort.
- get_kernel returns any archive kernel's source, design card and measurements; read_example shows annotated
  excerpts of the best public B200 norm kernels (see b200_sota.md).
- Budget: {gpu_calls} GPU calls (compile_b200 + run_tests) and {turns} turns. GPU calls queue on one shared GPU; a test
  takes 1-4 minutes, a compile under 1. Pass complete files every time; nothing persists between calls.
- Finish with a reply that makes no tool call and contains exactly one candidate in the OUTPUT CONTRACT format. Put
  what you measured into the design card (hypothesis, expected_effect). Submit the best version you tested, with its
  code unchanged apart from comments; if you clean it up, run_tests it again first. If its benefit is B200-only, say
  so in the card."""

DRAFT = {"type": "object", "properties": {
    "solution_spec": {"type": "object", "description": "the solution JSON without sources, e.g. {\"languages\": "
                      "[\"cute_dsl\"], \"entry_point\": \"kernel.py::run\", \"dependencies\": [\"torch\"], "
                      "\"destination_passing_style\": true}"},
    "files": {"type": "array", "items": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string", "description": "complete file"}},
        "required": ["path", "content"]}},
    "note": {"type": "string", "description": "one line: what this draft changes"}},
    "required": ["solution_spec", "files"]}

TOOLS = [
    {"type": "function", "function": {"name": "compile_b200", "parameters": DRAFT, "description":
        "Compile a draft for sm_100a (B200) and report each kernel's registers, shared memory, local (spill) bytes, "
        "CTAs per SM when known, and the count of global load/store instructions by width (LDG.256, LDG.128, ...). "
        "Returns compiler errors if it does not build. Under a minute."}},
    {"type": "function", "function": {"name": "run_tests", "parameters": DRAFT, "description":
        "Run NVIDIA's harness on the session GPU: correctness on all 16 workloads (with errors and logs on failure) "
        "and latency per workload, compared with the current best kernel timed on the same GPU, plus the projected "
        "B200 score. 1-4 minutes."}},
    {"type": "function", "function": {"name": "predict_score", "description":
        "Predict the B200 score from latency changes relative to the current best kernel's B200 times, using the "
        "real scoring formula and the hidden baseline per workload. Give pct_change per band (negative = faster), "
        "or fixed_us and tbs to model every size as fixed_us + bytes / tbs. Also reports how much a 10% gain in "
        "each band is worth.", "parameters": {"type": "object", "properties": {
            "pct_change": {"type": "object", "properties": {"S": {"type": "number"}, "M": {"type": "number"},
                                                            "L": {"type": "number"}}},
            "fixed_us": {"type": "number"}, "tbs": {"type": "number"}}}}},
    {"type": "function", "function": {"name": "read_example", "description":
        "Annotated excerpts of state-of-the-art public kernels (FlashInfer, quack, SGLang, CAKE), with source and "
        "licence. Call with no name for the list. Learn techniques from them; do not copy them wholesale.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "get_kernel", "description":
        "Source, design card, measurements and sm_100a statistics of an archive kernel, by id.",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}},
]


def ssh(cmd, stdin=None, check=True):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE, cmd], input=stdin, capture_output=True, text=True,
                          check=check).stdout


band_of = planner.band_of


def gmean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


class ToolServer:
    """Client side of loop/toolserver.py: start the Slurm job, send requests, wait for results."""

    def __init__(self, round_id, gpus, slurm, ref_solution, wait_s=2700):
        """gpus: one GPU type or several sharing a Slurm account ("A100,L40S"); the job runs on whichever frees first."""
        self.name = f"{round_id}-{time.strftime('%m%d-%H%M%S')}"
        self.queue = f"{RSOLX}/toolq/{self.name}"
        self.wait_s, self.ref_solution = wait_s, ref_solution
        accts = {slurm[g][0] for g in gpus.split(",")}
        if len(accts) != 1:
            raise SystemExit(f"--tool-gpu {gpus}: these GPUs use different Slurm accounts; pick one account's GPUs")
        parts = ",".join(slurm[g][1] for g in gpus.split(","))
        ssh(f"mkdir -p {self.queue}/req")
        self.job = ssh(f"cd {RSOLX} && sbatch --parsable -A {accts.pop()} -p {parts} --export=ALL,QUEUE={self.queue},"
                       f"REF={ref_solution} loop/jobs/toolserver.sbatch").strip()
        self.gpu, self.ref, self.lock, self.n = gpus, None, threading.Lock(), 0
        print(f"tool server: {gpus} job {self.job}, queue {self.queue}", flush=True)

    def wait_alive(self, timeout_s):
        """Block until the server job runs (returns the GPU name) or the timeout passes (returns None)."""
        t0, last = time.time(), None
        while time.time() - t0 < timeout_s:
            name = ssh(f"cat {self.queue}/alive 2>/dev/null", check=False).strip()
            if name:
                self.gpu = name
                return name
            state = ssh(f"squeue -h -j {self.job} -o '%T %R'", check=False).strip()
            if state != last:
                print(f"tool server job {self.job}: {state or 'not in queue'}", flush=True)
                last = state
            if not state:
                return None                               # job ended without starting the server
            time.sleep(30)
        return None

    def stop(self):
        ssh(f"touch {self.queue}/stop; scancel {self.job}", check=False)

    def call(self, kind, sol):
        with self.lock:
            self.n += 1
            rid = f"{kind}-{self.n:03d}"
        ssh(f"cat > {self.queue}/req/.{rid}.part && mv {self.queue}/req/.{rid}.part {self.queue}/req/{rid}.json",
            stdin=json.dumps(dict(id=rid, kind=kind, solution=sol)))
        t0 = time.time()
        while time.time() - t0 < self.wait_s:
            out = ssh(f"cat {self.queue}/res/{rid}.json 2>/dev/null", check=False)
            if out.strip():
                return json.loads(out)
            time.sleep(5)
        ssh(f"rm -f {self.queue}/req/{rid}.json", check=False)
        return dict(error=f"no result after {self.wait_s // 60} minutes (the GPU job may still be queued); "
                          "continue without this measurement")

    def reference(self):
        if self.ref is None:
            out = ssh(f"cat {self.queue}/ref.json 2>/dev/null", check=False)
            if out.strip():
                self.ref = json.loads(out)
        return self.ref


class ModalB200:
    """Same interface as ToolServer, backed by a real B200 on Modal (loop/b200_modal.py), billed per second.

    Calls are serialised on one container (max_containers=1), so timings never overlap. A cap on B200 minutes per
    round stops the GPU tools (sessions are told to finish) rather than letting spend run on.
    """

    def __init__(self, round_id, ref_solution, budget_minutes=60):
        import modal
        import b200_modal
        b200_modal.load_token()
        self.name, self.ref_solution, self.budget_s = f"{round_id}-modal", ref_solution, budget_minutes * 60
        self._ctx = b200_modal.app.run()
        self._ctx.__enter__()
        self.fn, self.gpu, self.ref = b200_modal.handle, "B200 (Modal)", None
        self.lock, self.n, self.used_s = threading.Lock(), 0, 0.0
        print(f"tool server: Modal B200, budget {budget_minutes} min", flush=True)

    def call(self, kind, sol):
        with self.lock:
            if self.used_s >= self.budget_s:
                return dict(error=f"the round's B200 budget ({self.budget_s // 60:.0f} min) is used up; finish "
                                  "with what you have")
            self.n += 1
            rid = f"{kind}-{self.n:03d}"
        t0 = time.time()
        try:
            res = self.fn.remote(dict(id=rid, kind=kind, solution=sol))
        except Exception as e:
            res = dict(id=rid, kind=kind, error=f"Modal call failed: {type(e).__name__}: {e}")
        with self.lock:      # container time for this call (+5 s overhead); queueing behind other sessions is free
            self.used_s += float(res.get("seconds") or (time.time() - t0)) + 5
        if res.get("gpu") and "B200" not in res["gpu"]:
            res = dict(id=rid, kind=kind, error=f"wrong GPU {res['gpu']!r}; result discarded")
        return res

    def wait_alive(self, timeout_s):
        """Time the reference kernel (this also starts and warms the container); returns the GPU name."""
        sol = json.loads((archive.ROOT / self.ref_solution).read_text())
        res = self.call("test", sol)
        if not res.get("workloads"):
            print(f"reference run failed: {str(res.get('error') or res.get('console_tail'))[-1500:]}", flush=True)
            return None
        self.ref = dict(ref=Path(self.ref_solution).stem, gpu=res.get("gpu"), workloads=res["workloads"])
        self.gpu = res.get("gpu") or self.gpu
        return self.gpu

    def reference(self):
        return self.ref

    def stop(self):
        print(f"Modal B200 used for about {self.used_s / 60:.1f} min (~${self.used_s / 3600 * 6.25:.2f})", flush=True)
        self._ctx.__exit__(None, None, None)


def make_solution(args, name):
    """Tool arguments -> (solution JSON, problems)."""
    spec = dict(args.get("solution_spec") or {})
    spec = dict(spec.get("spec", spec))
    files = [f for f in args.get("files") or [] if isinstance(f, dict) and f.get("path")]
    problems = [] if files else ["no files given"]
    entry = spec.get("entry_point", "")
    if files and entry.split("::")[0] not in [f["path"] for f in files]:
        problems.append(f"entry_point {entry!r} is not one of the files")
    spec.setdefault("target_hardware", ["B200", "LOCAL"])
    spec.setdefault("destination_passing_style", True)
    sol = {"name": name, "definition": "038_flux_multi_head_rmsnorm_qk", "author": "solx-loop",
           "description": str(args.get("note") or "")[:500], "spec": spec,
           "sources": [{"path": f["path"], "content": f.get("content", "")} for f in files]}
    return sol, problems + reply.lint(sol)


def fmt_compile(res):
    lines = [f"compile_b200 for sm_100a ({res.get('seconds', '?')} s): {len(res.get('kernels') or [])} kernel(s)"]
    for k in res.get("kernels") or []:
        ops = ", ".join(f"{op} x{n}" for op, n in sorted((k.get("ops") or {}).items()))
        fit = k.get("resident_per_sm")
        lines.append(f"- {str(k.get('kernel'))[:60]}: {k.get('regs')} regs/thread, {k.get('smem')} B shared, "
                     f"{k.get('local')} B local (spills), {'%s CTAs/SM' % fit if fit else 'CTAs/SM unknown'}; "
                     f"memory ops in SASS: {ops or 'none found'}")
    if res.get("error"):
        lines.append(f"error:\n{res['error'][-3000:]}")
    return "\n".join(lines)


def fmt_test(res, ref, best, anc, k=None):
    wl = res.get("workloads") or []
    passed = sum(w["status"] == "PASSED" for w in wl)
    lines = [f"run_tests on {res.get('gpu', '?')} ({res.get('seconds', '?')} s): {passed}/{len(wl) or 16} workloads "
             "PASSED"]
    for w in wl:
        if w["status"] != "PASSED":
            lines.append(f"- {w['workload']}: {w['status']}, max abs err {w['max_abs_err']}, max rel err "
                         f"{w['max_rel_err']}\n  {w['log'][-800:]}")
    if res.get("console_tail"):
        lines.append(f"harness output (tail):\n{res['console_tail'][-3000:]}")
    if res.get("error"):
        lines.append(f"error: {res['error']}")
    if not wl or passed < len(wl):
        return "\n".join(lines)
    if not ref or not ref.get("workloads"):
        lines.append("(reference timing not available yet)")
        return "\n".join(lines)
    rt = {w["workload"]: w["latency_us"] for w in ref["workloads"] if w.get("latency_us")}
    rel = {}
    rows = []
    for w in wl:
        if w["latency_us"] and rt.get(w["workload"]):
            r = w["latency_us"] / rt[w["workload"]]
            rel.setdefault(archive.tokens(w["workload"]), []).append(r)
            rows.append(f"{w['workload']} {w['latency_us']:.1f} vs {rt[w['workload']]:.1f} ({100 * (r - 1):+.1f}%)")
    lines.append(f"latency vs reference {ref['ref']} on the same GPU (µs, this vs reference):\n  " + "; ".join(rows))
    by_band = {b: gmean([x for bs, v in rel.items() if band_of(bs) == b for x in v]) for b in "SML"}
    lines.append("change by band (geomean): " + ", ".join(f"{b} {100 * (v - 1):+.1f}%" for b, v in by_band.items() if v))
    bt = planner.b200_times(best)
    on_b200 = "B200" in str(res.get("gpu"))
    k = (k or {}) if on_b200 else {}
    pred = {bs: t * (gmean(rel[bs]) ** k.get(band_of(bs), 1.0) if rel.get(bs) else 1.0) for bs, t in bt.items()}
    if on_b200:
        lines.append(f"projected portal score: {planner.score(pred, anc):.4f} (current best {best['b200']['score']:.4f}). "
                     f"This is a real B200 with unlocked clocks; differences at S/M come out larger than on the portal, "
                     f"so the projection shrinks them (portal change ~ measured change ** k, k = "
                     f"{', '.join(f'{b} {v:.2f}' for b, v in k.items())}, fitted on kernels measured on both).")
    else:
        lines.append(f"projected B200 score (emulator: the reference's B200 times x this GPU's relative speed): "
                     f"{planner.score(pred, anc):.4f} (current best {best['b200']['score']:.4f}). Not valid for "
                     "B200-only paths, which run a fallback on this GPU.")
    return "\n".join(lines)


def predict(args, best, anc):
    bt = planner.b200_times(best)
    if args.get("fixed_us") is not None and args.get("tbs"):
        pred = {bs: args["fixed_us"] + planner.mb(bs) / args["tbs"] for bs in bt}
        how = f"every size = {args['fixed_us']} µs + bytes / {args['tbs']} TB/s"
    else:
        pct = {k: float(v) for k, v in (args.get("pct_change") or {}).items()}
        pred = {bs: t * (1 + pct.get(band_of(bs), 0) / 100) for bs, t in bt.items()}
        how = f"per-band change {pct} vs {best['id']}"
    base = planner.score(bt, anc)
    worth = {b: planner.score({bs: t * (0.9 if band_of(bs) == b else 1) for bs, t in bt.items()}, anc) - base
             for b in "SML"}
    gm = {b: gmean([t for bs, t in pred.items() if band_of(bs) == b]) for b in "SML"}
    return (f"predicted B200 score {planner.score(pred, anc):.4f} for {how} (this model gives the current best "
            f"{base:.4f}; its portal score is {best['b200']['score']:.4f}). "
            f"Predicted B200 geomean µs: S {gm['S']:.1f}, M {gm['M']:.1f}, L {gm['L']:.1f}. "
            f"Worth of a 10% latency cut alone: S {worth['S']:+.4f}, M {worth['M']:+.4f}, L {worth['L']:+.4f}.")


def with_tail_cache(messages):
    """Copy of messages with a cache breakpoint on the last one, so the next turn re-reads the conversation cheaply."""
    out = list(messages)
    last = dict(out[-1])
    if isinstance(last.get("content"), str) and last["role"] in ("tool", "user"):
        last["content"] = [{"type": "text", "text": last["content"], "cache_control": CACHE}]
        out[-1] = last
    return out


class Session:
    def __init__(self, name, task, static, dyn, server, best, anc, arc, a, d):
        self.name, self.task, self.server, self.best, self.anc, self.arc, self.a = name, task, server, best, anc, arc, a
        self.messages = [llm.system_message(static),
                         {"role": "user", "content": [{"type": "text", "text": dyn, "cache_control": CACHE}]}]
        self.gpu_calls, self.drafts = 0, 0
        self.k = planner.rented_exponents(arc, best)
        self.totals = dict(prompt_tokens=0, completion_tokens=0, cached_tokens=0, reasoning_tokens=0, cost_usd=0.0,
                           seconds=0.0)
        self.log_json, self.log_md = d / f"{name}.jsonl", d / f"{name}.md"
        self.md = [f"# Design session {name}\n\nTask: {json.dumps(task)}\n"]

    def note(self, text):
        self.md.append(text)
        self.log_md.write_text("\n".join(self.md))

    def tool(self, call):
        fn = call["function"]["name"]
        try:
            args = json.loads(call["function"].get("arguments") or "{}")
        except json.JSONDecodeError as e:
            return f"invalid JSON arguments: {e}"
        if fn == "get_kernel":
            rec = self.arc.get(args.get("id", ""))
            return prompts.parent_block(rec) if rec and rec.get("solution") else \
                f"unknown id {args.get('id')!r}; ids: {', '.join(sorted(self.arc))}"
        if fn == "predict_score":
            return predict(args, self.best, self.anc)
        if fn == "read_example":
            files = {f.name: f for f in sorted(EXAMPLES.glob("*")) if f.is_file()}
            name = args.get("name") or ""
            if name in files:
                return files[name].read_text()
            return "examples: " + ", ".join(f"{n} ({len(f.read_text().splitlines())} lines)" for n, f in files.items())
        if fn not in ("compile_b200", "run_tests"):
            return f"unknown tool {fn}"
        if self.gpu_calls >= self.a.gpu_calls:
            return "GPU budget used up: give your final answer now."
        self.drafts += 1
        sol, problems = make_solution(args, f"{self.name}-d{self.drafts}")
        if problems:
            return "not sent to the GPU, fix these first:\n- " + "\n- ".join(problems)
        self.gpu_calls += 1
        self.note(f"\n### {fn} (draft {self.drafts}): {args.get('note', '')}\n")
        if fn == "compile_b200":
            return fmt_compile(self.server.call("compile", sol))
        return fmt_test(self.server.call("test", sol), self.server.reference(), self.best, self.anc, self.k)

    def turn(self, n, tool_choice=None):
        effort = self.a.effort if n == 0 else self.a.turn_effort
        for attempt in range(3):
            try:
                return llm.complete(with_tail_cache(self.messages), tools=TOOLS, tool_choice=tool_choice,
                                    model=self.a.model, max_tokens=self.a.max_tokens if n == 0 else 32000,
                                    effort=effort, tag=f"{self.name}/t{n}")
            except Exception as e:                       # network or provider error: back off and retry
                print(f"  {self.name} turn {n}: {e}; retrying", flush=True)
                time.sleep(30 * (attempt + 1))
        raise RuntimeError(f"{self.name}: three failed attempts at turn {n}")

    def run(self, first_turn_done=None):
        text, nudged, info = "", False, {}
        for n in range(self.a.turns):
            last_chance = n == self.a.turns - 1 or self.totals["cost_usd"] >= self.a.session_budget
            try:
                m, info = self.turn(n, tool_choice="none" if last_chance else None)
            finally:
                if first_turn_done:
                    first_turn_done.set()                 # release the other sessions even if this turn failed
            for k in self.totals:
                self.totals[k] += info.get(k) or 0
            self.messages.append(m)
            text = m.get("content") or ""
            calls = m.get("tool_calls") or []
            self.note(f"\n## Turn {n} ({info.get('completion_tokens')} tokens out, ${info.get('cost_usd')})\n\n{text}\n")
            if not calls:
                if reply.candidates(text) or nudged:
                    break
                nudged = True
                self.messages.append({"role": "user", "content": "No candidate in the OUTPUT CONTRACT format was found "
                                      "in your reply. Reply now with exactly one candidate in that format."})
                continue
            for c in calls:
                result = self.tool(c)
                self.note(f"**{c['function']['name']}** →\n```\n{result}\n```\n")
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
            left = self.a.turns - 1 - n
            if left <= 2 or self.gpu_calls >= self.a.gpu_calls:
                self.messages[-1]["content"] += (f"\n\n[session: {left} turn(s) and {self.a.gpu_calls - self.gpu_calls}"
                                                 " GPU call(s) left; give your final answer soon]")
            self.log_json.write_text("\n".join(json.dumps(x) for x in self.messages[1:]))
        self.log_json.write_text("\n".join(json.dumps(x) for x in self.messages[1:]))
        info = dict(info, **{k: round(v, 4) if isinstance(v, float) else v for k, v in self.totals.items()},
                    turns=n + 1, gpu_calls=self.gpu_calls)
        self.note(f"\n---\nsession total: {info['turns']} turns, {self.gpu_calls} GPU calls, ${info['cost_usd']:.2f}\n")
        return text, info


def run_sessions(round_id, jobs, static, a, slurm, sessions_dir):
    """jobs = [(i, task, dynamic prompt, name)] -> {i: (final text, info)}; one shared GPU tool server."""
    arc = archive.load()
    best, anc = planner.best_kernel(arc), planner.anchors()
    sessions_dir.mkdir(parents=True, exist_ok=True)
    if a.tool_gpu == "B200":
        server = ModalB200(round_id, best["solution"], budget_minutes=a.b200_minutes)
    else:
        server = ToolServer(round_id, a.tool_gpu, slurm, best["solution"])
    try:
        # No LLM turns until a GPU is ours: a session that cannot test would just be the one-shot mode, at more cost.
        gpu = server.wait_alive(a.gpu_wait * 60)
        if not gpu:
            raise SystemExit(f"no GPU for the tool server within {a.gpu_wait} minutes; nothing was spent on the LLM. "
                             "Try again later, or another --tool-gpu.")
        print(f"tool server up on {gpu}", flush=True)
        static = static + "\n\n" + PROTOCOL.format(gpu=gpu, gpu_calls=a.gpu_calls, turns=a.turns)
        sess = {i: Session(name, t, static, dyn, server, best, anc, arc, a, sessions_dir) for i, t, dyn, name in jobs}
        warm = threading.Event()
        with ThreadPoolExecutor(max_workers=len(sess)) as pool:
            first = min(sess)
            futs = {first: pool.submit(sess[first].run, warm)}
            warm.wait()                                   # the first turn writes the prompt cache for the rest
            futs.update({i: pool.submit(s.run) for i, s in sess.items() if i != first})
            results = {}
            for i, f in futs.items():
                try:
                    results[i] = f.result()
                except Exception as e:
                    print(f"  session {sess[i].name} failed: {e}", flush=True)
                    results[i] = ("", dict(sess[i].totals, finish="error", model=a.model))
        return results
    finally:
        server.stop()
