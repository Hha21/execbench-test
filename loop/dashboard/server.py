#!/usr/bin/env python3
"""Local live dashboard for the SOL-ExecBench optimisation loop. Standard library only, binds 127.0.0.1.

  .venv/bin/python loop/dashboard/server.py [--port 8765]

Read-only over problems/, poc/results/ and process info (/proc); the only write action is POST /api/upload, which
saves a portal page into html_results/ and runs `.venv/bin/python poc/ingest_portal.py <file>`.
It never starts the loop, calls claude, Modal or the network.
"""
import argparse
import csv
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STATIC = HERE / "static"
PROBLEMS = ROOT / "problems"
RESULTS = ROOT / "poc" / "results"
LEADERBOARD = RESULTS / "leaderboards" / "leaderboards_b200_v1.1.csv"
HTML_RESULTS = ROOT / "html_results"
PY = ROOT / ".venv" / "bin" / "python"
if not PY.exists():
    PY = Path(sys.executable)
MAX_UPLOAD = 20 * 1024 * 1024
RUNNING_WINDOW = 300          # seconds without file growth before a session without a result counts as stopped
SEG = re.compile(r"^[\w.#+\-]+$")
EXTRA = {"dir": None}         # --extra-session-dir: a fake problem "_extra" whose rounds/ lives there
ING = {"active": 0, "last": 0.0}
ING_LOCK = threading.Lock()


# ------------------------------------------------------------------------------------------------- helpers
def seg(s):
    if not s or not SEG.match(s) or s in (".", ".."):
        raise KeyError("bad path segment")
    return s


def read_text(p, default=""):
    try:
        return Path(p).read_text(errors="replace")
    except OSError:
        return default


def mtime(p):
    try:
        return os.stat(p).st_mtime
    except OSError:
        return 0.0


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def problem_dir(name):
    seg(name)
    if name == "_extra" and EXTRA["dir"]:
        return EXTRA["dir"]
    d = PROBLEMS / name
    if not d.is_dir():
        raise KeyError("no such problem")
    return d


def problem_names():
    names = sorted(d.name for d in PROBLEMS.iterdir() if d.is_dir() and (d / "card.md").exists()) if PROBLEMS.is_dir() else []
    if EXTRA["dir"]:
        names.append("_extra")
    return names


def round_key(n):
    m = re.fullmatch(r"r(\d+)", n)
    return (0, int(m.group(1)), "") if m else (1, 0, n)


def round_names(pdir):
    rd = pdir / "rounds"
    if not rd.is_dir():
        return []
    return sorted((d.name for d in rd.iterdir() if d.is_dir()), key=round_key)


def latest_round(pdir):
    rs = [r for r in round_names(pdir) if re.fullmatch(r"r\d+", r)]
    return rs[-1] if rs else (round_names(pdir) or [None])[-1]


# ------------------------------------------------------------------------------------------------- summaries
_summ = {}
_summ_lock = {}


def _summary_stamp(pdir):
    files = [pdir / f for f in ("archive.json", "ledger.yaml", "card.md", "sol.yaml")]
    files += sorted(RESULTS.glob("*.csv"))
    return tuple(mtime(f) for f in files)


def summary(name):
    """Authoritative per-kernel numbers from loop modules, via a subprocess per problem, cached by file mtimes."""
    if name == "_extra":
        return None
    pdir = problem_dir(name)
    stamp = _summary_stamp(pdir)
    lock = _summ_lock.setdefault(name, threading.Lock())
    with lock:
        hit = _summ.get(name)
        if hit and hit[0] == stamp:
            return hit[1]
        env = dict(os.environ, SOLX_PROBLEM=name)
        try:
            r = subprocess.run([str(PY), str(HERE / "summary.py"), name], cwd=ROOT, env=env, capture_output=True,
                               text=True, timeout=90)
            data = json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else {"error": r.stderr[-2000:]}
        except Exception as e:  # noqa: BLE001
            data = {"error": str(e)}
        _summ[name] = (stamp, data)
        return data


_lb = {"stamp": None, "rows": {}}


def leaderboard():
    st = mtime(LEADERBOARD)
    if _lb["stamp"] != st:
        rows = {}
        try:
            with open(LEADERBOARD, newline="") as f:
                for r in csv.DictReader(f):
                    rows[r["name"]] = r
        except OSError:
            pass
        _lb.update(stamp=st, rows=rows)
    return _lb["rows"]


def rank_info(name, score):
    r = leaderboard().get(name)
    if not r:
        return None
    out = {"n_ranked": r.get("n_ranked")}
    thr = []
    for k in ("s1", "s2", "s3", "s5", "s10"):
        try:
            out[k] = float(r[k])
            thr.append((k, float(r[k])))
        except (KeyError, ValueError, TypeError):
            out[k] = None
    out["top5_users"] = r.get("top5_users")
    if score is not None:
        out["above"] = sum(1 for k, v in thr if k in ("s1", "s2", "s3", "s5") and v > score)
        out["beats"] = [k for k, v in thr if v <= score]
    return out


# ------------------------------------------------------------------------------------------------- processes
def processes():
    """Running `loop/round.py` processes, read from /proc (no commands executed)."""
    out = []
    me = os.getpid()
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == me:
            continue
        try:
            argv = p.joinpath("cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        argv = [a.decode(errors="replace") for a in argv if a]
        if len(argv) < 2 or not os.path.basename(argv[0]).startswith("python"):
            continue
        idx = next((i for i, a in enumerate(argv[1:], 1) if a.endswith("round.py")), None)
        if idx is None:
            continue
        args = argv[idx + 1:]
        sub = next((a for a in args if not a.startswith("-")), None)

        def opt(flag):
            for i, a in enumerate(args):
                if a == flag and i + 1 < len(args):
                    return args[i + 1]
                if a.startswith(flag + "="):
                    return a.split("=", 1)[1]
            return None
        prob = opt("--problem")
        if not prob:
            try:
                env = p.joinpath("environ").read_bytes().split(b"\0")
                prob = next((e.decode().split("=", 1)[1] for e in env if e.startswith(b"SOLX_PROBLEM=")), None)
            except OSError:
                pass
        out.append({"pid": int(p.name), "sub": sub, "problem": prob, "round": opt("--round"), "args": " ".join(args)})
    return out


_resolved = {}


def resolve_problem(arg):
    """A process's --problem value ('L1/030', 'FlashInfer-Bench/009_gemm_n5120_k2048', a folder name, or none) -> the
    problem folder name, resolved like the loop does (loop/problem.find); no --problem means the loop's default."""
    if arg in _resolved:
        return _resolved[arg]
    name = None
    try:
        if str(ROOT / "loop") not in sys.path:
            sys.path.insert(0, str(ROOT / "loop"))
        import problem as loop_problem
        name = loop_problem.find(arg).name if arg else loop_problem.DEFAULT
    except BaseException:  # noqa: BLE001  (find() raises SystemExit when a name is unknown or ambiguous)
        name = None
    _resolved[arg] = name
    return name


def proc_matches(procs, problem):
    """Processes running for this problem folder (unknown or ambiguous --problem values match nothing)."""
    return [x for x in procs if resolve_problem(x["problem"]) == problem]


# ------------------------------------------------------------------------------------------------- sessions
KINDS = ("sessions", "lead", "research")


def session_files(rdir):
    """[(dir, name)] of every session-like file group in a round folder."""
    found = {}
    for kind in KINDS:
        d = rdir / kind
        if not d.is_dir():
            continue
        for f in d.iterdir():
            m = re.fullmatch(r"(.+?)\.(tools\.jsonl|stream\.jsonl|claude\.json)", f.name)
            if m:
                found[(kind, m.group(1))] = True
    return sorted(found, key=lambda t: (KINDS.index(t[0]), t[1]))


def start_time(rdir, kind, name):
    m = re.match(r"(\d{8})-(\d{6})", name)
    if m:
        try:
            return time.mktime(time.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"))
        except ValueError:
            pass
    d = rdir / kind
    for suffix in (".mcp.json", ".stream.jsonl", ".tools.jsonl", ".claude.json"):
        t = d / (name + suffix)
        if t.exists():
            return t.stat().st_ctime if suffix == ".stream.jsonl" else mtime(t)
    return time.time()


def last_tool(rdir, kind, name):
    """Latest tool name used by a session (stream file preferred, tools.jsonl fallback)."""
    d = rdir / kind
    sp, tp = d / (name + ".stream.jsonl"), d / (name + ".tools.jsonl")
    try:
        if sp.exists():
            tail = sp.read_bytes()[-200000:].decode(errors="replace").splitlines()
            for line in reversed(tail):
                if '"tool_use"' in line:
                    try:
                        for b in json.loads(line).get("message", {}).get("content", []):
                            if b.get("type") == "tool_use":
                                last = short_tool(b.get("name"))
                        return last
                    except (ValueError, UnboundLocalError):
                        continue
        if tp.exists():
            tail = tp.read_bytes()[-400000:].decode(errors="replace").splitlines()
            for line in reversed(tail):
                m = re.match(r'\{"tool": "([^"]+)"', line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return None


def short_tool(n):
    return re.sub(r"^mcp__solx__", "", n or "")


def session_info(pdir, rname, kind, name, procs=()):
    rdir = pdir / "rounds" / rname
    d = rdir / kind
    cj = read_json(d / (name + ".claude.json"))
    sp, tp = d / (name + ".stream.jsonl"), d / (name + ".tools.jsonl")
    last = max(mtime(sp), mtime(tp), mtime(d / (name + ".claude.json")))
    info = {"dir": kind, "name": name, "has_stream": sp.exists(), "started": start_time(rdir, kind, name),
            "model": None, "turns": None, "cost": None, "seconds": None, "tool_calls": 0}
    if tp.exists():
        try:
            info["tool_calls"] = sum(1 for _ in open(tp, "rb"))
        except OSError:
            pass
    finished = False
    if cj:
        info["turns"] = cj.get("num_turns")
        info["cost"] = cj.get("total_cost_usd")
        info["seconds"] = cj.get("seconds") or (cj.get("duration_ms") or 0) / 1000 or None
        status = "error" if cj.get("is_error") or str(cj.get("subtype", "")).startswith("error") else "done"
        finished = True
    elif sp.exists():
        res = stream_result(sp)
        if res:
            info.update(turns=res.get("num_turns"), cost=res.get("total_cost_usd"),
                        seconds=(res.get("duration_ms") or 0) / 1000 or None)
            status = "error" if res.get("is_error") else "done"
            finished = True
    if not finished:
        proc_here = any(x["sub"] in ("auto", "lead", "research", "test") and (x["round"] in (None, rname)) for x in procs)
        recent = time.time() - last < RUNNING_WINDOW
        status = "running" if recent or (proc_here and time.time() - last < 3600) else "stopped"
        info["last_tool"] = last_tool(rdir, kind, name)
    info["status"] = status
    info["last_activity"] = last
    if info["seconds"] is None:
        info["elapsed"] = (last if finished or status == "stopped" else time.time()) - info["started"]
    else:
        info["elapsed"] = info["seconds"]
    return info


def read_tail(p, n):
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode(errors="replace")
    except OSError:
        return ""


def stream_result(p):
    for line in reversed(read_tail(p, 300000).splitlines()):
        if '"result"' in line and line.lstrip().startswith("{"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("type") == "result":
                return d
    return None


# ------------------------------------------------------------------------------------------------- API builders
def api_problems():
    procs = processes()
    names = problem_names()
    res = {}
    threads = []

    def work(n):
        res[n] = summary(n)
    for n in names:
        t = threading.Thread(target=work, args=(n,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    out = []
    for n in names:
        pdir = problem_dir(n)
        s = res.get(n) or {}
        best, score = s.get("best"), s.get("best_score")
        lr = latest_round(pdir)
        mine = proc_matches(procs, n)
        out.append({"name": n, "level": s.get("level"), "best": best, "best_score": score,
                    "n_kernels": len(s.get("kernels", [])) if s else 0, "latest_round": lr,
                    "rounds": len(round_names(pdir)), "running": [{k: x[k] for k in ("pid", "sub", "round")} for x in mine],
                    "rank": rank_info(n, score), "error": s.get("error"),
                    "activity": max([mtime(pdir / "archive.json")] + [mtime(pdir / "rounds" / lr)] if lr else [0])})
    return {"problems": out, "processes": procs, "ingesting": ING["active"] > 0}


def parse_ledger(text):
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def api_problem(name):
    pdir = problem_dir(name)
    s = summary(name) or {}
    procs = processes()
    rounds = []
    for r in round_names(pdir):
        rd = pdir / "rounds" / r
        plan = read_json(rd / "plan.json") or {}
        rounds.append({"name": r, "tasks": len(plan.get("tasks", [])), "has_plan": bool(plan),
                       "sessions": len(session_files(rd)),
                       "candidates": len(list((rd / "candidates").glob("*.json"))) if (rd / "candidates").is_dir() else 0,
                       "lead": (rd / "lead" / "reply.md").exists(), "research": (rd / "research" / "reply.md").exists(),
                       "mtime": mtime(rd), "best": plan.get("best"), "best_score": plan.get("best_score")})
    led = parse_ledger(read_text(pdir / "ledger.yaml")) if (pdir / "ledger.yaml").exists() else {}
    return {"name": name, "card": read_text(pdir / "card.md"), "ledger": led, "summary": s, "rounds": rounds,
            "rank": rank_info(name, s.get("best_score")),
            "running": [x for x in proc_matches(procs, name)]}


def candidate_info(rdir, kid, scores):
    t = rdir / "results" / "timing" / "B200r" / (kid + ".jsonl")
    n = ok = 0
    bad = []
    if t.exists():
        for line in read_text(t).splitlines():
            try:
                ev = json.loads(line).get("evaluation", {})
            except ValueError:
                continue
            n += 1
            if ev.get("status") == "PASSED":
                ok += 1
            else:
                bad.append(ev.get("status"))
    sc = scores.get(kid) or {}
    return {"id": kid, "tested": n, "passed": ok, "failed": n - ok, "fail_status": sorted(set(map(str, bad))),
            "status": ("untested" if not n else "passed" if ok == n else "failed"),
            "arch_status": sc.get("status"), "score": sc.get("score"), "geomean_us": sc.get("geomean_us"),
            "submission": sc.get("submission"), "has_card": (rdir / "candidates" / (kid + ".card.yaml")).exists(),
            "portal_json": (rdir / "portal" / (kid + ".json")).exists()}


def api_round(pname, rname):
    pdir = problem_dir(pname)
    rd = pdir / "rounds" / seg(rname)
    if not rd.is_dir():
        raise KeyError("no such round")
    procs = proc_matches(processes(), pname)
    s = summary(pname) or {}
    scores = {k["id"]: k for k in s.get("kernels", [])}
    cands = sorted(p.name[:-5] for p in (rd / "candidates").glob("*.json")) if (rd / "candidates").is_dir() else []
    rej = sorted(p.name for p in (rd / "rejected").iterdir()) if (rd / "rejected").is_dir() else []
    sess = [session_info(pdir, rname, k, n, procs) for k, n in session_files(rd)]
    return {"problem": pname, "round": rname, "plan": read_json(rd / "plan.json"),
            "lead_reply": read_text(rd / "lead" / "reply.md") or None,
            "research_reply": read_text(rd / "research" / "reply.md") or None,
            "sessions": sess, "candidates": [candidate_info(rd, c, scores) for c in cands], "rejected": rej,
            "auto_log_tail": read_tail(rd / "auto.log", 6000), "running": procs}


def api_pipeline(pname=None):
    allp = problem_names()
    procs = processes()
    if not pname or pname not in allp:
        busy = next((x["problem"] for x in procs if x["problem"] in allp), None)
        pname = busy or ("_extra" if EXTRA["dir"] else
                         max(allp, key=lambda p: mtime(problem_dir(p) / "archive.json")))
    pdir = problem_dir(pname)
    mine = proc_matches(procs, pname)
    rname = next((x["round"] for x in mine if x["round"] and (pdir / "rounds" / x["round"]).is_dir()), None) or latest_round(pdir)
    nodes, links = [], []
    subs = {x["sub"] for x in mine}
    now = time.time()

    def node(id, label, state, col, row=0, note=None):
        nodes.append({"id": id, "label": label, "state": state, "col": col, "row": row, "note": note})

    rd = pdir / "rounds" / rname if rname else None
    sessions = []
    if rd and rd.is_dir():
        for k, n in session_files(rd):
            if k == "sessions":
                sessions.append(session_info(pdir, rname, k, n, mine))
    run = [s for s in sessions if s["status"] == "running"]
    cand_dir = rd / "candidates" if rd else None
    has_c = bool(cand_dir and cand_dir.is_dir() and any(cand_dir.glob("*.json")))
    portal_dir = rd / "portal" if rd else None
    has_portal = bool(portal_dir and portal_dir.is_dir() and any(portal_dir.glob("*.json")))
    ing_recent = ING["active"] > 0 or now - max(ING["last"], mtime(RESULTS / "b200_portal.csv")) < 90
    research_done = bool(rd and (rd / "research" / "reply.md").exists())
    lead_done = bool(rd and (rd / "lead" / "reply.md").exists())

    def st(active, done):
        return "active" if active else "done" if done else "idle"
    node("research", "research", st("research" in subs, research_done), 0, 0, "round " + str(rname))
    node("lead", "lead", st("lead" in subs, lead_done), 1, 0)
    ids = []
    for i, s in enumerate(sessions):
        nid = "s%d" % i
        ids.append(nid)
        short = re.sub(r"-all$", "", re.sub(r"^\d+-\d+_(\d+)-", r"\1 ", s["name"]))
        node(nid, short[:20], "active" if s["status"] == "running" else "done" if s["status"] == "done" else "error"
             if s["status"] == "error" else "idle", 2, i, s["status"])
        nodes[-1]["ref"] = "%s/%s" % (s["dir"], s["name"])
    if not sessions:
        node("s0", "sessions", st("auto" in subs and not (research_done and not lead_done), False), 2, 0)
        ids = ["s0"]
    cur = {}
    for s in run:
        if s.get("last_tool"):
            cur[s["last_tool"]] = True
    tool_def = [("compile", ("compile_b200",)), ("test", ("run_tests",)), ("probe", ("probe_b200",)),
                ("predict", ("predict_score", "get_kernel", "read_example"))]
    tids = []
    for i, (lab, names) in enumerate(tool_def):
        on = any(n in cur for n in names)
        node("t_" + lab, lab, "active" if on else "done" if sessions and not run else "idle", 3, i)
        tids.append("t_" + lab)
    node("candidates", "candidates", st("test" in subs, has_c), 4, 0, str(len(list(cand_dir.glob("*.json")))) if has_c else None)
    node("shortlist", "shortlist", st("shortlist" in subs, has_portal), 5, 0)
    node("portal", "portal", st(False, has_portal), 6, 0, "manual")
    node("ingest", "ingest", st(ing_recent, has_portal), 7, 0)
    links += [("research", "lead")] + [("lead", i) for i in ids]
    for i in ids:
        links += [(i, t) for t in tids]
    links += [(t, "candidates") for t in tids] + [("candidates", "shortlist"), ("shortlist", "portal"),
                                                  ("portal", "ingest"), ("ingest", "lead")]
    return {"problem": pname, "round": rname, "nodes": nodes, "links": links, "processes": mine,
            "running_sessions": [s["name"] for s in run]}


# ------------------------------------------------------------------------------------------------- live tail
class Tail:
    """Incremental line reader by byte offset; a partial last line is held back until its newline arrives."""

    def __init__(self, path):
        self.path, self.off, self.buf = Path(path), 0, b""

    def read(self):
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self.off:      # truncated
            self.off, self.buf = 0, b""
        if size == self.off:
            return []
        with open(self.path, "rb") as f:
            f.seek(self.off)
            chunk = f.read(size - self.off)
        self.off += len(chunk)
        self.buf += chunk
        *lines, self.buf = self.buf.split(b"\n")
        out = []
        for ln in lines:
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except ValueError:
                    pass
        return out


def block_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return json.dumps(c)


def normalise_stream(d):
    t = d.get("type")
    if t == "system" and d.get("subtype") == "init":
        return [{"k": "init", "model": d.get("model"), "tools": len(d.get("tools") or [])}]
    if t == "assistant":
        msg = d.get("message") or {}
        out = []
        for b in msg.get("content") or []:
            bt = b.get("type")
            if bt == "text":
                out.append({"k": "text", "text": b.get("text", ""), "msg": msg.get("id")})
            elif bt == "thinking":
                out.append({"k": "thinking", "text": b.get("thinking", ""), "msg": msg.get("id")})
            elif bt == "tool_use":
                out.append({"k": "call", "id": b.get("id"), "name": short_tool(b.get("name")), "input": b.get("input"),
                            "msg": msg.get("id")})
        return out
    if t == "user":
        content = (d.get("message") or {}).get("content")
        out = []
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    out.append({"k": "result", "id": b.get("tool_use_id"), "text": block_text(b.get("content")),
                                "is_error": bool(b.get("is_error"))})
        return out
    if t == "result":
        return [{"k": "final", "turns": d.get("num_turns"), "cost": d.get("total_cost_usd"),
                 "duration_ms": d.get("duration_ms"), "is_error": bool(d.get("is_error")), "text": d.get("result"),
                 "subtype": d.get("subtype")}]
    return []


class SessionFeed:
    def __init__(self, pdir, rname, kind, name):
        self.rdir = pdir / "rounds" / rname
        self.kind, self.name = kind, name
        self.d = self.rdir / kind
        self.mode = None
        self.tail = None
        self.n = 0
        self.stats = {"turns": 0, "calls": {}, "gpu_calls": 0, "probes": 0, "cost": None, "final": False}
        self.msgs = set()
        self.reply_sent = False
        self.final_sent = False
        self.started = start_time(self.rdir, kind, name)

    def _pick(self):
        sp, tp = self.d / (self.name + ".stream.jsonl"), self.d / (self.name + ".tools.jsonl")
        want = "stream" if sp.exists() else "tools" if tp.exists() else None
        if want and want != self.mode:
            self.mode = want
            self.tail = Tail(sp if want == "stream" else tp)

    def poll(self):
        """New normalised events since the last call."""
        if self.mode != "stream":
            self._pick()
        if not self.tail:
            return []
        out = []
        for raw in self.tail.read():
            if self.mode == "stream":
                evs = normalise_stream(raw)
            else:
                self.n += 1
                cid = "t%d" % self.n
                evs = [{"k": "call", "id": cid, "name": raw.get("tool"), "input": raw.get("args"), "msg": cid},
                       {"k": "result", "id": cid, "text": block_text(raw.get("result")), "is_error": False}]
            for e in evs:
                self._count(e)
            out += evs
        if self.mode == "tools":
            cj = read_json(self.d / (self.name + ".claude.json"))
            rp = self.rdir / "replies" / (self.name + ".reply.md")
            if rp.exists() and not self.reply_sent:
                self.reply_sent = True
                out.append({"k": "text", "text": read_text(rp), "msg": "reply"})
            if cj and not self.final_sent:
                self.final_sent = True
                e = {"k": "final", "turns": cj.get("num_turns"), "cost": cj.get("total_cost_usd"),
                     "duration_ms": (cj.get("seconds") or 0) * 1000 or cj.get("duration_ms"),
                     "is_error": bool(cj.get("is_error")), "text": None, "subtype": cj.get("subtype")}
                self._count(e)
                out.append(e)
        return out

    def _count(self, e):
        s = self.stats
        if e["k"] == "call":
            s["calls"][e["name"]] = s["calls"].get(e["name"], 0) + 1
            if e["name"] in ("compile_b200", "run_tests"):
                s["gpu_calls"] += 1
            if e["name"] == "probe_b200":
                s["probes"] += 1
        if e["k"] in ("call", "text", "thinking") and e.get("msg") and e["msg"] not in self.msgs:
            self.msgs.add(e["msg"])
            s["turns"] += 1
        if e["k"] == "final":
            s.update(final=True, cost=e.get("cost"), is_error=e.get("is_error"))
            if e.get("turns") is not None:
                s["turns"] = e["turns"]
            s["duration_s"] = (e.get("duration_ms") or 0) / 1000 or None

    def stat_event(self):
        s = dict(self.stats)
        s["mode"] = self.mode
        s["started"] = self.started
        s["elapsed"] = s.get("duration_s") or max(0.0, time.time() - self.started)
        if not self.stats["final"]:
            s["live"] = True
        return s


# ------------------------------------------------------------------------------------------------- upload
def sanitise_name(raw):
    base = os.path.basename((raw or "").replace("\\", "/"))
    base = re.sub(r"[^\w.#+ \-]", "_", base).strip(". ")
    base = base.lstrip("-")
    if not re.search(r"\.html?$", base, re.I):
        raise ValueError("only .html or .htm portal pages are accepted")
    return base[:120]


def do_ingest(raw_name, data):
    name = sanitise_name(raw_name)
    HTML_RESULTS.mkdir(exist_ok=True)
    dest = HTML_RESULTS / name
    if dest.exists() and dest.read_bytes() != data:
        stem, suf = dest.stem, dest.suffix
        i = 1
        while (HTML_RESULTS / f"{stem}_{i}{suf}").exists():
            i += 1
        dest = HTML_RESULTS / f"{stem}_{i}{suf}"
    dest.write_bytes(data)
    rel = os.path.relpath(dest, ROOT)
    with ING_LOCK:
        ING["active"] += 1
        try:
            r = subprocess.run([str(PY), "poc/ingest_portal.py", rel], cwd=ROOT, capture_output=True, text=True, timeout=180)
        finally:
            ING["active"] -= 1
            ING["last"] = time.time()
    return {"ok": r.returncode == 0, "returncode": r.returncode, "saved": rel,
            "command": ".venv/bin/python poc/ingest_portal.py " + rel, "stdout": r.stdout, "stderr": r.stderr}


# ------------------------------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "solx-dashboard"
    clients = 0

    def log_message(self, fmt, *a):
        if os.environ.get("DASH_VERBOSE"):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % a))

    # -- plumbing
    def host_ok(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost", "[::1]"):
            return False
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost"):
            return False
        return True

    def send_json(self, obj, code=200):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path):
        try:
            body = path.read_bytes()
        except OSError:
            return self.send_json({"error": "not found"}, 404)
        ct = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ct + ("; charset=utf-8" if ct.startswith("text/") or "javascript" in ct else ""))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.host_ok():
            return self.send_json({"error": "forbidden host"}, 403)
        u = urlparse(self.path)
        parts = [unquote(p) for p in u.path.strip("/").split("/") if p]
        q = parse_qs(u.query)
        try:
            if not parts:
                return self.send_file(STATIC / "index.html")
            if parts[0] == "static" and len(parts) == 2 and SEG.match(parts[1]):
                return self.send_file(STATIC / parts[1])
            if parts[0] != "api":
                return self.send_json({"error": "not found"}, 404)
            a = parts[1:]
            if a == ["problems"]:
                return self.send_json(api_problems())
            if a == ["processes"]:
                return self.send_json({"processes": processes()})
            if a == ["pipeline"]:
                return self.send_json(api_pipeline((q.get("problem") or [None])[0]))
            if len(a) == 2 and a[0] == "problem":
                return self.send_json(api_problem(a[1]))
            if len(a) == 4 and a[0] == "problem" and a[2] == "round":
                return self.send_json(api_round(a[1], a[3]))
            if len(a) == 5 and a[0] == "stream":
                return self.stream(a[1], a[2], a[3], a[4])
            if len(a) == 5 and a[0] == "session":
                return self.session_snapshot(a[1], a[2], a[3], a[4])
            return self.send_json({"error": "not found"}, 404)
        except KeyError as e:
            return self.send_json({"error": str(e)}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            return self.send_json({"error": repr(e)}, 500)

    def do_POST(self):
        if not self.host_ok():
            return self.send_json({"error": "forbidden host"}, 403)
        u = urlparse(self.path)
        if u.path != "/api/upload":
            return self.send_json({"error": "not found"}, 404)
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n <= 0:
            return self.send_json({"error": "empty body"}, 400)
        if n > MAX_UPLOAD:
            self.close_connection = True
            return self.send_json({"error": "file larger than 20 MB"}, 413)
        data = self.rfile.read(n)
        name = (parse_qs(u.query).get("name") or [self.headers.get("X-Filename") or ""])[0]
        try:
            return self.send_json(do_ingest(name, data))
        except ValueError as e:
            return self.send_json({"error": str(e)}, 400)
        except subprocess.TimeoutExpired:
            return self.send_json({"error": "ingest timed out"}, 504)

    # -- sessions
    def session_snapshot(self, p, r, kind, name):
        pdir = problem_dir(p)
        if seg(kind) not in KINDS:
            raise KeyError("bad kind")
        feed = SessionFeed(pdir, seg(r), kind, seg(name))
        events = feed.poll()
        self.send_json({"events": events, "stats": feed.stat_event(),
                        "info": session_info(pdir, r, kind, name, proc_matches(processes(), p))})

    def stream(self, p, r, kind, name):
        """Server-Sent Events: replays the session from the start, then follows it as files grow."""
        pdir = problem_dir(p)
        if seg(kind) not in KINDS:
            raise KeyError("bad kind")
        feed = SessionFeed(pdir, seg(r), kind, seg(name))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def emit(event, obj):
            self.wfile.write(("event: %s\ndata: %s\n\n" % (event, json.dumps(obj, default=str))).encode())
            self.wfile.write(b"")
        Handler.clients += 1
        try:
            idle = 0
            while True:
                evs = feed.poll()
                for e in evs:
                    emit("ev", e)
                emit("stat", feed.stat_event()) if evs or idle % 5 == 0 else self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
                if feed.stats["final"] and not evs:
                    emit("end", {"final": True})
                    self.wfile.flush()
                    break
                idle += 1 if not evs else 0
                time.sleep(1.0)
        except (BrokenPipeError, ConnectionResetError, ValueError):
            pass
        finally:
            Handler.clients -= 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--extra-session-dir", type=Path,
                    help="testing: expose DIR (containing rounds/<round>/sessions/...) as an extra problem '_extra'")
    args = ap.parse_args()
    if args.extra_session_dir:
        EXTRA["dir"] = args.extra_session_dir.resolve()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.daemon_threads = True
    print("dashboard on http://127.0.0.1:%d/  (repo %s)" % (srv.server_address[1], ROOT), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
