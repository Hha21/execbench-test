"use strict";
// SOLX ops console: vanilla JS, hash routing. No external libraries.
const $ = (s, el = document) => el.querySelector(s);
const view = $("#view");
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const enc = encodeURIComponent;
let timers = [], sources = [];

async function api(path) {
  const r = await fetch("/api/" + path, {cache: "no-store"});
  if (!r.ok) throw new Error(path + ": " + r.status);
  return r.json();
}
function every(ms, fn) { fn(); const t = setInterval(() => { if (!document.hidden) fn(); }, ms); timers.push(t); }
function cleanup() { timers.forEach(clearInterval); timers = []; sources.forEach(s => s.close()); sources = []; }

const fmt = (x, d = 3) => x == null ? "-" : Number(x).toFixed(d);
const fmtUs = x => x == null ? "-" : Number(x).toFixed(x < 10 ? 2 : 1);
const dur = s => { if (s == null) return "-"; s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60);
  return (h ? h + "h" : "") + (h || m ? String(m).padStart(h ? 2 : 1, "0") + "m" : "") + String(s % 60).padStart(h || m ? 2 : 1, "0") + "s"; };
const ago = t => { const s = Date.now() / 1000 - t; return s < 90 ? Math.round(s) + "s ago" : s < 5400 ? Math.round(s / 60) + "m ago" : s < 172800 ? Math.round(s / 3600) + "h ago" : Math.round(s / 86400) + "d ago"; };
const pill = s => `<span class="pill ${esc(s)}">${esc(s)}</span>`;

// ---------------------------------------------------------------- markdown (small subset, escapes everything)
function inline(s) {
  return esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[\s(])\*([^*\s][^*]*)\*/g, "$1<i>$2</i>");
}
function md(src) {
  const lines = String(src || "").split("\n"); let out = "", i = 0, list = false;
  const close = () => { if (list) { out += "</ul>"; list = false; } };
  while (i < lines.length) {
    const l = lines[i];
    if (/^\s*```/.test(l)) {
      close(); let code = []; i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i++]);
      i++; out += `<pre><code>${esc(code.join("\n"))}</code></pre>`; continue;
    }
    let m;
    if ((m = /^(#{1,4})\s+(.*)/.exec(l))) { close(); out += `<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`; }
    else if ((m = /^\s*[-*]\s+(.*)/.exec(l))) { if (!list) { out += "<ul>"; list = true; } out += `<li>${inline(m[1])}</li>`; }
    else if (/^\s*\|.*\|\s*$/.test(l)) {
      close(); const rows = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(lines[i++]);
      i--; out += "<table>" + rows.filter(r => !/^\s*\|[\s:|-]+\|\s*$/.test(r)).map((r, k) =>
        "<tr>" + r.trim().replace(/^\||\|$/g, "").split("|").map(c => `<t${k ? "d" : "h"}>${inline(c.trim())}</t${k ? "d" : "h"}>`).join("") + "</tr>").join("") + "</table>";
    }
    else if (!l.trim()) close();
    else { close(); out += `<p>${inline(l)}</p>`; }
    i++;
  }
  close(); return `<div class="md">${out}</div>`;
}

// ---------------------------------------------------------------- pipeline graph
function drawPipeline(host, p) {
  const colX = [60, 170, 300, 440, 560, 670, 770, 870], W = 940, nodeW = 92, nodeH = 26, gap = 34;
  const pos = {}; let maxRow = 0;
  const colN = {}; p.nodes.forEach(n => { colN[n.col] = Math.max(colN[n.col] || 0, n.row + 1); });
  const rows = Math.max(...Object.values(colN)); const H = rows * gap + 70;
  p.nodes.forEach(n => {
    const off = (rows - colN[n.col]) * gap / 2;
    pos[n.id] = {x: colX[n.col], y: 28 + off + n.row * gap, n};
  });
  const act = new Set(p.nodes.filter(n => n.state === "active").map(n => n.id));
  let s = `<svg class="pipe" viewBox="0 0 ${W} ${H}">`;
  p.links.forEach(([a, b]) => {
    const A = pos[a], B = pos[b]; if (!A || !B) return;
    const on = act.has(a) && act.has(b) || act.has(a) && b === "candidates" && false;
    if (a === "ingest" && b === "lead") {
      const y = H - 12; s += `<path class="lk back ${act.has("ingest") ? "on" : ""}" d="M${A.x},${A.y + nodeH / 2} V${y} H${B.x} V${B.y + nodeH / 2}"/>`;
      return;
    }
    const x1 = A.x + nodeW / 2, x2 = B.x - nodeW / 2, mx = (x1 + x2) / 2;
    s += `<path class="lk ${on || act.has(a) && act.has(b) ? "on" : ""}" d="M${x1},${A.y} C${mx},${A.y} ${mx},${B.y} ${x2},${B.y}"/>`;
  });
  p.nodes.forEach(n => {
    const q = pos[n.id];
    s += `<g class="pn ${n.state}" ${n.ref ? `data-ref="${esc(n.ref)}"` : ""}><rect x="${q.x - nodeW / 2}" y="${q.y - nodeH / 2}" width="${nodeW}" height="${nodeH}"/>` +
      `<text x="${q.x}" y="${q.y + (n.note ? -1 : 4)}">${esc(n.label)}</text>` +
      (n.note ? `<text class="sub" x="${q.x}" y="${q.y + 9}">${esc(n.note)}</text>` : "") + `<title>${esc(n.id)}: ${esc(n.state)}</title></g>`;
  });
  host.innerHTML = s + "</svg>";
  host.querySelectorAll("[data-ref]").forEach(g => g.onclick = () => {
    const [d, ...nm] = g.dataset.ref.split("/"); location.hash = `#/p/${enc(p.problem)}/r/${enc(p.round)}/s/${d}/${enc(nm.join("/"))}`;
  });
}

function pipelinePanel(host, problem, onData) {
  host.innerHTML = `<h2>Pipeline <span class="mut" id="pl-meta"></span></h2><div id="pl"></div>`;
  every(3000, async () => {
    try {
      const p = await api("pipeline" + (problem ? "?problem=" + enc(problem) : ""));
      drawPipeline($("#pl", host), p);
      $("#pl-meta", host).textContent = `${p.problem} / ${p.round || "-"} ` + (p.processes.length ? p.processes.map(x => x.sub + (x.round ? ":" + x.round : "")).join(" ") : "(no loop process)");
      onData && onData(p);
    } catch (e) { $("#pl", host).textContent = String(e); }
  });
}

// ---------------------------------------------------------------- live feed
const KEYARGS = ["note", "id", "kernel", "kernel_id", "name", "path", "file", "example", "query"];
function scriptLines(v, n = 6) { const ls = String(v).split("\n").filter(x => x.trim()); return ls.slice(0, n).join("\n") + (ls.length > n ? `\n... (+${ls.length - n} lines)` : ""); }
function renderCall(e) {
  const inp = e.input && typeof e.input === "object" ? e.input : {value: e.input};
  let args = "", code = "";
  for (const [k, v] of Object.entries(inp)) {
    if (typeof v === "string" && (v.includes("\n") || v.length > 160)) { if (!code && /script|code|source|solution|cuda|triton|python/i.test(k)) code = `<b class="mut">${esc(k)}</b>\n` + scriptLines(v); else args += `<span class="arg">${esc(k)}=<b>${esc(v.slice(0, 140))}...</b></span> `; }
    else if (v !== null && typeof v !== "object") args += `<span class="arg">${esc(k)}=<b>${esc(String(v))}</b></span> `;
    else if (v && typeof v === "object") args += `<span class="arg">${esc(k)}=<b>${esc(JSON.stringify(v).slice(0, 120))}</b></span> `;
  }
  return `<div><span class="nm">${esc(e.name)}</span> ${args}</div>` + (code ? `<pre class="code">${code}</pre>` : "") + `<div class="res"></div>`;
}
function hiResult(t) {
  return esc(t).replace(/\bPASSED\b/g, '<span class="ok">PASSED</span>').replace(/\b(FAILED|ERROR|Traceback|error)\b/g, '<span class="bad">$1</span>')
    .replace(/(\d+(?:\.\d+)?)\s?(µs|us)\b/g, '<span class="us">$1 $2</span>')
    .replace(/(predicted[^\n]{0,40}?score\s*)(\d\.\d+)/gi, '$1<span class="sc">$2</span>');
}
function renderResult(e) {
  const t = e.text || "", np = (t.match(/PASSED/g) || []).length, nf = (t.match(/FAILED/g) || []).length;
  const first = t.split("\n").find(l => l.trim()) || "";
  const head = (np || nf ? `<span class="ok">${np} PASSED</span> ${nf ? `<span class="bad">${nf} FAILED</span>` : ""} ` : "") +
    esc(first.slice(0, 110)) + (e.is_error ? ' <span class="bad">[error]</span>' : "");
  const body = t.length > 20000 ? t.slice(0, 20000) + "\n... truncated" : t;
  return `<details${nf || e.is_error ? " open" : ""}><summary>${head}</summary><pre>${hiResult(body)}</pre></details>`;
}

function makeFeed(host, side, url, opts = {}) {
  host.classList.add("feed"); if (opts.small) host.classList.add("small");
  const calls = {};
  let stat = null, started = null, el;
  const add = html => { const d = document.createElement("div"); d.innerHTML = html; const n = d.firstElementChild;
    const stick = host.scrollTop + host.clientHeight >= host.scrollHeight - 40; host.appendChild(n);
    while (opts.max && host.children.length > opts.max) host.removeChild(host.firstChild);
    if (stick) host.scrollTop = host.scrollHeight; return n; };
  const es = new EventSource(url); sources.push(es);
  es.addEventListener("ev", m => {
    const e = JSON.parse(m.data);
    if (e.k === "text") add(`<div class="ev text">${esc(e.text)}</div>`);
    else if (e.k === "thinking") add(`<details class="ev thinking"><summary>thinking</summary>${esc(e.text)}</details>`);
    else if (e.k === "call") calls[e.id] = add(`<div class="ev call">${renderCall(e)}</div>`);
    else if (e.k === "result") {
      const c = calls[e.id]; const html = renderResult(e);
      if (c) { c.querySelector(".res").insertAdjacentHTML("beforeend", html); host.scrollTop = Math.min(host.scrollTop + 0, host.scrollHeight); }
      else add(`<div class="ev call">${html}</div>`);
    }
    else if (e.k === "final") add(`<div class="ev final">session finished: ${e.turns ?? "?"} turns, $${fmt(e.cost, 2)}${e.is_error ? " [ERROR]" : ""}</div>`);
    else if (e.k === "init") add(`<div class="ev mut">session started${e.model ? " (" + esc(e.model) + ")" : ""}</div>`);
  });
  es.addEventListener("stat", m => { stat = JSON.parse(m.data); drawStat(); });
  es.addEventListener("end", () => { es.close(); });
  function drawStat() {
    if (!side || !stat) return;
    const s = stat, el = s.live ? (Date.now() / 1000 - s.started) : s.elapsed;
    const calls = Object.entries(s.calls || {}).map(([k, v]) => `<div class="kv"><span>${esc(k)}</span><span>${v}</span></div>`).join("");
    side.innerHTML = `<div class="stat"><div><b>${s.turns}</b><span>turns</span></div><div><b>${s.gpu_calls}</b><span>GPU calls (compile+test)</span></div>` +
      `<div><b>${s.probes}</b><span>probes</span></div><div><b>${dur(el)}</b><span>${s.live ? "elapsed (live)" : "elapsed"}</span></div>` +
      `<div><b>${s.cost != null ? "$" + fmt(s.cost, 2) : "-"}</b><span>cost (result event)</span></div><div><b>${s.final ? (s.is_error ? "ERROR" : "DONE") : "LIVE"}</b><span>source: ${s.mode || "waiting"}</span></div></div>` +
      `<h3 style="margin-top:10px">Tool calls</h3>${calls || '<span class="mut">none yet</span>'}`;
  }
  if (side) { timers.push(setInterval(drawStat, 1000)); }
  return es;
}

// ---------------------------------------------------------------- chart
function chartSvg(workloads, portal, rented) {
  const pts = workloads.map(w => w.key), W = 900, H = 330, L = 50, R = 28, T = 12, B = 54;
  const vals = []; workloads.forEach(w => { [w.tb, w.tsol, portal && portal[w.key], rented && rented[w.key]].forEach(v => v > 0 && vals.push(v)); });
  if (!vals.length) return '<div class="mut">no timing data</div>';
  const lo = Math.log10(Math.min(...vals)) - 0.05, hi = Math.log10(Math.max(...vals)) + 0.05;
  const x = i => L + (pts.length < 2 ? 0 : i * (W - L - R) / (pts.length - 1)), y = v => T + (hi - Math.log10(v)) / (hi - lo) * (H - T - B);
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}">`;
  for (let d = Math.ceil(lo); d <= Math.floor(hi); d++) for (const m of [1, 2, 5]) { const v = m * 10 ** d; if (Math.log10(v) < lo || Math.log10(v) > hi) continue;
    s += `<line class="gr" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 4}" y="${y(v) + 3}" text-anchor="end">${v >= 1000 ? v / 1000 + "ms" : v + "us"}</text>`; }
  s += `<line class="ax" x1="${L}" x2="${L}" y1="${T}" y2="${H - B}"/><line class="ax" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/>`;
  const line = (get, color, dash, label, dots) => {
    const p = workloads.map((w, i) => [x(i), get(w), w]).filter(a => a[1] > 0); if (!p.length) return "";
    let g = `<path d="${p.map((a, i) => (i ? "L" : "M") + a[0] + "," + y(a[1])).join(" ")}" fill="none" stroke="${color}" stroke-width="1.6" ${dash ? 'stroke-dasharray="5 4"' : ""}/>`;
    if (dots) g += p.map(a => `<circle cx="${a[0]}" cy="${y(a[1])}" r="3" fill="${color}"><title>${esc(label)} ${esc(a[2].key)}: ${a[1].toFixed(2)} us</title></circle>`).join("");
    return g;
  };
  s += line(w => w.tb, "#7a8f80", true, "Tb (baseline)") + line(w => w.tsol, "#3dff7a", true, "Tsol") + line(w => rented && rented[w.key], "#6fd6e6", false, "rented B200", true) + line(w => portal && portal[w.key], "#ffb000", false, "portal", true);
  workloads.forEach((w, i) => { s += `<text transform="translate(${x(i) + 3},${H - B + 8}) rotate(60)">${esc(w.key)}</text>`; });
  s += `<g font-size="10"><text x="${L + 6}" y="${T + 10}" style="fill:#ffb000">portal</text><text x="${L + 50}" y="${T + 10}" style="fill:#6fd6e6">rented B200</text><text x="${L + 125}" y="${T + 10}" style="fill:#3dff7a">Tsol</text><text x="${L + 160}" y="${T + 10}" style="fill:#7a8f80">Tb</text></g>`;
  return s + "</svg>";
}

// ---------------------------------------------------------------- pages
async function pageHome() {
  view.innerHTML = `<div class="row"><div class="col-main"><div class="panel" id="pipe"></div><div class="dropnote">Drop a saved portal result page anywhere on this window to ingest it.</div><div class="grid" id="cards"></div></div>` +
    `<div class="col-side"><div class="panel"><h2>Live agent feed <span class="mut" id="lf-name"></span></h2><div id="lf"><span class="mut">no running session</span></div><div id="lf-side" style="margin-top:8px"></div></div></div></div>`;
  let feedRef = null;
  pipelinePanel($("#pipe"), null, p => {
    const n = p.nodes.find(n => n.ref && n.state === "active");
    const ref = n ? `${p.problem}/${p.round}/${n.ref}` : null;
    if (ref && ref !== feedRef) {
      feedRef = ref; sources.forEach(s => s.close()); sources = [];
      const host = $("#lf"); host.innerHTML = ""; $("#lf-name").textContent = n.label;
      const [d, ...nm] = n.ref.split("/");
      makeFeed(host, $("#lf-side"), `/api/stream/${enc(p.problem)}/${enc(p.round)}/${d}/${enc(nm.join("/"))}`, {small: true, max: 80});
    }
  });
  every(5000, async () => {
    const d = await api("problems");
    $("#cards").innerHTML = d.problems.map(p => {
      const r = p.rank, run = p.running.length;
      return `<a class="pcard" href="#/p/${enc(p.name)}"><div class="nm">${esc(p.name)} ${run ? pill("running") : ""}</div>` +
        `<div class="mut">${esc(p.level || "")}</div><div class="big">${fmt(p.best_score, 4)}</div><div class="mut">${esc(p.best || "no portal result yet")}</div><hr style="border-color:#12251a">` +
        (r ? `<div class="kv"><span>public s1 / s5</span><span>${fmt(r.s1)} / ${fmt(r.s5)}</span></div>` +
          `<div class="kv"><span>rank (of s1,s2,s3,s5 above us)</span><span class="${r.above ? "a" : "g"}">${r.above ?? "-"} above${r.beats && r.beats.length ? " | beats " + r.beats.join(",") : ""}</span></div>` : '<div class="mut">no public leaderboard row</div>') +
        `<div class="kv"><span>kernels</span><span>${p.n_kernels}</span></div><div class="kv"><span>latest round</span><span>${esc(p.latest_round || "-")} (${p.rounds} total)</span></div>` +
        (run ? `<div class="kv"><span>running</span><span class="g">${p.running.map(x => esc(x.sub + (x.round ? ":" + x.round : ""))).join(" ")}</span></div>` : "") + `</a>`;
    }).join("");
  });
}

function ledgerTable(led) {
  const hs = (led && led.hypotheses) || [];
  const cons = (led && led.constraints) || [];
  return (cons.length ? `<details><summary>${cons.length} constraints</summary><ul>${cons.map(c => `<li>${esc(c)}</li>`).join("")}</ul></details>` : "") +
    `<div class="scroll"><table><tr><th>id</th><th>status</th><th>statement</th><th>evidence</th></tr>` +
    hs.map(h => `<tr class="st-${esc(h.status)}"><td>${esc(h.id)}</td><td class="stc">${esc(h.status)}</td><td>${esc(h.statement)}</td>` +
      `<td>${(h.evidence || []).length ? `<details><summary>${h.evidence.length}</summary>${h.evidence.map(e => `<div>- ${esc(e)}</div>`).join("")}</details>` : ""}</td></tr>`).join("") + "</table></div>";
}

async function pageProblem(name) {
  view.innerHTML = `<h1>${esc(name)}</h1><div id="body" class="mut">loading (first load runs the loop's own modules)...</div>`;
  const d = await api("problem/" + enc(name));
  const s = d.summary || {}, ks = (s.kernels || []).slice().sort((a, b) => (b.score ?? -1) - (a.score ?? -1));
  const best = ks.find(k => k.id === s.best) || ks[0];
  $("#body").outerHTML = `<div class="panel" id="pipe"></div>` +
    `<div class="row"><div class="col-main">` +
    `<div class="panel"><h2>Best ${best ? esc(best.id) : ""} <span class="a">${fmt(s.best_score, 4)}</span></h2>${d.rank ? `<span class="mut">public s1 ${fmt(d.rank.s1)} s2 ${fmt(d.rank.s2)} s3 ${fmt(d.rank.s3)} s5 ${fmt(d.rank.s5)} s10 ${fmt(d.rank.s10)} | ${d.rank.above ?? "-"} above us | ${esc(d.rank.top5_users || "")}</span>` : ""}</div>` +
    `<div class="panel"><h2>Per-workload times <select id="ksel"></select></h2><div id="chart"></div></div>` +
    `<div class="panel"><h2>Kernels (${ks.length})</h2><div class="scroll"><table><tr><th>id</th><th>round</th><th>status</th><th class="n">portal</th><th class="n">geomean us</th><th class="n">rented S</th><th class="n">M</th><th class="n">L</th></tr>` +
    ks.map(k => `<tr><td>${esc(k.id)}</td><td><a href="#/p/${enc(name)}/r/${enc(k.round || "")}">${esc(k.round)}</a></td><td>${esc((k.status || "").slice(0, 40))}</td><td class="n a">${fmt(k.score, 4)}</td><td class="n">${fmtUs(k.geomean_us)}</td>` +
      `<td class="n">${fmtUs(k.bands && k.bands.S)}</td><td class="n">${fmtUs(k.bands && k.bands.M)}</td><td class="n">${fmtUs(k.bands && k.bands.L)}</td></tr>`).join("") + `</table></div></div>` +
    `<div class="panel"><h2>Hypothesis ledger</h2>${ledgerTable(d.ledger)}</div>` +
    `<div class="panel"><h2>Card</h2><div class="scroll" style="max-height:600px">${md(d.card)}</div></div>` +
    `</div><div class="col-side"><div class="panel"><h2>Rounds</h2><table><tr><th>round</th><th class="n">tasks</th><th class="n">sess</th><th class="n">cand</th><th></th></tr>` +
    d.rounds.slice().reverse().map(r => `<tr class="clk" onclick="location.hash='#/p/${enc(name)}/r/${enc(r.name)}'"><td><a href="#/p/${enc(name)}/r/${enc(r.name)}">${esc(r.name)}</a></td><td class="n">${r.tasks}</td><td class="n">${r.sessions}</td><td class="n">${r.candidates}</td><td class="mut">${r.research ? "R" : ""}${r.lead ? "L" : ""} ${ago(r.mtime)}</td></tr>`).join("") +
    `</table></div></div></div>`;
  pipelinePanel($("#pipe"), name);
  const sel = $("#ksel"), withP = ks.filter(k => k.portal || k.rented);
  sel.innerHTML = withP.map(k => `<option value="${esc(k.id)}" ${k.id === s.best ? "selected" : ""}>${esc(k.id)}${k.score ? " (" + fmt(k.score, 4) + ")" : ""}</option>`).join("");
  const draw = () => { const k = ks.find(k => k.id === sel.value); $("#chart").innerHTML = k ? chartSvg(s.workloads || [], k.portal, k.rented) : '<span class="mut">no data</span>'; };
  sel.onchange = draw; draw();
}

async function pageRound(name, rn) {
  view.innerHTML = `<h1>${esc(name)} / ${esc(rn)}</h1><div id="body" class="mut">loading...</div>`;
  let first = true;
  const draw = async () => {
    const d = await api(`problem/${enc(name)}/round/${enc(rn)}`);
    const sess = d.sessions, tasks = (d.plan && d.plan.tasks) || [];
    const link = s => `#/p/${enc(name)}/r/${enc(rn)}/s/${s.dir}/${enc(s.name)}`;
    const openIds = new Set([...document.querySelectorAll("details[data-k][open]")].map(x => x.dataset.k));
    const html = `<div class="panel" id="pipe-r"></div>` +
      `<div class="panel"><h2>Plan ${d.plan ? `<span class="mut">best ${esc(d.plan.best)} ${fmt(d.plan.best_score, 4)}</span>` : '<span class="mut">(no plan.json)</span>'}</h2>` +
      tasks.map((t, i) => `<div style="margin:6px 0"><b class="g">${i}</b> ${esc(t.operation)} <span class="pill">${esc(t.model)}</span>${t.fresh ? '<span class="pill active">fresh</span>' : ""} band ${esc(t.band)} <span class="a">${esc(t.niche)}</span> <span class="mut">parents: ${esc((t.parents || []).join(", ") || "-")}</span>` +
        `<details data-k="t${i}" ${openIds.has("t" + i) ? "open" : ""}><summary>instructions</summary><pre>${esc(t.instructions)}</pre></details></div>`).join("") + `</div>` +
      (d.research_reply ? `<div class="panel"><h2>Research reply</h2><details data-k="rr" ${openIds.has("rr") ? "open" : ""}><summary>show</summary><div class="scroll" style="max-height:600px">${md(d.research_reply)}</div></details></div>` : "") +
      (d.lead_reply ? `<div class="panel"><h2>Lead reply</h2><details data-k="lr" ${openIds.has("lr") ? "open" : ""}><summary>show</summary><div class="scroll" style="max-height:600px">${md(d.lead_reply)}</div></details></div>` : "") +
      `<div class="panel"><h2>Sessions (${sess.length})</h2><table><tr><th>session</th><th>status</th><th class="n">turns</th><th class="n">calls</th><th class="n">cost</th><th class="n">time</th></tr>` +
      sess.map(s => `<tr class="clk" onclick="location.hash='${link(s)}'"><td><a href="${link(s)}">${s.dir === "sessions" ? esc(s.name) : esc(s.dir)}</a></td><td>${pill(s.status)} ${s.status === "running" && s.last_tool ? `<span class="mut">${esc(s.last_tool)}</span>` : ""}</td><td class="n">${s.turns ?? "-"}</td><td class="n">${s.tool_calls}</td><td class="n">${s.cost != null ? "$" + fmt(s.cost, 2) : "-"}</td><td class="n">${dur(s.elapsed)}</td></tr>`).join("") + `</table></div>` +
      `<div class="panel"><h2>Candidates (${d.candidates.length})</h2><table><tr><th>id</th><th>tests</th><th>archive</th><th class="n">portal score</th><th class="n">geomean us</th></tr>` +
      d.candidates.map(c => `<tr><td>${esc(c.id)}</td><td>${c.tested ? `<span class="${c.failed ? "bad" : "ok"}">${c.passed}/${c.tested} passed</span> ${esc(c.fail_status.join(","))}` : '<span class="mut">untested</span>'}</td><td class="mut">${esc((c.arch_status || "").slice(0, 40))}</td><td class="n a">${fmt(c.score, 4)}</td><td class="n">${fmtUs(c.geomean_us)}</td></tr>`).join("") + `</table>` +
      (d.rejected.length ? `<details><summary>${d.rejected.length} rejected</summary>${d.rejected.map(r => `<div class="mut">${esc(r)}</div>`).join("")}</details>` : "") + `</div>` +
      (d.auto_log_tail ? `<div class="panel"><h2>auto.log (tail)</h2><details data-k="al" ${openIds.has("al") ? "open" : ""}><summary>show</summary><pre>${esc(d.auto_log_tail)}</pre></details></div>` : "");
    const body = $("#body") || $("#rbody"); body.id = "rbody"; body.className = "";
    const keep = window.scrollY; body.innerHTML = html;
    pipelinePanelOnce($("#pipe-r"), name, rn); window.scrollTo(0, keep);
  };
  every(4000, () => draw().catch(e => { $("#body, #rbody").textContent = String(e); }));
}
function pipelinePanelOnce(host, name, rn) {
  host.innerHTML = `<h2>Pipeline</h2><div id="pl"></div>`;
  api("pipeline?problem=" + enc(name)).then(p => drawPipeline($("#pl", host), p)).catch(() => {});
}

async function pageSession(name, rn, dir, sn) {
  view.innerHTML = `<h1>${esc(name)} / ${esc(rn)} / ${esc(dir === "sessions" ? sn : dir)}</h1><div class="row"><div class="col-main"><div class="panel"><h2>Agent feed</h2><div id="feed"></div></div></div>` +
    `<div class="col-side"><div class="panel"><h2>Session</h2><div id="side"><span class="mut">connecting...</span></div></div><div class="panel" id="pipe"></div></div></div>`;
  makeFeed($("#feed"), $("#side"), `/api/stream/${enc(name)}/${enc(rn)}/${dir}/${enc(sn)}`);
  pipelinePanel($("#pipe"), name);
}

// ---------------------------------------------------------------- router
function crumbs(parts) {
  $("#crumbs").innerHTML = parts.map(([t, h]) => h ? `<a href="${h}">${esc(t)}</a>` : esc(t)).join(" / ");
}
async function route() {
  cleanup(); window.scrollTo(0, 0);
  const h = decodeURIComponent(location.hash.replace(/^#\/?/, "")).split("/").filter(Boolean);
  try {
    if (h[0] === "p" && h[1]) {
      const p = h[1];
      if (h[2] === "r" && h[3]) {
        if (h[4] === "s" && h[5]) { crumbs([["home", "#/"], [p, `#/p/${enc(p)}`], [h[3], `#/p/${enc(p)}/r/${enc(h[3])}`], ["session"]]); return pageSession(p, h[3], h[5], h.slice(6).join("/")); }
        crumbs([["home", "#/"], [p, `#/p/${enc(p)}`], [h[3]]]); return pageRound(p, h[3]);
      }
      crumbs([["home", "#/"], [p]]); return pageProblem(p);
    }
    crumbs([["home"]]); return pageHome();
  } catch (e) { view.innerHTML = `<div class="panel r">${esc(e)}</div>`; }
}
window.addEventListener("hashchange", route);
setInterval(() => { $("#clock").textContent = new Date().toLocaleTimeString(); }, 1000);

// ---------------------------------------------------------------- drag and drop ingest
let dragDepth = 0;
const veil = $("#dropveil");
const hasFiles = e => e.dataTransfer && [...e.dataTransfer.types].includes("Files");
window.addEventListener("dragenter", e => { if (hasFiles(e)) { e.preventDefault(); dragDepth++; veil.classList.add("on"); } });
window.addEventListener("dragover", e => { if (hasFiles(e)) e.preventDefault(); });
window.addEventListener("dragleave", e => { if (hasFiles(e) && --dragDepth <= 0) { dragDepth = 0; veil.classList.remove("on"); } });
window.addEventListener("drop", async e => {
  if (!hasFiles(e)) return;
  e.preventDefault(); dragDepth = 0; veil.classList.remove("on");
  const panel = $("#ingest"), out = $("#iout"), st = $("#istate");
  panel.hidden = false; out.textContent = ""; st.textContent = "";
  for (const f of e.dataTransfer.files) {
    st.textContent = "uploading " + f.name + "...";
    out.textContent += `$ ingest ${f.name} (${(f.size / 1024).toFixed(0)} KB)\n`;
    try {
      const r = await fetch("/api/upload?name=" + enc(f.name), {method: "POST", body: f});
      const j = await r.json();
      if (j.error) out.textContent += "ERROR: " + j.error + "\n";
      else out.textContent += `${j.command}\n${j.stdout}${j.stderr ? "\n[stderr]\n" + j.stderr : ""}\nexit ${j.returncode}\n`;
    } catch (err) { out.textContent += "ERROR: " + err + "\n"; }
    out.textContent += "\n";
  }
  st.textContent = "done"; route();
});
$("#iclose").onclick = () => { $("#ingest").hidden = true; };
route();
