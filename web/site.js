// Handoff: every prompt becomes a box. A box shows its sandbox's scraper terminal, then its tuning terminal,
// then the trained robot doing the task. Click a box for the technical overview.
//   live:   POST /api/runs, then a WebSocket per box
//   replay: ?run=<id>[,<id>...] replays recorded runs; ?demo=1 types three prompts (film mode adds &film=1)
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const short = (s, n = 64) => (String(s).length > n ? String(s).slice(0, n - 1) + "…" : String(s));
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const qs = new URLSearchParams(location.search);
  const film = qs.has("film");
  if (film) document.body.classList.add("is-film");

  const PHASES = ["Scrape", "Verify", "Retarget", "Tune", "Test"];
  const PHASE_OF = { queued: -1, plan: 0, sandbox_start: 0, sandbox: 0, search: 0, page: 0, download: 0, clip: 0, blocked: 0, warn: 0,
    sandbox_destroyed: 0, picked: 0, done: 0, verifying: 1, verdict: 1, moves: 2, motion: 2, episodes: 2, dataset: 2,
    train: 3, trained: 3, rollout: 4, evaluated: 4 };
  const boxes = [];
  let robot = "so101";

  // ---------- composer ----------
  const input = $("#prompt");
  let family = null;
  document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => { input.value = c.dataset.text; family = c.dataset.family; input.focus(); }));
  input.addEventListener("input", () => { family = null; });
  const list = $("#robot-list");
  $("#robot-btn").addEventListener("click", () => { list.hidden = !list.hidden; $("#robot-btn").setAttribute("aria-expanded", String(!list.hidden)); });
  list.addEventListener("click", (e) => {
    const o = e.target.closest(".robot-opt"); if (!o || o.disabled) return;
    robot = o.dataset.robot; $("#robot-label").textContent = o.querySelector("b").textContent; list.hidden = true;
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".composer")) list.hidden = true; });

  const submit = (inp) => async (ev) => {
    ev.preventDefault();
    const text = inp.value.trim() || input.placeholder;
    const r = await fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: `I want to train a robot to ${text}`, family, robot }) });
    if (!r.ok) return;
    const { id } = await r.json();
    const box = addBox(id, text);
    inp.value = ""; family = null;
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/runs/${id}/ws`);
    ws.onmessage = (m) => box.handle(JSON.parse(m.data));
  };
  $("#composer").addEventListener("submit", submit(input));
  $("#composer2").addEventListener("submit", submit($("#prompt2")));

  // ---------- a box ----------
  function addBox(id, task) {
    const fleet = $("#fleet");
    fleet.hidden = false;
    const n = boxes.length + 1;
    const root = el("article", "box");
    root.innerHTML = `
      <header class="box-head">
        <div><div class="box-id">ROBOT ${String(n).padStart(2, "0")} · SO-101 · <span class="sbx">queued</span></div>
        <div class="box-task">${esc(task)}</div></div>
        <span class="status wait"><i></i><span>Queued</span></span>
      </header>
      <div class="screen">
        <div class="term"><div class="term-bar"><i></i><i></i><i></i><span class="term-title">waiting for a sandbox</span></div>
          <div class="term-body"></div></div>
        <video muted loop playsinline preload="none"></video>
        <div class="deploy">Ready to deploy <small class="deploy-rate"></small></div>
      </div>
      <footer class="box-foot">
        <div class="phases">${PHASES.map(() => "<i></i>").join("")}</div>
        <div class="phase-names">${PHASES.map((p) => `<span>${p}</span>`).join("")}</div>
        <div class="nums"><span>clips <b class="n-clips">0</b></span><span>episodes <b class="n-eps">0</b></span><span>success <b class="n-succ">–</b></span></div>
      </footer>`;
    $("#grid").prepend(root);
    const box = new Box(id, task, n, root);
    boxes.push(box);
    root.addEventListener("click", () => openSheet(box));
    $("#fleet-count").textContent = `${boxes.length} robot${boxes.length > 1 ? "s" : ""} · one sandbox each`;
    return box;
  }

  class Box {
    constructor(id, task, n, root) {
      Object.assign(this, { id, task, n, root, events: [], log: [], phase: -1, mode: "scrape", clips: 0, verified: 0, eps: 0, rollouts: [] });
      this.body = $(".term-body", root);
    }
    line(text, cls = "") {
      this.log.push([this.mode, text, cls]);
      const d = el("div", cls, esc(text));
      this.body.appendChild(d);
      while (this.body.children.length > 40) this.body.firstChild.remove();
    }
    setStatus(label, kind = "") {
      const s = $(".status", this.root);
      s.className = `status ${kind}`;
      $("span", s).textContent = label;
    }
    setPhase(p) {
      if (p <= this.phase) return;
      this.phase = p;
      $$(".phases i", this.root).forEach((i, k) => { i.className = k < p ? "done" : k === p ? "on" : ""; });
      this.setStatus(["Scraping", "Verifying", "Retargeting", "Tuning", "Testing"][p]);
      if (p >= 2 && this.mode !== "tune") {
        this.mode = "tune";
        this.body.innerHTML = "";
        $(".term-title", this.root).textContent = `tune · so101-${this.family || "policy"} · cpu · MediaPipe → MuJoCo → PyTorch`;
      }
    }
    handle(e) {
      this.events.push(e);
      const p = PHASE_OF[e.type];
      if (p !== undefined && p >= 0) this.setPhase(p);
      const f = FORMAT[e.type];
      if (f) { const out = f(e, this); if (out) (Array.isArray(out[0]) ? out : [out]).forEach(([t, c]) => this.line(t, c)); }
      if (e.type === "error") { this.setStatus("Failed", "no"); this.line(`error: ${e.message}`, "t-no"); }
      if (e.type === "done" && e.timings_s) this.finish();
    }
    finish() {
      $$(".phases i", this.root).forEach((i) => { i.className = "done"; });
      const ev = this.events.find((e) => e.type === "evaluated");
      if (!ev || !ev.n) { this.setStatus("Not enough data", "no"); return; }
      this.success = `${ev.success}/${ev.n}`;
      $(".n-succ", this.root).textContent = this.success;
      $(".deploy-rate", this.root).textContent = `${this.success} unseen layouts`;
      this.setStatus("Ready to deploy", "ok");
      this.root.classList.add("is-ready");
      const v = $("video", this.root);
      v.src = `/runs/${this.id}/showcase.mp4`;
      v.onerror = () => { v.src = `/runs/${this.id}/policy.mp4`; };
      $(".screen", this.root).classList.add("is-video");
      v.play().catch(() => {});
    }
  }
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  // ---------- terminal lines ----------
  const FORMAT = {
    queued: (e) => [`queued · position ${e.position + 1}`, "t-dim"],
    plan: (e, b) => { b.family = e.family; return [[`agent> task family: ${e.family} · "${e.summary}"`, "t-sys"]]; },
    sandbox_start: (e, b) => {
      $(".sbx", b.root).textContent = "sandbox " + e.container.replace("arena-sbx-", "").slice(0, 6);
      $(".term-title", b.root).textContent = `${e.container} · scraper · ${e.caps.cpus} CPU · ${e.caps.memory} · ${e.caps.seconds}s kill`;
      return [[`$ docker run --rm --read-only --cap-drop ALL --pids-limit ${e.caps.pids} arena-scraper`, "t-sys"]];
    },
    sandbox: (e) => [`  uid ${e.user} · ${e.env_secrets} secrets in env · writable ${e.writable.join(" ")}`, "t-dim"],
    search: (e) => [`GET ${short(e.source, 46)} "${short(e.query, 20)}" → ${e.hits}`, "t-info"],
    page: (e) => { let u = e.url; try { u = new URL(e.url).pathname; } catch {} return [`GET ${u} → ${e.videos} video link(s), ${e.scripts_ignored} script(s) not run`, ""]; },
    blocked: (e) => [`✗ BLOCKED ${e.reason}`, "t-no"],
    download: (e) => [`↓ ${short(e.title, 52)}  [${e.licence}]`, ""],
    clip: (e, b) => { b.clips += 1; $(".n-clips", b.root).textContent = b.clips; return [`✓ kept ${short(e.title, 44)} · ${e.seconds}s → h264 360p`, "t-ok"]; },
    warn: (e) => [`· ${e.message}`, "t-dim"],
    sandbox_destroyed: (e) => [`$ ${e.container} destroyed after ${e.seconds}s`, "t-sys"],
    picked: (e) => (e.of ? [`agent> picked ${e.n} of ${e.of} results`, "t-sys"] : null),
    verifying: (e) => [`vlm> checking ${short(e.title, 50)}`, "t-dim"],
    verdict: (e, b) => { if (e.accept) { b.verified += 1; $(".n-clips", b.root).textContent = `${b.verified}/${b.clips}`; }
      return [`vlm> ${e.accept ? "✓ verified" : "✗ rejected"}: ${short(e.reason, 70)}`, e.accept ? "t-ok" : "t-no"]; },
    moves: (e) => [`mediapipe> ${e.found} clean hand moves in ${e.id} (${e.rejected} too small)`, "t-info"],
    motion: (e) => [`mujoco> move ${e.move.split("@")[1] || ""}: ${e.probe.filter(Boolean).length}/3 probe layouts · ${e.ok ? "kept" : "dropped"} · lift ${e.max_lift}`, e.ok ? "t-ok" : "t-no"],
    episodes: (e, b) => { b.eps += e.kept; $(".n-eps", b.root).textContent = b.eps; return [`dataset> +${e.kept}/${e.tried} physics-checked episodes`, ""]; },
    dataset: (e, b) => { $(".n-eps", b.root).textContent = e.episodes; return [`dataset> ${e.episodes} episodes · ${e.frames.toLocaleString()} frames · ${(e.bytes / 1e6).toFixed(1)} MB`, "t-sys"]; },
    train: (e) => [`train> step ${String(e.iter).padStart(4)}/${e.iters}  loss ${e.loss.toFixed(4)}${e.lr ? `  lr ${e.lr.toExponential(1)}` : ""}${e.samples_per_s ? `  ${(e.samples_per_s / 1000).toFixed(1)}k samples/s` : ""}`, ""],
    trained: (e) => [`train> done · ${e.params.toLocaleString()} params · ${e.train_seconds}s on CPU · ${(e.bytes / 1024).toFixed(0)} KB`, "t-ok"],
    rollout: (e, b) => { b.rollouts.push(e.ok); return [`eval> unseen layout ${String(e.i + 1).padStart(2, "0")}/${e.n} ${e.ok ? "✓ success" : "✗ miss"}`, e.ok ? "t-ok" : "t-no"]; },
    evaluated: (e) => (e.n ? [`eval> ${e.success}/${e.n} unseen layouts · policy.pt ready to deploy`, "t-sys"] : [`eval> ${e.reason}`, "t-no"]),
  };

  // ---------- technical overview ----------
  async function openSheet(box) {
    const sheet = $("#sheet");
    const d = await fetch(`/api/runs/${box.id}`).then((r) => r.json()).catch(() => ({}));
    const s = d.summary || {};
    const ev = (t) => box.events.filter((e) => e.type === t);
    const pol = s.policy || {};
    const ready = pol.eval_n ? `${pol.eval_success}/${pol.eval_n}` : null;
    $("#sheet-kicker").textContent = `ROBOT ${String(box.n).padStart(2, "0")} · SO-101 · ${box.id}`;
    $("#sheet-title").textContent = box.task;
    const verdicts = ev("verdict");
    const moves = ev("motion");
    const t = s.timings_s || {};
    const rolls = (pol.eval || box.rollouts);
    const steps = [
      ["Plan", "the agent picks the task family and the people to learn from", t.plan],
      ["Sandbox", "throwaway containers fetch and re-encode openly licensed video; no secrets inside", t.scrape],
      ["Verify", "a VLM checks every clip: real person, hand, object, the right motion", t.verify],
      ["Retarget", "MediaPipe hand track → SO-101 path → MuJoCo physics gate", t.motion],
      ["Dataset", "each kept move replayed on new layouts, every rollout re-gated", t.dataset],
      ["Tune", `ChunkMLP, ${(pol.params || 0).toLocaleString()} params, on the CPU`, t.train],
      ["Test", `${pol.eval_n || 20} fixed layouts the policy never saw`, t.evaluate],
    ];
    $("#sheet-body").innerHTML = `
      <section class="card">
        <h3>Deployment</h3>
        ${ready ? `<video class="hero-video" src="/runs/${box.id}/showcase.mp4" autoplay muted loop playsinline></video>` : ""}
        <div class="ready">${ready ? `<span class="status ok"><i></i><span>Ready to deploy</span></span>` : `<span class="status wait"><i></i><span>${esc($(".status span", box.root).textContent)}</span></span>`}
          <span class="nums">policy.pt · ${pol.bytes ? (pol.bytes / 1024).toFixed(0) + " KB" : "–"} · state-based · SO-101</span></div>
        <p class="note">Sim-tested only: the robot passed ${ready || "–"} unseen layouts in MuJoCo. It reads joint angles and object positions, not camera pixels.</p>
      </section>
      <section class="card">
        <h3>At a glance</h3>
        <dl class="kv">
          <div><dd>${ev("clip").length}</dd><dt>clips pulled in the sandbox</dt></div>
          <div><dd>${verdicts.filter((v) => v.accept).length}/${verdicts.length}</dd><dt>verified by the VLM</dt></div>
          <div><dd>${ev("blocked").length}</dd><dt>hostile or junk fetches blocked</dt></div>
          <div><dd>${moves.filter((m) => m.ok).length}/${moves.length}</dd><dt>human hand moves kept by physics</dt></div>
          <div><dd>${(s.dataset || {}).episodes ?? box.eps}</dd><dt>physics-checked episodes</dt></div>
          <div><dd>${pol.train_seconds ?? "–"}s</dd><dt>tuning on the CPU</dt></div>
        </dl>
        <h3 style="margin-top:16px">Pipeline</h3>
        <ol class="steps">${steps.map(([a, b, c]) => `<li><b>${a}</b><code>${c != null ? c + " s" : ""}</code><span style="grid-column:2/-1">${b}</span></li>`).join("")}</ol>
      </section>
      <section class="card">
        <h3>Where it learned from</h3>
        <div class="src">${verdicts.map((v) => `<div><em class="${v.accept ? "ok" : "no"}">${v.accept ? "used" : "rejected"}</em>
          <span>${esc(short(v.title, 80))}<small>${esc(v.licence)} · ${esc(v.author || v.source)}</small><small>${esc(v.reason)}</small></span></div>`).join("") || "<p class='note'>No clips yet.</p>"}</div>
      </section>
      <section class="card">
        <h3>Contained in the sandbox</h3>
        <div class="src">${ev("blocked").map((b) => `<div><em class="no">blocked</em><span>${esc(b.reason)}<small>${esc(short(b.url, 70))}</small></span></div>`).join("") || "<p class='note'>Nothing blocked.</p>"}</div>
        <h3 style="margin-top:16px">Test on unseen layouts</h3>
        <div class="dots">${rolls.map((ok) => `<i class="${ok ? "" : "no"}"></i>`).join("")}</div>
      </section>
      <section class="card span">
        <h3>Full log</h3>
        <div class="log">${box.log.map(([, text, cls]) => `<div class="${cls}">${esc(text)}</div>`).join("")}</div>
      </section>`;
    sheet.hidden = false;
    document.body.classList.add("is-locked");
  }
  const closeSheet = () => { $("#sheet").hidden = true; document.body.classList.remove("is-locked"); };
  $("#sheet-close").addEventListener("click", closeSheet);
  $("#sheet-backdrop").addEventListener("click", closeSheet);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSheet(); });

  // ---------- replay and demo ----------
  async function replay(id, task, speed, maxGap) {
    const d = await fetch(`/api/runs/${id}`).then((r) => r.json());
    const box = addBox(id, task || (d.summary?.text || "").replace(/^I want to train a robot to /, ""));
    let prev = null;
    for (const e of d.events) {
      const at = e.at ?? e.t;
      if (prev != null && at != null) await wait(Math.min(maxGap, Math.max(30, ((at - prev) * 1000) / speed)));
      if (at != null) prev = at;
      box.handle(e);
    }
    return box;
  }
  async function typeAndTrain(text, inp, btn) {
    inp.value = "";
    for (const ch of text) { inp.value += ch; await wait(45); }
    await wait(350);
    btn.classList.add("is-pressed"); await wait(160); btn.classList.remove("is-pressed");
    inp.value = "";
  }
  async function glideTo(y, ms = 900) {
    const y0 = scrollY, steps = Math.max(1, Math.round(ms / 33));
    for (let k = 1; k <= steps; k++) {
      const t = k / steps, e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      window.scrollTo(0, y0 + (y - y0) * e);
      await wait(33);
    }
  }
  const speed = Number(qs.get("speed") || 14);
  const maxGap = Number(qs.get("gap") || 1400);
  if (qs.get("demo")) {
    const plan = [["box-place", "put the block in the bowl"], ["box-stack", "stack the block on another block"], ["box-tower", "add one more block to make a tower"]];
    (async () => {
      await wait(Number(qs.get("delay") || 900));
      const runs = [];
      const tag = qs.get("speedtag") ? el("div", "speedtag", "") : null;
      if (tag) document.body.appendChild(tag);
      const setTag = (html) => { if (tag) { tag.innerHTML = html; tag.classList.add("on"); } };
      for (const [k, [id, text]] of plan.entries()) {
        if (k === 0) await typeAndTrain(text, input, $("#go"));
        else await typeAndTrain(text, $("#prompt2"), $("#go2"));
        runs.push(replay(id, text, speed, maxGap));
        if (k === 0) setTag(`<b>▶▶ ${esc(qs.get("speedtag"))}× speed</b><span>each robot took ${esc(qs.get("livetime") || "10 to 11 min")} live on a laptop CPU</span>`);
        if (k === 0) { await wait(300); await glideTo($("#fleet").offsetTop - 12, 1100); }
        await wait(Number(qs.get("stagger") || 3500));
      }
      const done = await Promise.all(runs);
      setTag(`<b>▶ 1× real time</b><span>robot videos: MuJoCo physics, not animation</span>`);
      await wait(2000);
      if (qs.get("open") !== "0") await openSheet(done[done.length - 1]);
      window.__demoDone = true;
    })();
  } else if (qs.get("run")) {
    qs.get("run").split(",").forEach((id) => replay(id, null, speed, maxGap));
  }
})();
