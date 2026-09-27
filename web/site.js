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
  const mark = (k) => { if (film) (window.__marks ||= []).push([k, Date.now()]); };

  // one card and one detail view for every kind of robot; only the labels differ
  const KIND = {
    robot: { phases: ["Scrape", "Verify", "Retarget", "Tune", "Test"], labels: ["Scraping", "Verifying", "Retargeting", "Tuning", "Testing"], nums: ["clips", "episodes", "success"] },
    workout: { phases: ["Scrape", "Segment", "Verify", "Copy 1:1", "Ready"], labels: ["Scraping", "Segmenting", "Verifying", "Copying 1:1", "Ready"], nums: ["videos", "reps found", "elbow error"] },
    chess: { phases: ["Search", "Scrape", "Check", "Rehearse", "Ready"], labels: ["Searching", "Scraping", "Checking", "Rehearsing", "Ready"], nums: ["moves", "pick-and-places", "worst"] },
    puzzle: { phases: ["Plan", "Lay out", "Rehearse", "Check", "Ready"], labels: ["Planning", "Laying out", "Rehearsing", "Checking", "Ready"], nums: ["moves", "checked", "worst"] },
  };
  const PHASE_OF = {
    robot: { queued: -1, plan: 0, sandbox_start: 0, sandbox: 0, search: 0, page: 0, download: 0, clip: 0, blocked: 0, warn: 0,
      sandbox_destroyed: 0, vm_create: 0, vm_booting: 0, vm_ready: 0, picked: 0, verifying: 1, verdict: 1, moves: 2, motion: 2, episodes: 2, dataset: 2,
      train: 3, trained: 3, rollout: 4, evaluated: 4 },
    workout: { plan: 0, vm_create: 0, sandbox_start: 0, search: 0, candidate: 0, picked: 0, download: 0, clip: 0, clips: 1, pose_start: 1, pose: 1, pose_done: 1,
      verdict: 2, chosen: 2, robot_start: 3, robot: 3 },
    chess: { plan: 0, vm_create: 0, search: 0, picked: 1, page: 1, pgn_candidate: 1, validate: 2, game: 2, replay: 3 },
  };
  const boxes = [];
  let robot = "so101";

  // ---------- composer ----------
  const input = $("#prompt");
  let family = null;
  let mode = null;
  document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => { input.value = c.dataset.text; family = c.dataset.family || null; mode = c.dataset.mode || null; input.focus(); }));
  input.addEventListener("input", () => { family = null; mode = null; });
  const list = $("#robot-list");
  $("#robot-btn").addEventListener("click", () => { list.hidden = !list.hidden; $("#robot-btn").setAttribute("aria-expanded", String(!list.hidden)); });
  list.addEventListener("click", (e) => {
    const o = e.target.closest(".robot-opt"); if (!o || o.disabled) return;
    robot = o.dataset.robot; $("#robot-label").textContent = o.querySelector("b").textContent; list.hidden = true;
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".composer")) list.hidden = true; });
  const modeOf = (text) => mode || (/workout|exercise|curl|dumbbell|gym|lift|reps?\b/i.test(text) ? "workout"
    : /chess|\bvs\.?\b|kasparov|fischer|game \d/i.test(text) ? "chess" : "robot");

  const submit = (inp) => async (ev) => {
    ev.preventDefault();
    const text = inp.value.trim() || input.placeholder;
    const kind = modeOf(text);
    const body = kind === "robot" ? { text: `I want to train a robot to ${text}`, family, robot } : { text, mode: kind, robot };
    const r = await fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) return;
    const { id } = await r.json();
    const box = addBox(id, text, kind);
    inp.value = ""; family = null; mode = null;
    $("#fleet").scrollIntoView({ behavior: "smooth", block: "start" });
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/runs/${id}/ws`);
    ws.onmessage = (m) => box.handle(JSON.parse(m.data));
  };
  $("#composer").addEventListener("submit", submit(input));
  $("#composer2").addEventListener("submit", submit($("#prompt2")));

  // ---------- a box (identical for every kind) ----------
  function addBox(id, task, kind = "robot") {
    const fleet = $("#fleet");
    fleet.hidden = false;
    const n = boxes.length + 1;
    const K = KIND[kind];
    const root = el("article", `box kind-${kind}`);
    root.innerHTML = `
      <header class="box-head">
        <div><div class="box-id">ROBOT ${String(n).padStart(2, "0")} · SO-101 · <span class="sbx">queued</span></div>
        <div class="box-task">${esc(task)}</div></div>
        <div class="head-r"><span class="status wait"><i></i><span>Queued</span></span>
        <button type="button" class="kill" title="Destroy this robot's sandbox now">Kill</button>
        <a class="dl" href="/api/runs/${esc(id)}/download" download title="Download this robot's skill">Download</a></div>
      </header>
      <div class="screen">
        <div class="term"><div class="term-bar"><i></i><i></i><i></i><span class="term-title">waiting for a sandbox</span></div>
          <div class="term-body"></div></div>
        <div class="tiles"></div>
        <video muted loop playsinline preload="none"></video>
        <div class="deploy">Ready to deploy <small class="deploy-rate"></small></div>
      </div>
      <footer class="box-foot">
        <div class="phases">${K.phases.map(() => "<i></i>").join("")}</div>
        <div class="phase-names">${K.phases.map((p) => `<span>${p}</span>`).join("")}</div>
        <div class="nums">${K.nums.map((l, i) => `<span>${l} <b data-n="${i}">–</b></span>`).join("")}</div>
      </footer>`;
    $("#grid").prepend(root);
    const box = new Box(id, task, n, root, kind);
    boxes.push(box);
    root.addEventListener("click", () => openSheet(box));
    $(".kill", root).addEventListener("click", (ev) => { ev.stopPropagation(); killBox(box); });
    $(".dl", root).addEventListener("click", (ev) => ev.stopPropagation());
    $("#fleet-count").textContent = `${boxes.length} robot${boxes.length > 1 ? "s" : ""} · one sandbox each`;
    return box;
  }

  // a scraped video with MediaPipe's skeleton drawn on it, frame by frame as the pose model reaches it
  const BONES = [[1, 2], [1, 3], [3, 5], [2, 4], [4, 6], [1, 7], [2, 8], [7, 8], [7, 9], [9, 11], [8, 10], [10, 12], [0, 1], [0, 2]];
  class Tile {
    constructor(rid, clip, big = false) {
      Object.assign(this, { rid, clip, frames: [], last: 0, done: false });
      this.root = el("div", `tile${big ? " big" : ""}`);
      this.root.innerHTML = `<video muted playsinline preload="auto" src="/runs/${esc(rid)}/clips/${esc(clip.id)}.mp4"></video><canvas></canvas>
        <b class="tile-badge"></b><div class="tile-bar"><i></i></div><span class="tile-cap">${esc(short(clip.title.replace(/\.(webm|ogv|ogg|mp4)$/i, ""), 44))}</span>`;
      this.v = $("video", this.root);
      this.c = $("canvas", this.root);
    }
    rect() {
      const W = this.root.clientWidth, H = this.root.clientHeight;
      const vw = this.v.videoWidth || 16, vh = this.v.videoHeight || 9;
      const s = Math.min(W / vw, H / vh);
      return [(W - vw * s) / 2, (H - vh * s) / 2, vw * s, vh * s, W, H];
    }
    draw(lm) {
      const [ox, oy, w, h, W, H] = this.rect();
      if (this.c.width !== W) { this.c.width = W; this.c.height = H; }
      const g = this.c.getContext("2d");
      g.clearRect(0, 0, W, H);
      if (!lm) return;
      g.lineWidth = Math.max(2, W / 120); g.lineCap = "round";
      const P = (k) => [ox + lm[k][0] * w, oy + lm[k][1] * h];
      for (const [a, b] of BONES) {
        if (lm[a][2] < 0.4 || lm[b][2] < 0.4) continue;
        const arm = [3, 4, 5, 6].includes(a) || [3, 4, 5, 6].includes(b);
        g.strokeStyle = arm ? "rgba(241,207,138,0.95)" : "rgba(143,211,166,0.85)";
        g.beginPath(); g.moveTo(...P(a)); g.lineTo(...P(b)); g.stroke();
      }
      g.fillStyle = "#fff";
      for (const k of [3, 4, 5, 6]) if (lm[k][2] >= 0.4) { const [x, y] = P(k); g.beginPath(); g.arc(x, y, Math.max(2, W / 90), 0, 7); g.fill(); }
    }
    update(e) {
      this.frames.push(e);
      $(".tile-bar i", this.root).style.width = `${Math.min(100, (100 * (e.i + 1)) / Math.max(1, e.n))}%`;
      this.root.classList.add("is-live");
      const now = performance.now();
      if (!this.done && now - this.last > 90) { this.last = now; try { this.v.currentTime = e.t; } catch {} this.draw(e.lm); }
    }
    finish(e) {
      this.done = true;
      this.root.classList.remove("is-live");
      this.root.classList.add(e.ok ? "ok" : "no");
      $(".tile-badge", this.root).textContent = e.ok ? `${e.reps} reps` : e.reps ? `${e.reps} rep` : "no reps";
      $(".tile-bar i", this.root).style.width = "100%";
      this.loop();
    }
    loop(window) {
      this.v.loop = true;
      this.v.play().catch(() => {});
      const tick = () => {
        if (!this.root.isConnected) return;
        const t = this.v.currentTime;
        if (window && (t < window[0] || t > window[1])) { try { this.v.currentTime = window[0]; } catch {} }
        let best = null;
        for (const f of this.frames) { if (f.t <= t) best = f; else break; }
        this.draw(best && best.lm);
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }
  }

  class Box {
    constructor(id, task, n, root, kind) {
      Object.assign(this, { id, task, n, root, kind, events: [], log: [], phase: -1, mode: "scrape", clips: 0, verified: 0, eps: 0, rollouts: [], tiles: {} });
      this.body = $(".term-body", root);
      this.prompt = kind === "robot" ? `I want to train a robot to ${task}` : task;
    }
    num(i, v) { const b = $(`[data-n="${i}"]`, this.root); if (b) b.textContent = v; }
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
      this.setStatus(KIND[this.kind].labels[p]);
      if (this.kind === "robot" && p >= 2 && this.mode !== "tune") {
        this.mode = "tune";
        this.body.innerHTML = "";
        $(".term-title", this.root).textContent = `tune · so101-${this.family || "policy"} · cpu · MediaPipe → MuJoCo → PyTorch`;
      }
    }
    handle(e) {
      this.events.push(e);
      const p = (PHASE_OF[this.kind] || {})[e.type];
      if (p !== undefined && p >= 0) this.setPhase(p);
      const f = (this.kind === "robot" ? FORMAT : WFORMAT)[e.type];
      if (f) { const out = f(e, this); if (out) (Array.isArray(out[0]) ? out : [out]).forEach(([t, c]) => this.line(t, c)); }
      if (this.kind === "workout") this.workout(e);
      if (e.type === "blocked") mark("blocked");
      if (e.type === "error") { this.setStatus("Failed", "no"); this.line(`error: ${e.message}`, "t-no"); this.ended(); }
      if (e.type === "killed") { this.setStatus("Killed", "no"); this.ended(); }
      if (e.type === "done" && e.timings_s) this.finish();
    }
    workout(e) {
      if (e.type === "clips") { this.clipList = e.clips; this.num(0, e.clips.length); }
      if (e.type === "pose") ((this.pose ||= {})[e.id] ||= []).push({ t: e.t, lm: e.lm });
      if (e.type === "pose_done") {
        (this.results ||= {})[e.id] = e;
        this.num(1, Object.values(this.results).reduce((n, r) => n + r.reps, 0));
      }
      if (e.type === "verdict") (this.verdicts ||= {})[e.id] = e;
      if (e.type === "chosen") this.chosen = e;
      if (e.type === "robot") { this.robot = e; this.num(2, `${e.elbow_tracking_rms_deg}°`); }
    }
    ended() { this.root.classList.add("is-ended"); }
    showVideo(src, fallback) {
      const v = $("video", this.root);
      v.src = src;
      if (fallback) v.onerror = () => { v.src = fallback; };
      const sc = $(".screen", this.root);
      sc.classList.remove("is-tiles");
      sc.classList.add("is-video");
      v.play().catch(() => {});
    }
    ready(rate) {
      this.ended();
      $$(".phases i", this.root).forEach((i) => { i.className = "done"; });
      $(".deploy-rate", this.root).textContent = rate;
      this.setStatus("Ready to deploy", "ok");
      mark("ready");
      this.root.classList.add("is-ready");
    }
    finish() {
      if (this.kind === "workout") {
        if (!this.robot) { this.setStatus("Not enough data", "no"); this.ended(); return; }
        this.ready(`${this.chosen ? this.chosen.reps : ""} reps copied 1:1`);
        this.showVideo(`/runs/${this.id}/robot.mp4`);
        return;
      }
      const ev = this.events.find((x) => x.type === "evaluated");
      if (!ev || !ev.n) { this.setStatus("Not enough data", "no"); this.ended(); return; }
      this.success = `${ev.success}/${ev.n}`;
      this.num(2, this.success);
      this.ready(`${this.success} unseen layouts`);
      this.showVideo(`/runs/${this.id}/showcase.mp4`, `/runs/${this.id}/policy.mp4`);
    }
  }
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  // ---------- terminal lines ----------
  const FORMAT = {
    queued: (e) => [`queued · position ${e.position + 1}`, "t-dim"],
    plan: (e, b) => { b.family = e.family; return [[`agent> task family: ${e.family} · "${e.summary}"`, "t-sys"]]; },
    vm_create: (e, b) => { $(".sbx", b.root).textContent = `vultr ${e.instance.slice(0, 8)}`;
      $(".term-title", b.root).textContent = `vultr ${e.plan} · ${e.region} · booting`;
      return [[`vultr> create instance ${e.instance.slice(0, 8)} · ${e.plan} · ${e.region}${e.hourly_usd ? ` · $${e.hourly_usd}/h` : ""}`, "t-sys"], [`vultr> firewall: ${e.firewall} · fresh SSH key · API key stays in the app`, "t-dim"]]; },
    vm_booting: (e) => [`vultr> ${e.ip} up · ${e.step}`, "t-dim"],
    vm_ready: (e) => [`vultr> ready in ${e.seconds}s · docker ${e.docker}`, "t-ok"],
    vm_destroyed: (e) => [`vultr> instance ${e.instance.slice(0, 8)} deleted after ${e.seconds}s${e.cost_usd != null ? ` · ≈$${e.cost_usd}` : ""}`, "t-sys"],
    model_call: (e) => [`${e.kind.toLowerCase()}> ${e.model} @ ${e.endpoint} · ${e.ms} ms`, "t-dim"],
    killed: () => ["■ KILLED by operator · sandbox destroyed", "t-no"],
    sandbox_start: (e, b) => {
      const where = e.host && e.host !== "local" ? e.host : "local docker";
      if (!e.host || e.host === "local") $(".sbx", b.root).textContent = "sandbox " + e.container.replace("arena-sbx-", "").slice(0, 6);
      $(".term-title", b.root).textContent = `${e.container} · ${where} · ${e.caps.cpus} CPU · ${e.caps.memory} · ${e.caps.seconds}s kill`;
      return [[`$ docker run --rm --read-only --cap-drop ALL --pids-limit ${e.caps.pids} arena-scraper`, "t-sys"]];
    },
    sandbox: (e) => [`  uid ${e.user} · ${e.env_secrets} secrets in env · writable ${e.writable.join(" ")}`, "t-dim"],
    search: (e) => [`GET ${short(e.source, 46)} "${short(e.query, 20)}" → ${e.hits}`, "t-info"],
    page: (e) => { let u = e.url; try { u = new URL(e.url).pathname; } catch {} return [`GET ${u} → ${e.videos} video link(s), ${e.scripts_ignored} script(s) not run`, ""]; },
    blocked: (e) => [`✗ BLOCKED ${e.reason}`, "t-no"],
    download: (e) => [`↓ ${short(e.title, 52)}  [${e.licence}]`, ""],
    clip: (e, b) => { b.clips += 1; b.num(0, b.clips); return [`✓ kept ${short(e.title, 44)} · ${e.seconds}s → h264 360p`, "t-ok"]; },
    warn: (e) => [`· ${e.message}`, "t-dim"],
    sandbox_destroyed: (e) => [`$ ${e.container} destroyed after ${e.seconds}s`, "t-sys"],
    picked: (e) => (e.of ? [`agent> picked ${e.n} of ${e.of} results`, "t-sys"] : null),
    verifying: (e) => [`vlm> checking ${short(e.title, 50)}`, "t-dim"],
    verdict: (e, b) => { if (e.accept) { b.verified += 1; b.num(0, `${b.verified}/${b.clips}`); }
      return [`vlm> ${e.accept ? "✓ verified" : "✗ rejected"}: ${short(e.reason, 70)}`, e.accept ? "t-ok" : "t-no"]; },
    moves: (e) => [`mediapipe> ${e.found} clean hand moves in ${e.id} (${e.rejected} too small)`, "t-info"],
    motion: (e) => [`mujoco> move ${e.move.split("@")[1] || ""}: ${e.probe.filter(Boolean).length}/3 probe layouts · ${e.ok ? "kept" : "dropped"} · lift ${e.max_lift}`, e.ok ? "t-ok" : "t-no"],
    episodes: (e, b) => { b.eps += e.kept; b.num(1, b.eps); return [`dataset> +${e.kept}/${e.tried} physics-checked episodes`, ""]; },
    dataset: (e, b) => { b.num(1, e.episodes); return [`dataset> ${e.episodes} episodes · ${e.frames.toLocaleString()} frames · ${(e.bytes / 1e6).toFixed(1)} MB`, "t-sys"]; },
    train: (e) => [`train> step ${String(e.iter).padStart(4)}/${e.iters}  loss ${e.loss.toFixed(4)}${e.lr ? `  lr ${e.lr.toExponential(1)}` : ""}${e.samples_per_s ? `  ${(e.samples_per_s / 1000).toFixed(1)}k samples/s` : ""}`, ""],
    trained: (e) => [`train> done · ${e.params.toLocaleString()} params · ${e.train_seconds}s on CPU · ${(e.bytes / 1024).toFixed(0)} KB`, "t-ok"],
    rollout: (e, b) => { b.rollouts.push(e.ok); return [`eval> unseen layout ${String(e.i + 1).padStart(2, "0")}/${e.n} ${e.ok ? "✓ success" : "✗ miss"}`, e.ok ? "t-ok" : "t-no"]; },
    evaluated: (e) => (e.n ? [`eval> ${e.success}/${e.n} unseen layouts · policy.pt ready to deploy`, "t-sys"] : [`eval> ${e.reason}`, "t-no"]),
  };

  // ---------- terminal lines for workout and chess runs ----------
  const WFORMAT = {
    plan: (e) => [`agent> ${e.exercise || `${e.white} vs ${e.black}, ${e.year}`}${e.why ? ": " + e.why : ""}`, "t-sys"],
    vm_create: (e) => [`vultr> VM ${e.instance.slice(0, 8)} · ${e.plan} · ${e.region}${e.warm ? " · pre-warmed" : ""}`, "t-sys"],
    vm_destroyed: (e) => [`vultr> VM deleted after ${e.seconds}s`, "t-sys"],
    sandbox_start: (e, b) => { $(".sbx", b.root).textContent = e.host && e.host !== "local" ? e.host.split(" · ")[0] : `sandbox ${e.container.replace("arena-sbx-", "").slice(0, 6)}`; return [`$ docker run --rm --read-only --cap-drop ALL ${e.container}`, "t-sys"]; },
    sandbox: (e) => [`  uid ${e.user} · ${e.env_secrets} secrets in env`, "t-dim"],
    search: (e) => [`scrapling> ${short(e.source, 30)}: "${short(e.query, 30)}" → ${e.hits}`, "t-info"],
    picked: (e) => [`agent> picked ${e.n} of ${e.of}`, "t-sys"],
    blocked: (e) => [`✗ BLOCKED ${short(e.reason, 60)}`, "t-no"],
    download: (e) => [`↓ ${short(e.title, 52)} [${short(e.licence, 20)}]`, ""],
    clip: (e) => [`✓ ${short(e.title, 44)} · ${e.seconds}s → h264 360p`, "t-ok"],
    clips: (e) => [`mediapipe> segmenting ${e.clips.length} videos`, "t-sys"],
    pose_done: (e, b) => [`mediapipe> ${short(((b.clipList || []).find((c) => c.id === e.id) || {}).title || e.id, 36)}: ${e.reps} reps, ${Math.round(e.tracked * 100)}% tracked`, e.ok ? "t-ok" : "t-dim"],
    verdict: (e) => [`vlm> ${e.accept ? "✓" : "✗"} ${short(e.reason, 64)}`, e.accept ? "t-ok" : "t-no"],
    chosen: (e) => [`agent> copying ${e.reps} reps from ${short(e.title, 40)}`, "t-sys"],
    robot_start: (e) => [`mujoco> ${e.message}`, "t-info"],
    robot: (e) => [`mujoco> ${e.held_through_curl ? "held the dumbbell ✓" : "dropped it ✗"} · elbow within ${e.elbow_tracking_rms_deg}° of the human`, e.ok ? "t-ok" : "t-no"],
    page: (e) => [`scrapling> ${short(e.title || e.url, 40)} → ${e.move_lists} move lists`, "t-info"],
    validate: (e) => [`python-chess> ${e.ok ? `${e.plies} legal moves ✓` : e.reason}`, e.ok ? "t-ok" : "t-no"],
    game: (e) => [`game> ${e.plies} moves found and checked`, "t-ok"],
    replay: (e) => [`mujoco> ${e.transfers} pick-and-places, each rehearsed in physics ✓`, "t-ok"],
    warn: (e) => [`· ${e.message}`, "t-dim"],
    done: (e) => (e.timings_s ? [`done in ${e.timings_s.total}s`, "t-dim"] : null),
  };

  // ---------- a chess box: the same card, fed from a recorded chess run ----------
  async function addChessBox(rid) {
    const d = await fetch(`/api/runs/${rid}`).then((r) => r.json());
    const box = addBox(rid, d.events[0]?.text || "", "chess");
    box.mode = "tune";
    for (const e of d.events) {
      box.events.push(e);
      const f = WFORMAT[e.type];
      const out = f && f(e, box);
      if (out) box.line(...out);
    }
    const rep = d.events.find((e) => e.type === "replay");
    box.replay = rep;
    $(".term-title", box.root).textContent = "chess agent · Scrapling · python-chess · MuJoCo";
    const vm = d.events.find((e) => e.type === "vm_create");
    if (vm) $(".sbx", box.root).textContent = `vultr ${vm.instance.slice(0, 8)}`;
    if (!rep) { box.setStatus("Not rehearsed", "no"); return box; }
    box.num(0, rep.plies); box.num(1, rep.transfers); box.num(2, `${rep.max_err_mm} mm`);
    box.ready(`${rep.plies} moves, both sides`);
    box.showVideo(rep.video);
    return box;
  }

  // ---------- a puzzle box (Tower of Hanoi, cup pyramid): a computed plan, every move rehearsed in physics ----------
  const PUZZLE = {
    "task-hanoi": { prompt: "Solve the Tower of Hanoi", title: "Tower of Hanoi, three discs", agent: "hanoi agent · recursion · MuJoCo",
      plan: "optimal solution: 2^3 - 1 = 7 moves, never a larger disc on a smaller one", page: "https://en.wikipedia.org/wiki/Tower_of_Hanoi" },
    "task-cups": { prompt: "Stack six cups into a pyramid", title: "Six-cup pyramid", agent: "cups agent · layout · MuJoCo",
      plan: "3-2-1 pyramid laid out from the cup size; each cup carried just high enough", page: "https://en.wikipedia.org/wiki/Cup_stacking" },
  };
  async function addPuzzleBox(rid) {
    const P = PUZZLE[rid];
    const moves = await fetch(`/tasks/${rid}.json`).then((r) => r.json());  // copied from runs/<rid>/moves.json by scripts/task_render.py
    const box = addBox(rid, P.prompt, "puzzle");
    box.mode = "tune";
    box.moves = moves;
    $(".term-title", box.root).textContent = P.agent;
    $(".sbx", box.root).textContent = "local";
    box.line(`agent> ${P.plan}`, "t-sys");
    for (const m of moves) box.line(`mujoco> ${m.label || "move"} · ${m.err_mm} mm ✓`, "t-ok");
    const worst = Math.max(0, ...moves.map((m) => m.err_mm || 0));
    box.num(0, moves.length); box.num(1, `${moves.filter((m) => m.ok !== false).length}/${moves.length}`); box.num(2, `${worst} mm`);
    box.ready(`${moves.length} moves, each checked in physics`);
    box.showVideo(`/runs/${rid}/robot.mp4`);
    return box;
  }

  // ---------- technical overview: one template for every robot ----------
  const kv = (items) => `<dl class="kv">${items.map(([v, l]) => `<div><dd>${esc(v)}</dd><dt>${esc(l)}</dt></div>`).join("")}</dl>`;
  const stepList = (items) => `<ol class="steps">${items.map(([a, b, c]) => `<li><b>${esc(a)}</b><code>${c != null ? esc(c) + " s" : ""}</code><span style="grid-column:2/-1">${esc(b)}</span></li>`).join("")}</ol>`;
  const rows = (items) => `<div class="src">${items.map((r) => `<div><em class="${r.ok ? "ok" : "no"}">${esc(r.tag)}</em><span>${esc(short(r.title, 90))}<small>${esc(r.meta || "")}</small></span></div>`).join("") || "<p class='note'>Nothing yet.</p>"}</div>`;

  function sheetModel(box, summary) {
    const ev = (t) => box.events.filter((e) => e.type === t);
    const vmEv = ev("vm_create")[0];
    const host = vmEv ? `Vultr VM ${vmEv.instance.slice(0, 8)} · ${vmEv.plan}` : "local Docker sandbox";
    if (box.kind === "puzzle") {
      const P = PUZZLE[box.id];
      const mv = box.moves || [];
      return {
        title: P.title,
        media: `<video class="hero-video" src="/runs/${esc(box.id)}/robot.mp4" autoplay muted loop playsinline controls></video>`,
        note: "MuJoCo physics (Google DeepMind), official SO-101 model and servo gains. Every move is rehearsed first and only played if it lands.",
        glance: [[mv.length, "moves"], [`${Math.max(0, ...mv.map((m) => m.err_mm || 0))} mm`, "worst placement error"],
          [`${Math.max(0, ...mv.map((m) => m.knocked_mm || 0))} mm`, "most a neighbour was nudged"], ["local", "where it ran"]],
        steps: [["Plan", P.plan], ["Lay out", "objects and targets placed inside the arm's reach"],
          ["Rehearse", "each grasp tried in physics, with variants if it slips"], ["Check", "placement error, tilt and knocked neighbours measured"]],
        sources: rows([{ ok: true, tag: "rules", title: P.page, meta: "Wikipedia, CC BY-SA" }, ...mv.map((m) => ({ ok: true, tag: "move", title: m.label, meta: `${m.err_mm} mm off target` }))]),
      };
    }
    if (box.kind === "chess") {
      const r = box.replay || {};
      return {
        title: "Deep Blue vs. Kasparov, 1997, Game 6",
        media: r.video ? `<video class="hero-video" id="sheet-video" src="${esc(r.video)}" autoplay muted loop playsinline controls></video>
          <ol class="c-moves" id="sheet-moves">${(r.moves || []).map((m, i) => `<li>${i % 2 === 0 ? `<b>${i / 2 + 1}.</b>` : ""}${esc(m.san)}</li>`).join("")}</ol>` : "<p class='note'>Not rehearsed yet.</p>",
        note: `MuJoCo physics (Google DeepMind), official SO-101 model and servo gains. ${Math.round(r.sim_s || 0)} s of robot time shown at ${(r.speed || 1).toFixed(1)}×.`,
        glance: [[r.plies, "moves, both colours"], [r.captures, "captures, pieces to the tray"], [r.transfers, "pick-and-places, each rehearsed first"],
          [`${r.max_err_mm} mm`, "worst placement off a square's centre"], [ev("page").length, "pages read by the scraper"], [host.split(" · ")[0], "where the scraping ran"]],
        steps: [["Plan", "the agent turned the prompt into a search"], ["Scrape", `Scrapling in a throwaway sandbox (${host})`],
          ["Check", `python-chess re-read every move: ${r.plies} legal`], ["Rehearse", "every grasp simulated and checked before it is played"]],
        sources: rows(ev("page").map((e) => ({ ok: e.move_lists > 0, tag: e.move_lists > 0 ? "used" : "read", title: e.title || e.url, meta: `${e.move_lists} move lists · Wikipedia, CC BY-SA` }))),
        onOpen: () => syncMoves(r),
      };
    }
    if (box.kind === "workout") {
      const rob = box.robot || {};
      const ch = box.chosen;
      const res = box.results || {};
      const clipsList = box.clipList || [];
      return {
        title: "Dumbbell biceps curl, copied 1:1",
        media: rob.ok !== undefined ? `<video class="hero-video" src="/runs/${esc(box.id)}/robot.mp4" autoplay muted loop playsinline controls></video>` : "<p class='note'>Still running.</p>",
        note: "The SO-101 picks up a dumbbell and copies a person's curl: their elbow angle drives its elbow joint degree for degree. MuJoCo physics (Google DeepMind), official SO-101 model.",
        glance: [[clipsList.length, "openly licensed videos scraped"], [Object.values(res).reduce((n, r) => n + r.reps, 0), "reps found in them"],
          [ch ? ch.reps : "–", "reps copied by the robot"], [rob.elbow_tracking_rms_deg != null ? `${rob.elbow_tracking_rms_deg}°` : "–", "robot elbow vs human (RMS)"],
          [rob.max_slip_mm != null ? `${rob.max_slip_mm} mm` : "–", "dumbbell slip in the grip"], [host.split(" · ")[0], "where the scraping ran"]],
        steps: [["Plan", "the agent chose an exercise the arm can copy 1:1"], ["Scrape", `Scrapling in a throwaway sandbox (${host})`],
          ["Segment", "MediaPipe Pose on every video: joints, elbow angle, reps"], ["Verify", "a VLM confirmed the best clip is a real curl"],
          ["Copy", "the SO-101 picks up the dumbbell and follows the elbow 1:1"]],
        sources: `<p class="note">Every scraped video, segmented by MediaPipe Pose. Click one to play it with the skeleton; the gold line under it is that person's elbow angle over the whole video, which is what the robot copies.</p><div class="srcgrid" id="srcgrid"></div>`,
        onOpen: () => {
          const grid = $("#srcgrid");
          const order = [...clipsList].sort((a, b) => (ch && b.id === ch.id) - (ch && a.id === ch.id) || ((res[b.id] || {}).reps || 0) - ((res[a.id] || {}).reps || 0));
          for (const c of order) {
            const r = res[c.id] || {};
            const used = ch && ch.id === c.id;
            const card = el("figure", `srccard${used ? " used" : ""}`);
            const tile = new Tile(box.id, c);
            tile.frames = (box.pose || {})[c.id] || [];
            tile.v.preload = "auto";
            // thumbnail: the frame where MediaPipe sees the person most clearly (never an intro card or logo)
            const inWin = (f) => !(used && ch.window) || (f.t >= ch.window[0] && f.t <= ch.window[1]);
            const seen = tile.frames.filter((f) => f.lm && inWin(f)).map((f) => [f.t, f.lm.slice(1, 7).reduce((n, j) => n + j[2], 0)]);
            const clear = seen.length ? seen.reduce((a, b) => (b[1] > a[1] ? b : a))[0] : null;
            const thumbT = clear ?? (used && ch.window ? (ch.window[0] + ch.window[1]) / 2 : Math.min(3, (c.seconds || 6) / 3));
            tile.v.addEventListener("loadedmetadata", () => { try { tile.v.currentTime = thumbT; } catch {} }, { once: true });
            const drawAt = () => { let best = null; for (const f of tile.frames) { if (f.t <= tile.v.currentTime) best = f; else break; } tile.draw(best && best.lm); };
            tile.v.addEventListener("seeked", drawAt);
            $(".tile-badge", tile.root).textContent = used ? "copied by the robot" : r.reps ? `${r.reps} reps` : "no reps";
            tile.root.classList.add(used || r.ok ? "ok" : "no");
            const play = el("span", "play", "▶");
            tile.root.appendChild(play);
            let playing = false;
            tile.root.addEventListener("click", () => {
              playing = !playing;
              play.style.opacity = playing ? 0 : 1;
              if (playing) { tile.loop(used ? ch.window : null); } else { tile.v.pause(); }
            });
            const cap = el("figcaption", "", `<b>${esc(c.title.replace(/\.(webm|ogv|ogg|mp4)$/i, ""))}</b>
              <span>MediaPipe Pose: ${r.reps ?? "–"} reps · ${r.tracked != null ? Math.round(r.tracked * 100) : "–"}% tracked · ${esc(c.licence)}</span>`);
            const spark = el("canvas", "spark");
            card.append(tile.root, spark, cap);
            grid.appendChild(card);
            requestAnimationFrame(() => sparkline(spark, r.angle || [], (box.events.find((x) => x.type === "pose_done" && x.id === c.id) || {}), c.seconds));
          }
        },
      };
    }
    const s = summary || {};
    const pol = s.policy || {};
    const t = s.timings_s || {};
    const verdicts = ev("verdict");
    return {
      title: box.task,
      media: pol.eval_n ? `<video class="hero-video" src="/runs/${esc(box.id)}/showcase.mp4" autoplay muted loop playsinline controls></video>` : "<p class='note'>Still running.</p>",
      note: `Sim-tested: ${pol.eval_n ? `${pol.eval_success}/${pol.eval_n}` : "–"} unseen layouts in MuJoCo (Google DeepMind), official SO-101 model. The policy reads joint angles and object positions.`,
      glance: [[ev("clip").length, "clips pulled in the sandbox"], [`${verdicts.filter((v) => v.accept).length}/${verdicts.length}`, "verified by the VLM"],
        [ev("blocked").length, "hostile or junk fetches blocked"], [(s.dataset || {}).episodes ?? box.eps, "physics-checked episodes"],
        [pol.params ? pol.params.toLocaleString() : "–", "policy parameters"], [pol.eval_n ? `${pol.eval_success}/${pol.eval_n}` : "–", "unseen layouts solved"]],
      steps: [["Plan", "the agent picked the task and the people to learn from", t.plan], ["Scrape", `throwaway sandboxes fetch and re-encode openly licensed video (${host})`, t.scrape],
        ["Verify", "a VLM checks every clip", t.verify], ["Retarget", "MediaPipe hand track → SO-101 path → MuJoCo physics gate", t.motion],
        ["Tune", "a small policy trained on the CPU", t.train], ["Test", "fixed layouts the policy never saw", t.evaluate]],
      sources: rows(verdicts.map((v) => ({ ok: v.accept, tag: v.accept ? "used" : "rejected", title: v.title, meta: `${v.licence || ""} · ${v.reason || ""}` }))),
    };
  }

  function sparkline(cv, angle, done, seconds) {
    const W = (cv.width = cv.clientWidth * 2), H = (cv.height = 56);
    const g = cv.getContext("2d");
    const vals = angle.filter((x) => x != null);
    if (vals.length < 2) return;
    const lo = 0, hi = 180;
    g.strokeStyle = "rgba(241,207,138,0.95)"; g.lineWidth = 2.5; g.beginPath();
    angle.forEach((a, i) => { if (a == null) return; const x = (i / (angle.length - 1)) * W, y = H - ((a - lo) / (hi - lo)) * (H - 6) - 3; i ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.stroke();
  }

  function syncMoves(r) {
    const v = $("#sheet-video");
    if (!v || !r.moves) return;
    v.addEventListener("timeupdate", () => {
      let k = -1;
      r.moves.forEach((m, i) => { if (v.currentTime >= m.start_s) k = i; });
      $$("#sheet-moves li").forEach((li, i) => { li.className = i < k ? "done" : i === k ? "on" : ""; });
    });
  }

  async function openSheet(box) {
    const summary = box.kind === "robot" ? (await fetch(`/api/runs/${box.id}`).then((r) => r.json()).catch(() => ({}))).summary : null;
    const m = sheetModel(box, summary);
    $("#sheet-kicker").textContent = `ROBOT ${String(box.n).padStart(2, "0")} · SO-101 · simulated in MuJoCo`;
    $("#sheet-title").textContent = m.title;
    let dl = $("#sheet-dl");
    if (!dl) { dl = el("a", "dl big"); dl.id = "sheet-dl"; dl.textContent = "Download"; $(".sheet-head").insertBefore(dl, $("#sheet-close")); }
    dl.href = `/api/runs/${box.id}/download`; dl.setAttribute("download", "");
    dl.hidden = !box.root.classList.contains("is-ready");
    $("#sheet-body").innerHTML = `
      <section class="card span prompt-card"><h3>Prompt</h3><p class="prompt-text">“${esc(box.prompt)}”</p></section>
      <section class="card span"><h3>Result</h3>${m.media}<p class="note">${esc(m.note)}</p></section>
      <section class="card"><h3>At a glance</h3>${kv(m.glance)}</section>
      <section class="card"><h3>How it ran</h3>${stepList(m.steps)}
        ${box.root.classList.contains("is-ended") ? "" : `<button type="button" class="kill big" id="sheet-kill">Kill this robot's sandbox</button>`}</section>
      <section class="card span"><h3>Sources</h3>${m.sources}</section>
      <section class="card span"><h3>Full log</h3><div class="log">${box.log.map(([, text, cls]) => `<div class="${cls}">${esc(text)}</div>`).join("")}</div></section>`;
    $("#sheet").hidden = false;
    mark("sheet");
    document.body.classList.add("is-locked");
    $("#sheet-kill")?.addEventListener("click", () => killBox(box));
    if (m.onOpen) m.onOpen();
  }
  async function killBox(box) {
    if (box.root.classList.contains("is-ended")) return;
    box.setStatus("Killing", "no");
    await fetch(`/api/runs/${box.id}/kill`, { method: "POST" }).catch(() => {});
  }
  const closeSheet = () => { $("#sheet").hidden = true; document.body.classList.remove("is-locked"); };
  $("#sheet-close").addEventListener("click", closeSheet);
  $("#sheet-backdrop").addEventListener("click", closeSheet);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSheet(); });

  // ---------- replay and demo ----------
  async function replay(id, task, speed, maxGap) {
    const d = await fetch(`/api/runs/${id}`).then((r) => r.json());
    const kind = id.startsWith("workout-") ? "workout" : "robot";
    const text = task || (d.summary?.text || d.events[0]?.text || "").replace(/^I want to train a robot to /, "");
    const box = addBox(id, text, kind);
    if (!isFinite(speed)) { for (const e of d.events) box.handle(e); return box; }
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
    for (const ch of text) { inp.value += ch; mark("key"); await wait(45); }
    await wait(350);
    mark("submit");
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
        if (k === 0) { await wait(300); mark("glide"); await glideTo($("#fleet").offsetTop - 12, 1100); }
        await wait(Number(qs.get("stagger") || 3500));
      }
      const done = await Promise.all(runs);
      if (tag) tag.classList.remove("on");
      await wait(2000);
      if (qs.get("open") !== "0") await openSheet(done[done.length - 1]);
      window.__demoDone = true;
    })();
  } else if (qs.get("run")) {
    qs.get("run").split(",").forEach((id) => replay(id, null, speed, maxGap));
  } else if (qs.get("preload") !== "0") {
    // finished robots from earlier runs; a new prompt appears above them. ?preload=run,chess:run to choose
    (async () => {
      let list = qs.get("preload")?.split(",");
      if (!list) {  // the six skills in the public library, one per medium; the newest ends up on top
        const runs = await fetch("/api/runs").then((r) => r.json()).catch(() => ({ runs: [] }));
        const w = runs.runs.some((r) => r.id === "workout-030427-b43c") ? "workout-030427-b43c" : null;  // the curl in the public library
        list = [w || "workout-030427-b43c", "box-place", "box-tower", "task-cups", "task-hanoi", "chess:chess-140740-0d00"];
      }
      for (const item of list) {
        if (item.startsWith("chess:")) await addChessBox(item.slice(6)).catch(() => {});
        else if (PUZZLE[item]) await addPuzzleBox(item).catch(() => {});
        else await replay(item, null, Infinity, 0).catch(() => {});
      }
    })();
  }
})();
