// Handoff Chess: prompt -> agent + sandbox events in the terminal -> the robot's rehearsed replay, synced to the
// move list and a mini board.   ?demo=1 types the prompt by itself (for the live slot).
(() => {
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const short = (s, n = 60) => (String(s).length > n ? String(s).slice(0, n - 1) + "…" : String(s));
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const input = $("#prompt");
  const log = $("#log");
  const video = $("#video");
  let moves = [];

  const line = (text, cls = "") => {
    const d = document.createElement("div");
    d.className = cls;
    d.textContent = text;
    log.appendChild(d);
    while (log.children.length > 60) log.firstChild.remove();
  };

  fetch("/api/config").then((r) => r.json()).then((c) => {
    $("#backend").textContent = c.sandbox === "vultr" ? "Sandboxes on Vultr · one arm, both sides" : "One arm. Both sides of the board.";
  }).catch(() => {});

  document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => { input.value = c.dataset.text; input.focus(); }));

  $("#composer").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const text = input.value.trim() || input.placeholder;
    $("#stage").hidden = false;
    log.innerHTML = "";
    $("#moves").innerHTML = "";
    $("#stats").hidden = true;
    $("#tag").hidden = true;
    $("#caption").hidden = true;
    $("#wait").hidden = false;
    $("#wait span").textContent = "finding the game…";
    video.removeAttribute("src");
    board(null);
    line(`> ${text}`, "t-sys");
    const r = await fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, mode: "chess" }) });
    if (!r.ok) { line("could not start", "t-no"); return; }
    const { id } = await r.json();
    $("#stage").scrollIntoView({ behavior: "smooth", block: "start" });
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/runs/${id}/ws`);
    ws.onmessage = (m) => handle(JSON.parse(m.data));
  });

  const FORMAT = {
    queued: () => null,
    plan: (e) => [`agent> ${e.white || "?"} vs ${e.black || "?"}, ${e.event || ""} ${e.year || ""} ${e.game ? "game " + e.game : ""}`.trim(), "t-sys"],
    model_call: (e) => [`${e.kind.toLowerCase()}> ${e.model} · ${e.ms} ms`, "t-dim"],
    vm_create: (e) => [`vultr> VM ${e.instance.slice(0, 8)} · ${e.plan} · ${e.region}`, "t-sys"],
    vm_ready: (e) => [`vultr> ready in ${e.seconds}s`, "t-ok"],
    vm_destroyed: (e) => [`vultr> VM ${e.instance.slice(0, 8)} deleted after ${e.seconds}s`, "t-sys"],
    sandbox_start: (e) => [`$ docker run --rm --read-only --cap-drop ALL ${e.container} (${e.host === "local" ? "local" : e.host})`, "t-sys"],
    sandbox: (e) => [`  uid ${e.user} · ${e.env_secrets} secrets in env`, "t-dim"],
    search: (e) => [`scrapling> ${e.source}: "${short(e.query, 40)}" → ${e.hits}`, "t-info"],
    candidate: (e) => [`  · ${short(e.title, 58)}`, "t-dim"],
    picked: (e) => [`agent> picked ${e.n} of ${e.of}: ${e.titles.map((t) => short(t, 36)).join(" · ")}`, "t-sys"],
    page: (e) => [`scrapling> GET ${short(decodeURIComponent(new URL(e.url).pathname), 44)} → ${e.move_lists} move lists`, "t-info"],
    blocked: (e) => [`✗ blocked ${short(e.reason, 60)}`, "t-no"],
    sandbox_destroyed: (e) => [`$ ${e.container} destroyed after ${e.seconds}s`, "t-sys"],
    validate: (e) => [`python-chess> list ${e.index}: ${e.ok ? `${e.plies} legal moves ✓` : e.reason}`, e.ok ? "t-ok" : "t-no"],
    game: (e) => [`game> ${e.plies} moves from ${short(e.title, 40)}`, "t-ok"],
    replay: (e) => [`robot> ${e.plies} moves, ${e.transfers} pick-and-places, all rehearsed in physics ✓`, "t-ok"],
    not_rehearsed: (e) => [`robot> ${e.message}`, "t-no"],
    warn: (e) => [`· ${e.message}`, "t-dim"],
    error: (e) => [`error: ${e.message}`, "t-no"],
    killed: () => ["■ killed by operator", "t-no"],
    done: (e) => (e.timings_s ? [`done in ${e.timings_s.total}s`, "t-dim"] : [`  sandbox finished · ${e.candidates} found`, "t-dim"]),
  };

  function handle(e) {
    const f = FORMAT[e.type];
    const out = f && f(e);
    if (out) line(...out);
    if (e.type === "game") { $("#term-title").textContent = `agent · ${short(e.title, 40)}`; showMoves(e.sans); }
    if (e.type === "replay") play(e);
    if (e.type === "not_rehearsed") $("#wait span").textContent = "not rehearsed in physics yet";
  }

  function showMoves(sans) {
    $("#moves").innerHTML = sans.map((s, i) => `<li data-i="${i}">${i % 2 === 0 ? `<b>${i / 2 + 1}.</b>` : ""}${esc(s)}</li>`).join("");
    document.querySelectorAll("#moves li").forEach((li) => li.addEventListener("click", () => {
      const m = moves[+li.dataset.i];
      if (m) { video.currentTime = m.start_s; video.play().catch(() => {}); }
    }));
  }

  function play(e) {
    moves = e.moves;
    $("#tag").textContent = `simulated SO-101 · both sides · ${e.speed.toFixed(1)}× speed`;
    $("#tag").hidden = false;
    const s = $("#stats");
    const kv = [
      [e.plies, "moves, both colours"],
      [e.captures, "captures (pieces to the tray)"],
      [`${e.max_err_mm} mm`, "worst placement off square centre"],
      [e.transfers, "pick-and-places, each rehearsed first"],
      [`${Math.round(e.sim_s)} s`, "of robot time, shown in " + Math.round(e.video_s) + " s"],
    ];
    s.innerHTML = kv.map(([a, b]) => `<div><dd>${esc(a)}</dd><dt>${esc(b)}</dt></div>`).join("");
    s.hidden = false;
    video.poster = e.poster;
    video.src = e.video;
    video.muted = true;
    video.oncanplay = () => { $("#wait").hidden = true; video.play().catch(() => {}); };
    video.load();
  }

  video.addEventListener("timeupdate", () => {
    if (!moves.length) return;
    const t = video.currentTime;
    let k = -1;
    moves.forEach((m, i) => { if (t >= m.start_s) k = i; });
    document.querySelectorAll("#moves li").forEach((li, i) => { li.className = i < k ? "done" : i === k ? "on" : ""; });
    const cap = $("#caption");
    if (k >= 0) {
      const m = moves[k];
      cap.innerHTML = `${Math.floor(k / 2) + 1}.${k % 2 ? ".." : ""} ${esc(m.san)}<small>${k % 2 ? "Black" : "White"}</small>`;
      cap.hidden = false;
      board(t >= m.end_s ? m.fen : k > 0 ? moves[k - 1].fen : null, m);
    }
  });

  const GLYPH = { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };
  const START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";
  function board(fen, move) {
    const rows = (fen || START).split(" ")[0].split("/");
    const sq = [];
    rows.forEach((row, r) => {
      for (const ch of row) {
        if (/\d/.test(ch)) for (let k = 0; k < +ch; k++) sq.push([r, sq.length % 8, null]);
        else sq.push([r, sq.length % 8, ch]);
      }
    });
    const hl = new Set();
    if (move && move.uci) { hl.add(move.uci.slice(0, 2)); hl.add(move.uci.slice(2, 4)); }
    $("#mini").innerHTML = sq.map(([r, c, p]) => {
      const name = "abcdefgh"[c] + (8 - r);
      const cls = `${(r + c) % 2 ? "d" : "l"}${p ? (p === p.toUpperCase() ? " w" : " b") : ""}${hl.has(name) ? " hl" : ""}`;
      return `<i class="${cls}">${p ? GLYPH[p.toLowerCase()] : ""}</i>`;
    }).join("");
  }
  board(null);

  const replayKey = new URLSearchParams(location.search).get("replay");
  if (replayKey && /^[0-9a-f]{16}$/.test(replayKey)) {  // backup: play a rehearsed game without the live agent run
    fetch(`/api/chess/${replayKey}`).then((r) => r.json()).then((e) => {
      $("#stage").hidden = false;
      line("> replay of a rehearsed game (no live agent run)", "t-dim");
      showMoves(e.moves.map((m) => m.san));
      play(e);
    });
  }

  if (new URLSearchParams(location.search).has("demo")) {
    (async () => {
      await wait(900);
      const text = "Replay Deep Blue vs. Kasparov, 1997, Game 6";
      input.focus();
      for (const ch of text) { input.value += ch; await wait(38); }
      await wait(400);
      $("#composer").requestSubmit();
    })();
  }
})();
