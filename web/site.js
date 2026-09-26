// Handoff: compose a sentence, start a run, render its event stream stage by stage.
// ?run=<id> replays a recorded run (&speed=N compresses its real timing N times).
(() => {
  const $ = (s) => document.querySelector(s);
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const task = $("#task");
  let family = null;
  let runId = null;
  let t0 = 0;
  let clockTimer = null;
  const counts = { found: 0, downloaded: 0, blocked: 0, accepted: 0, episodes: 0 };
  const stageT = {};

  document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => {
    task.textContent = c.dataset.text;
    family = c.dataset.family;
    task.focus();
  }));
  task.addEventListener("input", () => { family = null; });
  task.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); $("#composer").requestSubmit(); } });

  function setStat(k, v) {
    const s = document.querySelector(`.stat[data-k="${k}"]`);
    if (!s) return;
    s.querySelector("dd").textContent = v;
    s.classList.remove("bump"); void s.offsetWidth; s.classList.add("bump");
  }
  function bump(k, n = 1) { counts[k] += n; setStat(k, counts[k]); }

  function stage(name) {
    const order = ["scrape", "verify", "motion", "train"];
    const i = order.indexOf(name);
    order.forEach((n, j) => {
      const li = $(`#st-${n}`);
      if (j < i && li.classList.contains("is-on")) {
        li.classList.replace("is-on", "is-done");
        if (stageT[n]) li.querySelector(".stage-took").textContent = `${((Date.now() - stageT[n]) / 1000).toFixed(0)} s`;
      }
      if (j === i && !li.classList.contains("is-on") && !li.classList.contains("is-done")) { li.classList.add("is-on"); stageT[n] = Date.now(); }
    });
  }
  function finishAll() {
    ["scrape", "verify", "motion", "train"].forEach((n) => {
      const li = $(`#st-${n}`);
      if (li.classList.contains("is-on")) {
        li.classList.replace("is-on", "is-done");
        if (stageT[n]) li.querySelector(".stage-took").textContent = `${((Date.now() - stageT[n]) / 1000).toFixed(0)} s`;
      }
    });
  }
  function feed(text, cls = "") {
    const f = $("#feed");
    const line = el("div", cls, esc(text));
    f.appendChild(line);
    f.scrollTop = f.scrollHeight;
  }
  const media = (name) => `/runs/${runId}/frames/${name}`;
  const short = (s, n = 70) => (String(s).length > n ? String(s).slice(0, n - 1) + "…" : String(s));
  const hostOf = (u) => { try { return new URL(u).host; } catch { return ""; } };

  const handlers = {
    plan(e) {
      stage("scrape");
      $("#run-title").textContent = `Train an SO-101 to ${e.summary || task.textContent}`;
      $("#queries").innerHTML = `<span class="q">family <b>${esc(e.family)}</b></span>` + e.queries.map((q) => `<span class="q">${esc(q)}</span>`).join("");
    },
    sandbox_start(e) {
      feed(`$ sandbox up: ${e.container} · ${e.caps.cpus} CPU · ${e.caps.memory} RAM · ${e.caps.pids} pids · ${e.caps.seconds}s kill · read-only root · no network creds`, "l-sys");
    },
    sandbox(e) { feed(`  inside: uid ${e.user}, ${e.env_secrets} secrets in env, writable ${e.writable.join(" ")}`, "l-sys"); },
    search(e) { counts.found += e.hits; setStat("found", counts.found); feed(`search ${e.source}: "${e.query}" → ${e.hits}`); },
    page(e) { feed(`visited ${hostOf(e.url)}${new URL(e.url).pathname}: ${e.videos} video link(s), ${e.scripts_ignored} script(s) never executed`); },
    candidate() {},
    skip(e) { if (/licen/.test(e.reason)) feed(`skip  ${short(e.title, 50)}: ${e.reason}`, "l-no"); },
    warn(e) { feed(e.message, "l-sys"); },
    picked(e) { if (!e.of) { feed("source this run: an open dataset, HO-Cap (CC BY 4.0), read inside a fresh sandbox", "l-sys"); return; } feed(e.seeded ? `seeded run: ${e.n} of ${e.of} open-licence results picked by hand (the VLM still checks each)` : `agent picked ${e.n} of ${e.of} open-licence results by title`, "l-sys"); },
    download(e) { feed(`fetch ${short(e.title, 60)} (${e.licence})`); },
    clip(e) { bump("downloaded"); feed(`kept  ${short(e.title, 60)} · ${e.seconds}s re-encoded`, "l-ok"); },
    blocked(e) {
      bump("blocked");
      feed(`BLOCKED ${e.reason}`, "l-no");
      const c = el("div", "bcard", `<b>Blocked in sandbox</b><p>${esc(e.reason)}</p><code>${esc(short(e.url, 90))}</code>`);
      $("#blocked").appendChild(c);
    },
    sandbox_destroyed(e) { feed(`$ sandbox ${e.container} destroyed after ${e.seconds}s`, "l-sys"); },
    verifying(e) {
      stage("verify");
      const c = el("div", "clip", `<img alt="" src="${media(e.id + "_sheet.jpg")}" onerror="this.style.visibility='hidden'"><div class="meta"><div class="t">${esc(e.title)}</div><span class="verdict v-wait">checking…</span></div>`);
      c.id = `clip-${e.id}`;
      $("#clips").appendChild(c);
    },
    verdict(e) {
      let c = $(`#clip-${e.id}`);
      if (!c) { handlers.verifying(e); c = $(`#clip-${e.id}`); }
      c.querySelector("img").src = media(e.id + "_sheet.jpg");
      c.querySelector(".meta").innerHTML = `<div class="t">${esc(e.title)}</div><div class="lic">${esc(e.licence)} · ${esc(e.author || e.source)}</div>` +
        `<span class="verdict ${e.accept ? "v-ok" : "v-no"}">${e.accept ? "Verified" : "Rejected"}</span><p class="why">${esc(e.reason)}</p>`;
      c.classList.toggle("is-rejected", !e.accept);
      if (e.accept) bump("accepted");
    },
    moves(e) {
      stage("motion");
      $("#moves").appendChild(el("div", "mv", `<span>${e.id}</span><span>${e.found} clean move(s) found, ${e.rejected} rejected${e.reasons.length ? ": " + esc(e.reasons.join("; ")) : ""}</span><span></span>`));
    },
    motion(e) {
      const ok = e.probe.filter(Boolean).length;
      $("#moves").appendChild(el("div", "mv", `<span>${esc(e.move.split("@")[1] || "")}</span><span class="bar"><i style="width:${(ok / 3) * 100}%"></i></span><span class="${e.ok ? "ok" : "no"}">${e.ok ? "kept" : "dropped"} · ${ok}/3 in physics · lift ${e.max_lift}</span>`));
    },
    episodes(e) {
      bump("episodes", e.kept);
      $("#moves").appendChild(el("div", "mv", `<span>${esc((e.source.split("@")[1]) || "")}</span><span>replayed on ${e.tried} new layouts</span><span class="ok">${e.kept} episodes passed</span>`));
    },
    dataset(e) { feed; $("#train-line").textContent = `${e.episodes} physics-checked episodes (${e.frames} frames). A small policy trains on the CPU, then runs on 20 fixed layouts it never saw.`; },
    train(e) { stage("train"); $("#evaldots").innerHTML = `<span>training · step ${e.iter}/${e.iters} · loss ${e.loss}</span>`; },
    trained(e) { stage("train"); $("#evaldots").innerHTML = `<span>trained ${e.params.toLocaleString()} params in ${e.train_seconds}s on CPU · testing…</span>`; },
    evaluated(e) {
      stage("train");
      const dots = $("#evaldots");
      dots.innerHTML = "";
      if (!e.n) { dots.innerHTML = `<span>${esc(e.reason || "nothing to train")}</span>`; setStat("success", "0"); return; }
      fetch(`/api/runs/${runId}`).then((r) => r.json()).then((d) => {
        const res = d.summary?.policy?.eval || [];
        res.forEach((ok, i) => setTimeout(() => dots.appendChild(el("i", ok ? "ok" : "no")), i * 90));
        setTimeout(() => dots.appendChild(el("span", "", `${e.success}/${e.n} unseen layouts`)), res.length * 90);
      });
      setStat("success", `${e.success}/${e.n}`);
      const v = $("#policy");
      v.src = `/runs/${runId}/policy.mp4`;
      v.hidden = false;
      v.play().catch(() => {});
    },
    done(e) { if (e && "kept" in e) { feed(`sandbox finished: kept ${e.kept} of ${e.candidates}`, "l-sys"); return; } finishAll(); $("#rec-dot").classList.remove("is-live"); clearInterval(clockTimer); $("#kicker").textContent = "Run complete"; $("#go").disabled = false; },
    error(e) { feed(`error: ${e.message}`, "l-no"); handlers.done(); },
  };
  const handle = (e) => (handlers[e.type] || (() => {}))(e);

  function openRun(title) {
    $("#output").classList.add("is-open");
    $("#rec-dot").classList.add("is-live");
    $("#run-title").textContent = title;
    t0 = Date.now();
    clearInterval(clockTimer);
    clockTimer = setInterval(() => {
      const s = Math.floor((Date.now() - t0) / 1000);
      $("#clock").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    }, 500);
    if (!document.body.classList.contains("is-film")) setTimeout(() => $("#output").scrollIntoView({ behavior: "smooth", block: "start" }), 250);
  }

  $("#composer").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const text = task.textContent.trim() || task.dataset.placeholder;
    $("#go").disabled = true;
    const r = await fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: `I want to train a robot to ${text}`, family }) });
    if (!r.ok) { $("#go-hint").textContent = (await r.json()).detail || "Could not start"; $("#go").disabled = false; return; }
    runId = (await r.json()).id;
    openRun(`Train an SO-101 to ${text}`);
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/runs/${runId}/ws`);
    ws.onmessage = (m) => handle(JSON.parse(m.data));
  });

  // Replay a recorded run with its real pacing compressed.
  const qs = new URLSearchParams(location.search);
  if (qs.get("run")) {
    runId = qs.get("run");
    const speed = Number(qs.get("speed") || 8);
    fetch(`/api/runs/${runId}`).then((r) => r.json()).then(async (d) => {
      const text = (d.summary?.text || "").replace(/^I want to train a robot to /, "");
      task.textContent = text;
      $("#kicker").textContent = "Recorded run";
      const wait = (ms) => new Promise((res) => setTimeout(res, ms));
      if (qs.get("film")) {
        // film mode: type the sentence, press the button, then follow the run down the page
        document.body.classList.add("is-film");
        task.textContent = "";
        await wait(Number(qs.get("delay") || 900));
        for (const ch of text) { task.textContent += ch; await wait(55); }
        await wait(600);
        $("#go").classList.add("is-pressed");
        await wait(180);
        $("#go").classList.remove("is-pressed");
        let y = scrollY;
        setInterval(() => {
          const st = document.querySelector(".stage.is-on") || document.querySelector(".stage.is-done:last-of-type");
          if (!st || !$("#output").classList.contains("is-open")) return;
          const r = st.getBoundingClientRect();
          const top = scrollY + r.top - 150;
          const bottom = scrollY + r.bottom - innerHeight + 60;
          const goal = Math.max(Math.min(top, bottom), document.querySelector("#output").offsetTop - 16);
          y += (Math.min(goal, top) - y) * 0.12;
          window.scrollTo(0, y);
        }, 33);
      } else if (qs.get("autostart") !== "0") await wait(Number(qs.get("delay") || 1200));
      openRun(`Train an SO-101 to ${text}`);
      let prev = null;
      for (const e of d.events) {
        const at = e.at ?? e.t;
        if (prev != null && at != null) await new Promise((res) => setTimeout(res, Math.min(2500, Math.max(40, ((at - prev) * 1000) / speed))));
        if (at != null) prev = at;
        handle(e);
      }
      handlers.done();
    });
  }
})();
