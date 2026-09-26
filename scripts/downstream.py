"""Stages 3-4 only, on clips a human (here: Claude's blind hand-check) accepted although the VLM rejected them.
Recorded as override=True everywhere. Same motion gates, physics gates, dataset, policy and evaluation code."""
import json, sys, time
from pathlib import Path
import numpy as np
from arena.motion import shapes_from_pose, track_hands
from arena.sim.scene import Scene
from arena.sim.ik import IK
from arena.sim.retarget import replay
from arena.layouts import PROBES, sample
from arena.policy import train, evaluate, N_EVAL
from arena.pipeline import renderer, write_mp4
src, task, out, ids = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4].split(",")
aug = int(sys.argv[5]) if len(sys.argv) > 5 else 24
out.mkdir(parents=True, exist_ok=True)
sc = Scene.make(task); ik = IK(sc.model); rep = {"override": True, "task": task, "clips": {}}
shapes = []; t0 = time.time()
for cid in ids:
    clip = src / "clips" / f"{cid}.mp4"
    pose = track_hands(clip, [0, 999], out / f"pose_{cid}.json")
    good, bad = shapes_from_pose(pose, task, cid)
    kept = []
    for s, q in good:
        probe = [replay(sc, s, b, t, y, ik=ik).success for b, t, y in PROBES[task]]
        kept.append({**q, "probe": probe, "ok": sum(probe) >= 2})
        if sum(probe) >= 2: shapes.append(s)
    rep["clips"][cid] = {"moves_found": len(good), "moves_rejected": len(bad), "reasons": sorted({b["reason"] for b in bad}), "moves": kept}
    print(cid, rep["clips"][cid], flush=True)
rep["motion_s"] = round(time.time() - t0, 1)
rng = np.random.default_rng(1); eps = []; t0 = time.time(); tried = 0
for s in shapes:
    for _ in range(aug):
        b, t, y = sample(task, rng); tried += 1
        ep = replay(sc, s, b, t, y, ik=ik)
        if ep.success: eps.append((ep.obs, ep.act))
rep["dataset"] = {"shapes": len(shapes), "tried": tried, "episodes": len(eps), "frames": int(sum(len(a) for _, a in eps)), "seconds": round(time.time() - t0, 1)}
print(rep["dataset"], flush=True)
(out / "shapes.json").write_text(json.dumps([s.to_json() for s in shapes]))
if len(eps) >= 5:
    np.savez_compressed(out / "dataset.npz", obs=np.concatenate([o for o, _ in eps]), act=np.concatenate([a for _, a in eps]))
    info = train(eps, out / "policy.pt"); print(info, flush=True)
    t0 = time.time(); res, vids = evaluate(task, out / "policy.pt", render_first=4, render=renderer(sc.model))
    write_mp4([f for _, fr in vids for f in fr], out / "policy.mp4")
    rep["policy"] = {**info, "eval": res, "eval_success": int(sum(res)), "eval_n": N_EVAL, "eval_seconds": round(time.time() - t0, 1)}
    print("EVAL", sum(res), "/", N_EVAL, flush=True)
(out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
