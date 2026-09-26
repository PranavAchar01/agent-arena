"""Fine-tune a pretrained Hugging Face vision model as the policy's image encoder, on CPU, and measure it honestly.

Model: microsoft/resnet-18 (ImageNet-pretrained, Apache-2.0), pinned revision below. The policy sees a 96x96 image
from the overhead camera, the 6 joint positions and the elapsed time. It does NOT get the block or bowl positions:
it has to find them in the image. That is what makes the pretrained encoder do real work.

Data: the place-seeded run (real CC BY-SA chess video -> 6 human moves -> 136 physics-checked episodes), replayed
exactly from runs/place-seeded/{shapes.json,episodes.jsonl} with the camera on.

Variants, all scored on the same 50 fixed unseen layouts (policy.EVAL_SEED stream):
  frozen     pretrained ResNet-18 frozen, only the action head trained   ("pretrained, not fine-tuned")
  finetune   pretrained ResNet-18 fine-tuned end to end with the head     ("fine-tuned")
  scratch    the state-based MLP from the pipeline (gets object positions directly), for reference

usage: pretrained_policy.py {frozen|finetune|scratch} [iters]
"""

import json
import resource
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn
from transformers import ResNetModel

from arena.policy import CHUNK, EVAL_SEED, MAX_T, Runner
from arena.layouts import sample
from arena.sim.ik import IK
from arena.sim.retarget import MotionShape, plan, to_joints
from arena.sim.scene import DT, HOME, Scene

MID, REV = "microsoft/resnet-18", "65a5785d9156231087c481e0c7dd33a5ff6f7e3e"
RUN = Path("runs/hocap-place")
OUT = Path("runs/pretrained")
RES, EVERY, N_EVAL = 96, 4, 50
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
torch.set_num_threads(4)
OUT.mkdir(parents=True, exist_ok=True)


def to_tensor(img):
    return torch.from_numpy(
        ((img.astype(np.float32) / 255).transpose(2, 0, 1) - MEAN) / STD
    )


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = ResNetModel.from_pretrained(MID, revision=REV)
        self.head = nn.Sequential(
            nn.Linear(512 + 7, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, CHUNK * 6),
        )

    def feats(self, x):
        return self.enc(pixel_values=x).pooler_output.flatten(1)

    def forward(self, x, s, f=None):
        f = self.feats(x) if f is None else f
        return self.head(torch.cat([f, s], 1)).view(-1, CHUNK, 6)


def build_data():
    cache = OUT / "data.npz"
    if cache.exists():
        d = np.load(cache)
        return d["img"], d["state"], d["act"], float(d["render_s"])
    shapes = {
        s["source"]: MotionShape.from_json(s)
        for s in json.loads((RUN / "shapes.json").read_text())
    }
    meta = [
        json.loads(line) for line in (RUN / "episodes.jsonl").read_text().splitlines()
    ]
    sc = Scene.make("place")
    ik = IK(sc.model)
    r = mujoco.Renderer(sc.model, RES, RES)
    imgs, states, acts = [], [], []
    t0 = time.time()
    for m in meta:
        sc.reset(m["block"], m["target"], m["yaw"])
        sc._ez_sign = 1.0
        pts = plan(sc, shapes[m["source"]])
        q, _, _ = ik.solve(pts[len(pts) // 3][0], HOME, pts[0][1])
        _, R = ik.fk(q)
        sc._ez_sign = float(np.sign(R[:, 2] @ np.array([*pts[0][1], 0.0])) or 1.0)
        pts = plan(sc, shapes[m["source"]])
        qs, _, _ = to_joints(sc, pts, ik)
        T = len(qs)
        for k, target in enumerate(qs):
            if k % EVERY == 0:
                r.update_scene(sc.data, camera="top")
                imgs.append(r.render().copy())
                states.append(np.append(sc.q(), k * DT / MAX_T))
                acts.append(qs[np.minimum(np.arange(k, k + CHUNK), T - 1)])
            sc.step(target)
    render_s = time.time() - t0
    img, state, act = (
        np.array(imgs, np.uint8),
        np.array(states, np.float32),
        np.array(acts, np.float32),
    )
    np.savez(cache, img=img, state=state, act=act, render_s=render_s)
    return img, state, act, render_s


def evaluate(policy, norm, n=N_EVAL):
    sc = Scene.make("place")
    r = mujoco.Renderer(sc.model, RES, RES)
    rng = np.random.default_rng(EVAL_SEED + 1)
    layouts = [sample("place", rng) for _ in range(n)]
    smu, ssd, amu, asd = norm
    res = []
    policy.eval()
    for bxy, txy, yaw in layouts:
        sc.reset(bxy, txy, yaw)
        steps = int(9.0 / DT)
        buf, wsum = np.zeros((steps + CHUNK, 6)), np.zeros(steps + CHUNK)
        for k in range(steps):
            r.update_scene(sc.data, camera="top")
            x = to_tensor(r.render())[None]
            s = torch.from_numpy(
                ((np.append(sc.q(), k * DT / MAX_T) - smu) / ssd).astype(np.float32)
            )[None]
            with torch.no_grad():
                a = policy(x, s)[0].numpy() * asd + amu
            w = np.exp(-0.05 * np.arange(CHUNK))
            buf[k : k + CHUNK] += a * w[:, None]
            wsum[k : k + CHUNK] += w
            sc.step(buf[k] / wsum[k])
        res.append(bool(sc.success()))
    return res


def main():
    mode = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    report = {
        "mode": mode,
        "model": MID,
        "revision": REV,
        "task": "place",
        "eval_n": N_EVAL,
    }
    if mode == "scratch":
        sc = Scene.make("place")
        rng = np.random.default_rng(EVAL_SEED + 1)
        runner = Runner(RUN / "policy.pt")
        t0 = time.time()
        res = [runner.rollout(sc, *sample("place", rng))[0] for _ in range(N_EVAL)]
        report.update(
            eval=res,
            success=int(sum(res)),
            eval_s=round(time.time() - t0, 1),
            params=166008,
            note="state-based MLP from scratch; gets block and bowl positions directly",
        )
    else:
        img, state, act, render_s = build_data()
        smu, ssd = state.mean(0), state.std(0) + 1e-6
        amu, asd = act.reshape(-1, 6).mean(0), act.reshape(-1, 6).std(0) + 1e-6
        S = torch.from_numpy((state - smu) / ssd)
        Y = torch.from_numpy((act - amu) / asd)
        policy = Policy()
        total = sum(p.numel() for p in policy.parameters())
        enc_params = sum(p.numel() for p in policy.enc.parameters())
        t0 = time.time()
        if mode == "frozen":
            for p in policy.enc.parameters():
                p.requires_grad = False
            policy.enc.eval()
            with torch.no_grad():  # frozen: encode every frame once
                F = torch.cat(
                    [
                        policy.feats(
                            torch.stack([to_tensor(im) for im in img[i : i + 256]])
                        )
                        for i in range(0, len(img), 256)
                    ]
                )
            opt = torch.optim.AdamW(
                policy.head.parameters(), lr=1e-3, weight_decay=1e-4
            )
            for it in range(iters):
                b = torch.randint(0, len(F), (256,))
                loss = nn.functional.smooth_l1_loss(policy(None, S[b], F[b]), Y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            trained = sum(p.numel() for p in policy.head.parameters())
        else:
            opt = torch.optim.AdamW(
                [
                    {"params": policy.enc.parameters(), "lr": 1e-4},
                    {"params": policy.head.parameters(), "lr": 1e-3},
                ],
                weight_decay=1e-4,
            )
            policy.train()
            for it in range(iters):
                b = torch.randint(0, len(img), (32,))
                x = torch.stack([to_tensor(img[i]) for i in b.tolist()])
                loss = nn.functional.smooth_l1_loss(policy(x, S[b]), Y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                if it % 100 == 0:
                    print(
                        f"iter {it} loss {loss.item():.4f} {time.time() - t0:.0f}s",
                        flush=True,
                    )
            trained = total
        train_s = time.time() - t0
        t0 = time.time()
        res = evaluate(policy, (smu, ssd, amu, asd))
        report.update(
            frames=int(len(img)),
            render_s=round(render_s, 1),
            iters=iters,
            train_s=round(train_s, 1),
            params_total=int(total),
            params_encoder=int(enc_params),
            params_trained=int(trained),
            final_loss=round(float(loss), 4),
            eval=res,
            success=int(sum(res)),
            eval_s=round(time.time() - t0, 1),
        )
    report["peak_rss_gb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2
    )
    (OUT / f"{mode}.json").write_text(json.dumps(report, indent=1))
    print(
        "RESULT",
        json.dumps({k: v for k, v in report.items() if k != "eval"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
