"""A small behaviour-cloning policy per task, trained on the CPU, and its honest evaluation.

Policy: an MLP that maps the robot state (6 joint positions), the block and target positions and the elapsed time
to the next CHUNK joint targets (action chunking), blended at run time with temporal ensembling. State-based: no
camera input. That is stated wherever the number is shown.

Evaluation: N_EVAL fixed layouts drawn from a separate random stream (never used for training data), a full
physics rollout each, success judged by the same scene.success() used to gate the data.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .sim.scene import DT, GRIP_CLOSED, HOME, Scene

CHUNK, HIDDEN = 20, 256
N_EVAL = 20
EVAL_SEED = 90_000
MAX_T = 12.0


class ChunkMLP(nn.Module):
    def __init__(self, obs_dim=13, act_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, CHUNK * act_dim),
        )
        self.act_dim = act_dim

    def forward(self, x):
        return self.net(x).view(-1, CHUNK, self.act_dim)


def _with_time(obs: np.ndarray) -> np.ndarray:
    t = (np.arange(len(obs)) * DT / MAX_T)[:, None]
    return np.concatenate([obs, t], 1)


def windows(episodes):
    X, Y = [], []
    for ob, ac in episodes:
        o = _with_time(ob)
        T = len(ac)
        idx = np.minimum(np.arange(T)[:, None] + np.arange(CHUNK)[None], T - 1)
        X.append(o)
        Y.append(ac[idx])
    return np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32)


def train(episodes, out: Path, iters: int = 3000, seed: int = 0, log=None) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    X, Y = windows(episodes)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    amu, asd = Y.reshape(-1, 6).mean(0), Y.reshape(-1, 6).std(0) + 1e-6
    Xn = torch.tensor((X - mu) / sd)
    Yn = torch.tensor((Y - amu) / asd)
    model = ChunkMLP(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    t0 = time.time()
    loss = None
    for it in range(iters):
        b = torch.randint(0, len(Xn), (256,))
        loss = nn.functional.smooth_l1_loss(model(Xn[b]), Yn[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and it % 500 == 0:
            log(
                {
                    "type": "train",
                    "iter": it,
                    "iters": iters,
                    "loss": round(float(loss), 4),
                }
            )
    wall = time.time() - t0
    params = sum(p.numel() for p in model.parameters())
    torch.save(
        {
            "state": model.state_dict(),
            "mu": mu,
            "sd": sd,
            "amu": amu,
            "asd": asd,
            "obs_dim": X.shape[1],
        },
        out,
    )
    return {
        "samples": int(len(X)),
        "iters": iters,
        "train_seconds": round(wall, 1),
        "params": int(params),
        "final_loss": round(float(loss), 4),
        "bytes": out.stat().st_size,
    }


class Runner:
    def __init__(self, path: Path):
        ck = torch.load(path, weights_only=False)
        self.m = ChunkMLP(ck["obs_dim"])
        self.m.load_state_dict(ck["state"])
        self.m.eval()
        self.ck = ck

    def rollout(
        self, sc: Scene, block_xy, target_xy, block_yaw=0.0, seconds=9.0, render=None
    ):
        sc.reset(block_xy, target_xy, block_yaw)
        n = int(seconds / DT)
        buf = np.zeros((n + CHUNK, 6))
        wsum = np.zeros(n + CHUNK)
        frames = []
        c = self.ck
        for k in range(n):
            o = np.append(sc.obs(), k * DT / MAX_T).astype(np.float32)
            with torch.no_grad():
                a = (
                    self.m(torch.tensor((o - c["mu"]) / c["sd"])[None])[0].numpy()
                    * c["asd"]
                    + c["amu"]
                )
            w = np.exp(
                -0.05 * np.arange(CHUNK)
            )  # temporal ensembling: older predictions weigh more
            buf[k : k + CHUNK] += a * w[:, None]
            wsum[k : k + CHUNK] += w
            sc.step(buf[k] / wsum[k])
            if render is not None and k % 2 == 0:
                frames.append(render(sc))
        return sc.success(), frames


def eval_layouts(task: str, n: int = N_EVAL):
    from .layouts import sample

    rng = np.random.default_rng(EVAL_SEED + {"push": 0, "place": 1, "stack": 2, "unjar": 3, "tower": 4}[task])
    return [sample(task, rng) for _ in range(n)]


def evaluate(
    task: str, path: Path, n: int = N_EVAL, render_first: int = 0, render=None
):
    sc = Scene.make(task)
    r = Runner(path)
    results, videos = [], []
    for i, (bxy, txy, yaw) in enumerate(eval_layouts(task, n)):
        ok, frames = r.rollout(
            sc, bxy, txy, yaw, render=render if i < render_first else None
        )
        results.append(bool(ok))
        if frames:
            videos.append((ok, frames))
    return results, videos


__all__ = ["train", "evaluate", "Runner", "HOME", "GRIP_CLOSED"]
