"""Damped least squares IK for the SO-101 gripperframe site.

Targets: the site position, the jaws pointing straight down (site +x onto world -z), and optionally the jaws'
closing axis (site +z) turned onto a horizontal direction. Five joints cannot always satisfy all six residuals;
the closing-axis term has a lower weight so position and "down" win.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .scene import JOINTS

DOWN = np.array([0.0, 0.0, -1.0])


class IK:
    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.d = mujoco.MjData(model)
        self.site = model.site("gripperframe").id
        self.qadr = np.array([model.joint(j).qposadr[0] for j in JOINTS[:5]])
        self.dadr = np.array([model.joint(j).dofadr[0] for j in JOINTS[:5]])
        self.lo = model.jnt_range[[model.joint(j).id for j in JOINTS[:5]], 0]
        self.hi = model.jnt_range[[model.joint(j).id for j in JOINTS[:5]], 1]

    def fk(self, q5: np.ndarray):
        self.d.qpos[self.qadr] = q5
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(
            3, 3
        ).copy()

    def solve(
        self,
        pos,
        q0,
        close_dir=None,
        iters: int = 60,
        w_down: float = 0.3,
        w_close: float = 0.08,
    ):
        q = np.array(q0, float)
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(iters):
            p, R = self.fk(q)
            ex, ez = R[:, 0], R[:, 2]
            res = [pos - p, w_down * np.cross(ex, DOWN)]
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
            Jp, Jr = jp[:, self.dadr], jr[:, self.dadr]
            rows = [Jp, w_down * Jr]
            if close_dir is not None:
                c = np.array([close_dir[0], close_dir[1], 0.0])
                c /= np.linalg.norm(c)
                # the closing axis is sign-symmetric for a parallel grasp: aim at whichever of +-c is nearer
                if ez @ c < 0:
                    c = -c
                res.append(w_close * np.cross(ez, c))
                rows.append(w_close * Jr)
            e = np.concatenate(res)
            if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) < 2e-3:
                break
            J = np.vstack(rows)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(e)), e)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        p, R = self.fk(q)
        return (
            q,
            float(np.linalg.norm(pos - p)),
            float(np.degrees(np.arccos(np.clip(R[:, 0] @ DOWN, -1, 1)))),
        )
