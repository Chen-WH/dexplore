"""Check which retargeted sequences a UR10-mounted hand can actually follow.

Each sequence is translated so that its table top center lands on the real
table center (calibrated, UR "base" frame); the hand-base trajectory from the
retargeted floating wrist is then tracked with damped-least-squares IK on the
UR10 + hand URDF. Only the task window is checked: from the first frame the
hand is within 0.2 m of the object (episodes start before that, see
DexploreTask._reset_ref_state_init) to 0.5 s after the last object contact.

A frame fails if IK misses the pose (>5 mm / >3 deg) or an arm link above the
table footprint dips below the table top plus a margin. Joint speed is reported
separately as the slowdown needed to stay within the URDF velocity limits,
since references can be replayed slower (e.g. dataFramesScale).

Usage:
    python data_processing/check_ur10_reachability.py \
        --robot_assets ~/ros2_ws/src/robot-assets --output ur10_reachability.json
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
import torch
import trimesh
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_configs import ROBOT_CONFIGS

UR10_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
ARM_LINKS = ["forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link", "mount_link"]
FPS = 30.0
POS_TOL, ROT_TOL = 0.005, np.deg2rad(3.0)


class UR10IK:
    def __init__(self, urdf, ee_frame):
        full = pin.buildModelFromUrdf(urdf)
        lock = [full.getJointId(n) for n in full.names[1:] if n not in UR10_JOINTS]
        self.model = pin.buildReducedModel(full, lock, pin.neutral(full))
        self.data = self.model.createData()
        self.ee = self.model.getFrameId(ee_frame)
        self.arm_frames = [self.model.getFrameId(n) for n in ARM_LINKS]
        # Calibrations are expressed in the UR controller base, which is the URDF "base" link
        self.T_world_base = self.frame_pose(pin.neutral(self.model), self.model.getFrameId("base"))
        self.lower, self.upper = self.model.lowerPositionLimit, self.model.upperPositionLimit
        self.vmax = self.model.velocityLimit

    def frame_pose(self, q, fid):
        pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[fid].copy()

    def solve(self, target_base, q0, iters=100, damping=1e-3):
        target = self.T_world_base * target_base
        q = q0.copy()
        for _ in range(iters):
            cur = self.frame_pose(q, self.ee)
            err = pin.log6(cur.actInv(target)).vector
            if np.linalg.norm(err[:3]) < POS_TOL * 0.2 and np.linalg.norm(err[3:]) < ROT_TOL * 0.2:
                break
            J = pin.computeFrameJacobian(self.model, self.data, q, self.ee, pin.LOCAL)
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
            q = np.clip(pin.integrate(self.model, q, dq), self.lower, self.upper)
        cur = self.frame_pose(q, self.ee)
        err = pin.log6(cur.actInv(target)).vector
        return q, np.linalg.norm(err[:3]), np.linalg.norm(err[3:])

    def arm_points_base(self, q):
        pin.framesForwardKinematics(self.model, self.data, q)
        inv = self.T_world_base.inverse()
        return np.stack([(inv * self.data.oMf[f]).translation for f in self.arm_frames])


def table_top_center(table_mesh, d):
    v = sRot.from_quat(d[0, 241:245]).apply(table_mesh.vertices) + d[0, 238:241]
    lo, hi = v.min(0), v.max(0)
    return np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]])


def task_window(d, hand_pts, obj_pts_local, n_frames):
    obj = np.einsum("tij,pj->tpi", sRot.from_quat(d[:, 201:205]).as_matrix(), obj_pts_local) + d[:, None, 198:201]
    dist = np.linalg.norm(hand_pts[:, :, None] - obj[:, None], axis=-1).min(axis=(1, 2))
    near = np.where(dist < 0.2)[0]
    contact = np.where(d[:, 205] > 0.5)[0]
    if len(near) == 0 or len(contact) == 0:
        return None
    return int(near[0]), int(min(n_frames - 1, contact[-1] + int(0.5 * FPS)))


def main():
    parser = argparse.ArgumentParser(description="UR10 reachability of retargeted sequences")
    parser.add_argument("--robot", type=str, default="dexhand021", choices=list(ROBOT_CONFIGS.keys()))
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--assets_dir", type=str, default="dexplore/data/assets")
    parser.add_argument("--robot_assets", type=str, required=True, help="robot-assets repository")
    parser.add_argument("--scene_urdf", type=str, default="scene/urdf/ur10_dexhand.urdf")
    parser.add_argument("--table_calib", type=str, default="scene/calibration/table_boundary_20261012.json")
    parser.add_argument("--ee_frame", type=str, default="left_hand_base")
    parser.add_argument("--table_offset", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="Extra shift of the real table center (base frame), to evaluate moving the table")
    parser.add_argument("--table_margin", type=float, default=0.05,
                        help="Minimum clearance of arm links above the table top")
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--output", type=str, default="ur10_reachability.json")
    args = parser.parse_args()

    robot_cfg = ROBOT_CONFIGS[args.robot]
    from dex_retargeting.retargeting_config import RetargetingConfig
    RetargetingConfig.set_default_urdf_dir(Path(args.assets_dir))
    hand = RetargetingConfig.load_from_file(
        Path(__file__).parent / robot_cfg.retarget_config).build().optimizer.robot
    hand_base = hand.get_link_index(args.ee_frame)
    pad_ids = [hand.get_link_index(n) for n in hand.link_names if n.endswith("_pad")]

    ra = os.path.expanduser(args.robot_assets)
    ik = UR10IK(os.path.join(ra, args.scene_urdf), args.ee_frame)
    calib = json.load(open(os.path.join(ra, args.table_calib)))
    table_center = np.array(calib["center_base"]) + np.array(args.table_offset)
    footprint = np.array(calib["collision_footprint_base_xy"]) + np.array(args.table_offset[:2])
    fp_lo, fp_hi = footprint.min(0), footprint.max(0)

    mjcf = os.path.join(args.assets_dir, "mjcf")
    table_mesh = trimesh.load(os.path.join(mjcf, "objects/table/table.obj"), force="mesh", process=False)
    obj_dir = os.path.join(mjcf, "objects_mirror" if robot_cfg.mirror else "objects")
    obj_cache = {}

    files = sorted(glob.glob(os.path.join(args.data_dir, "*", f"interaction_hand_{robot_cfg.robot_name}.pt")))
    if args.filter:
        files = [f for f in files if args.filter in f]
    # IK seeds: arm reaching towards the table (-y in the base frame), elbow up / down
    seeds = [np.array([-np.pi / 2, -1.2, s * 1.8, -1.5, -np.pi / 2, 0.0]) for s in (1, -1)] + \
            [np.array([np.pi / 2, -2.0, s * 1.8, -1.5, np.pi / 2, 0.0]) for s in (1, -1)]
    report = {}
    for f in tqdm(files, desc="Reachability"):
        name = os.path.basename(os.path.dirname(f))
        if "doorknob" in name:  # not used by the task
            continue
        d = torch.load(f).detach().double().numpy()
        if (np.round(d[:, 206:222]) > 0).any():  # left-hand contact: filtered out by the task
            continue
        obj = name.split("_")[1]
        if obj not in obj_cache:
            m = trimesh.load(os.path.join(obj_dir, obj, f"{obj}.obj"), force="mesh", process=False)
            obj_cache[obj] = trimesh.sample.sample_surface_even(m, 256, seed=2024)[0]
        T = d.shape[0]
        qpos = d[:, 245 + 32 * 4:245 + 32 * 4 + robot_cfg.num_dof]
        poses, pads = [], []
        for t in range(T):
            hand.compute_forward_kinematics(qpos[t])
            poses.append(hand.get_link_pose(hand_base))
            pads.append([hand.get_link_pose(i)[:3, 3] for i in pad_ids])
        window = task_window(d, np.array(pads), obj_cache[obj], T)
        if window is None:
            continue
        shift = table_center - table_top_center(table_mesh, d)

        q, ok = None, []
        pos_errs, rot_errs, slowdown, ik_fail = [], [], [], 0
        clash_links = np.zeros(len(ARM_LINKS), dtype=int)
        for t in range(window[0], window[1] + 1):
            P = poses[t].copy()
            P[:3, 3] += shift
            target = pin.SE3(P[:3, :3], P[:3, 3])
            if q is None:
                sols = [ik.solve(target, s, iters=300) for s in seeds]
                valid = [s for s in sols if s[1] < POS_TOL and s[2] < ROT_TOL]
                if valid:  # prefer the branch that keeps the arm highest above the table
                    q, pe, re = max(valid, key=lambda s: ik.arm_points_base(s[0])[:, 2].min())
                else:
                    q, pe, re = min(sols, key=lambda s: s[1] + 0.1 * s[2])
                step = np.zeros(6)
            else:
                q_new, pe, re = ik.solve(target, q)
                step = np.abs(q_new - q)
                q = q_new
            arm = ik.arm_points_base(q)
            over = (arm[:, 0] > fp_lo[0]) & (arm[:, 0] < fp_hi[0]) & (arm[:, 1] > fp_lo[1]) & (arm[:, 1] < fp_hi[1])
            clash = over & (arm[:, 2] < table_center[2] + args.table_margin)
            ok.append(pe < POS_TOL and re < ROT_TOL and not clash.any())
            pos_errs.append(pe); rot_errs.append(re)
            slowdown.append(float((step / (ik.vmax / FPS)).max()))
            ik_fail += not (pe < POS_TOL and re < ROT_TOL)
            clash_links += clash
        report[name] = dict(
            window=list(window), ok_frac=float(np.mean(ok)), reachable=bool(np.all(ok)),
            max_pos_err=float(max(pos_errs)), max_rot_err_deg=float(np.rad2deg(max(rot_errs))),
            ik_fail_frames=int(ik_fail),
            table_clash_frames={n: int(c) for n, c in zip(ARM_LINKS, clash_links) if c},
            # 95th percentile, so that a single IK branch switch does not dominate
            slowdown=float(np.percentile(slowdown, 95)),
            shift=shift.tolist(),
        )

    n = len(report)
    n_ok = sum(r["reachable"] for r in report.values())
    summary = dict(sequences=n, fully_reachable=n_ok,
                   mean_ok_frac=float(np.mean([r["ok_frac"] for r in report.values()])) if n else 0.0,
                   median_slowdown=float(np.median([r["slowdown"] for r in report.values()])) if n else 0.0,
                   table_offset=args.table_offset)
    with open(args.output, "w") as fp:
        json.dump(dict(summary=summary, sequences=report), fp, indent=1)
    print(f"{n_ok}/{n} sequences fully reachable, mean reachable frame fraction "
          f"{summary['mean_ok_frac']:.2f} -> {args.output}", flush=True)
    # pinocchio and torch both loaded crash in their destructors at interpreter exit
    os._exit(0)


if __name__ == "__main__":
    main()
