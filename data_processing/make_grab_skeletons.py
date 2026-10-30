"""Write per-subject MuJoCo skeletons `smplx_grab_s{N}.xml` for convert_grab.py.

Reproduces only the skeleton-export part of InterAct's
`simulation/interact2mimic.py --dataset_name grab`, so the full InterAct
pipeline (SMPL-H models, canonicalization, text annotations) is not needed.

Usage:
    python data_processing/make_grab_skeletons.py \
        --interact_dir /path/to/InterAct \
        --grab_dir /path/to/grab_dir \
        --smplx_model_dir /path/to/models    # contains smplx/SMPLX_*.npz
"""
import argparse
import os
import sys

import numpy as np
import torch
import trimesh

GENDER_NUMBER = {"neutral": [0], "male": [1], "female": [2]}


def main():
    parser = argparse.ArgumentParser(description="Generate GRAB SMPL-X MuJoCo skeletons")
    parser.add_argument("--interact_dir", type=str, required=True)
    parser.add_argument("--grab_dir", type=str, required=True)
    parser.add_argument("--smplx_model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="dexplore/data/assets/smplx")
    parser.add_argument("--subjects", type=str, nargs="+", default=[f"s{i}" for i in range(1, 11)])
    args = parser.parse_args()

    for k in ("interact_dir", "grab_dir", "smplx_model_dir", "output_dir"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    # uhc loads its MJCF templates relative to InterAct/simulation
    sim_dir = os.path.join(args.interact_dir, "simulation")
    sys.path.insert(0, sim_dir)
    os.chdir(sim_dir)
    from uhc.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot

    os.makedirs(args.output_dir, exist_ok=True)
    for sub in args.subjects:
        raw_dir = os.path.join(args.grab_dir, "raw", sub)
        seq = np.load(os.path.join(raw_dir, sorted(os.listdir(raw_dir))[0]), allow_pickle=True)
        gender = str(seq["gender"])
        vtemp_rel = seq["body"].item()["vtemp"]
        sbj_vtemp = np.array(trimesh.load(os.path.join(args.grab_dir, vtemp_rel), process=False).vertices)
        beta = np.zeros(10, dtype=np.float32)  # GRAB uses v_template, betas are zero

        robot_cfg = {
            "mesh": False, "model": "smplx", "gender": gender, "upright_start": True,
            "body_params": {}, "joint_params": {}, "geom_params": {}, "actuator_params": {},
            "beta": beta, "flat_hand_mean": True,
        }
        robot = LocalRobot(robot_cfg, data_dir=os.path.join(args.smplx_model_dir, "smplx"), sbj_vtemp=sbj_vtemp)
        robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=GENDER_NUMBER[gender], objs_info=None)
        out = os.path.join(args.output_dir, f"smplx_grab_{sub}.xml")
        robot.write_xml(out)
        print(f"Saved {out} ({gender})")


if __name__ == "__main__":
    main()
