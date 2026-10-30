"""Build the `sequences/` folder that convert_grab.py expects from raw GRAB data.

convert_grab.py reads `{grab_dir}/sequences/{sub}_{seq}/motion.npz` and `object.npz`.
The public InterAct `process/process_grab.py` only writes `human.npz` / `object.npz`,
so this script reproduces InterAct's GRAB processing (y-up world, 120 -> 30 fps,
height fix) and additionally writes `motion.npz`: a GRAB-format dict whose body
params are in the processed frame at 30 fps, with the raw 120 fps contact labels.

Usage:
    python data_processing/prepare_grab_sequences.py \
        --grab_dir /path/to/grab_dir \
        --smplx_model_dir /path/to/models    # contains smplx/SMPLX_*.npz
"""
import argparse
import copy
import os

import numpy as np
import smplx
import torch
import trimesh
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def DotDict(in_dict):
    out = copy.copy(in_dict)
    for k, v in out.items():
        if isinstance(v, dict):
            out[k] = DotDict(v)
    return dotdict(out)


def parse_npz(path):
    npz = np.load(path, allow_pickle=True)
    return DotDict({k: npz[k].item() for k in npz.files})


def smplx_forward(model, params, device, chunk=256):
    """Run SMPL-X in chunks (model built with batch_size=1).

    Returns the pelvis trajectory and the vertices of the first 30 frames (CPU).
    """
    T = params["transl"].shape[0]
    pelvis, verts30 = [], None
    for s in range(0, T, chunk):
        p = {k: torch.from_numpy(v[s:s + chunk]).float().to(device) for k, v in params.items()}
        with torch.no_grad():
            out = model(**p)
        pelvis.append(out.joints[:, 0].cpu())
        if s == 0:
            verts30 = out.vertices[:30].cpu()
    return torch.cat(pelvis).numpy(), verts30


def process_sequence(raw_file, sub, args, models, vtemps, device):
    seq_name = os.path.splitext(os.path.basename(raw_file))[0]
    save_name = f"{sub}_{seq_name}"
    out_dir = os.path.join(args.grab_dir, "sequences", save_name)
    if not args.overwrite and os.path.exists(os.path.join(out_dir, "motion.npz")):
        return
    seq = parse_npz(raw_file)
    gender = str(seq.gender)
    obj_name = seq_name.split("_")[0]

    key = (gender, str(seq.sbj_id), int(seq.n_comps))
    if key not in models:
        vtemp_path = os.path.join(args.grab_dir, seq.body.vtemp)
        vtemps[key] = np.array(trimesh.load(vtemp_path, process=False).vertices, dtype=np.float32)
        models[key] = smplx.create(
            model_path=args.smplx_model_dir, model_type="smplx", gender=gender,
            num_pca_comps=int(seq.n_comps), v_template=vtemps[key], batch_size=1,
        ).to(device)
    model = models[key]

    body = {k: v.astype(np.float32) for k, v in seq.body.params.items() if k != "fullpose"}
    pelvis, _ = smplx_forward(model, body, device)

    # y-up world: rotate global orient / transl by -90 deg around x (as InterAct does)
    rot_x = sRot.from_euler("x", -np.pi / 2)
    body_rot = dict(body)
    body_rot["global_orient"] = (rot_x * sRot.from_rotvec(body["global_orient"])).as_rotvec().astype(np.float32)
    body_rot["transl"] = rot_x.apply(body["transl"]).astype(np.float32)

    obj_params = seq.object.params
    # GRAB stores object rotations for row-vector vertices; invert to column-vector convention
    obj_angles = (rot_x * sRot.from_rotvec(obj_params["global_orient"]).inv()).as_rotvec()
    obj_trans_delta = rot_x.apply(obj_params["transl"] - pelvis)

    pelvis_rot, verts30 = smplx_forward(model, body_rot, device)
    obj_trans = pelvis_rot + obj_trans_delta
    diff_fix = verts30[:, :, 1].min().item()
    body_rot["transl"][:, 1] -= diff_fix
    obj_trans[:, 1] -= diff_fix

    # 120 -> 30 fps
    body_ds = {k: v[::4] for k, v in body_rot.items()}
    obj_angles, obj_trans = obj_angles[::4], obj_trans[::4]
    B = body_ds["transl"].shape[0]

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "object.npz"), angles=obj_angles, trans=obj_trans, name=obj_name)
    np.savez(os.path.join(out_dir, "human.npz"),
             poses=np.concatenate([body_ds["global_orient"], body_ds["body_pose"],
                                   body_ds["left_hand_pose"], body_ds["right_hand_pose"]], axis=1),
             trans=body_ds["transl"], vtemp=vtemps[key], gender=gender)
    # GRAB-format dict; contact labels stay at 120 fps (convert_grab indexes them with 4*t)
    motion = dict(
        gender=gender, sbj_id=str(seq.sbj_id), obj_name=obj_name,
        motion_intent=str(seq.motion_intent), framerate=30.0,
        n_comps=int(seq.n_comps), n_frames=B,
        body={"params": body_ds, "vtemp": seq.body.vtemp},
        contact={"body": seq.contact["body"], "object": seq.contact["object"]},
    )
    np.savez_compressed(os.path.join(out_dir, "motion.npz"),
                        **{k: np.array(v, dtype=object) for k, v in motion.items()})


def main():
    parser = argparse.ArgumentParser(description="Prepare GRAB sequences for convert_grab.py")
    parser.add_argument("--grab_dir", type=str, required=True,
                        help="Contains raw/s{N}/*.npz and tools/subject_meshes/")
    parser.add_argument("--smplx_model_dir", type=str, required=True,
                        help="Directory containing smplx/SMPLX_{MALE,FEMALE,NEUTRAL}.npz")
    parser.add_argument("--subjects", type=str, nargs="+", default=[f"s{i}" for i in range(1, 11)])
    parser.add_argument("--filter", type=str, default=None,
                        help="Only process sequences matching this substring")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = []
    for sub in args.subjects:
        sub_dir = os.path.join(args.grab_dir, "raw", sub)
        for f in sorted(os.listdir(sub_dir)):
            if f.endswith(".npz") and (args.filter is None or args.filter in f"{sub}_{f}"):
                files.append((os.path.join(sub_dir, f), sub))

    models, vtemps = {}, {}
    for raw_file, sub in tqdm(files, desc="Sequences"):
        process_sequence(raw_file, sub, args, models, vtemps, device)


if __name__ == "__main__":
    main()
