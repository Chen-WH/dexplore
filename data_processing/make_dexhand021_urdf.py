"""Build the Isaac Gym URDF for DexHand021 from the robot-assets description.

Prepends the same 6-DOF floating-wrist chain used by the Inspire sim URDF
(3 prismatic xyz + 3 revolute xyz, link0..link6) and attaches the hand base
through `base_joint`. Mimic tags are dropped because Isaac Gym ignores them;
couplings are applied in the task's `_action_to_pd_targets` instead.

Also writes `dexhand021_<side>_retarget.urdf` (hand only, mimic kept) for
dex-retargeting, so retargeting and simulation share the same kinematics and
the same hand base frame.

Meshes are not copied: `<output_dir>/meshes` is a symlink to robot-assets.

Usage:
    python data_processing/make_dexhand021_urdf.py \
        --robot_assets ~/ros2_ws/src/robot-assets --side left
"""
import argparse
import copy
import os
import xml.etree.ElementTree as ET

WRIST_JOINTS = [
    # name, type, axis, lower, upper, velocity
    ("joint1", "prismatic", "1 0 0", -5.0, 5.0, 2),
    ("joint2", "prismatic", "0 1 0", -5.0, 5.0, 2),
    ("joint3", "prismatic", "0 0 1", 0.0, 3.0, 2),
    ("joint4", "continuous", "1 0 0", None, None, 3.14),
    ("joint5", "continuous", "0 1 0", None, None, 3.14),
    ("joint6", "continuous", "0 0 1", None, None, 3.14),
]


def massless_link(name):
    link = ET.Element("link", name=name)
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value="0")
    ET.SubElement(inertial, "inertia", ixx="0", ixy="0", ixz="0", iyy="0", iyz="0", izz="0")
    return link


def main():
    parser = argparse.ArgumentParser(description="Generate DexHand021 URDF for Isaac Gym")
    parser.add_argument("--robot_assets", type=str, required=True,
                        help="Path to the robot-assets repository")
    parser.add_argument("--side", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--variant", type=str, default="convex_mimic",
                        help="robot-assets URDF variant (convex_mimic, primitive_mimic, ...)")
    parser.add_argument("--base_rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="Rotation from the wrist frame (link6) to the hand base")
    parser.add_argument("--output_dir", type=str, default="dexplore/data/assets/dexhand021")
    args = parser.parse_args()

    src_dir = os.path.join(os.path.expanduser(args.robot_assets), "dexrobot_021")
    src = os.path.join(src_dir, "urdf", f"dexhand021_{args.side}_{args.variant}.urdf")
    root = ET.parse(src).getroot()

    for el in root.findall("mujoco"):
        root.remove(el)
    for mesh in root.iter("mesh"):
        mesh.set("filename", mesh.get("filename").replace("../meshes/", "meshes/"))
    retarget_root = copy.deepcopy(root)

    n_mimic = 0
    for joint in root.findall("joint"):
        for mimic in joint.findall("mimic"):
            joint.remove(mimic)
            n_mimic += 1

    hand_base = f"{args.side}_hand_base"
    assert root.find(f"link[@name='{hand_base}']") is not None, f"{hand_base} not found in {src}"

    wrist = [massless_link(f"link{i}") for i in range(7)]
    for i, (name, jtype, axis, lower, upper, vel) in enumerate(WRIST_JOINTS):
        joint = ET.Element("joint", name=name, type=jtype)
        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(joint, "parent", link=f"link{i}")
        ET.SubElement(joint, "child", link=f"link{i + 1}")
        ET.SubElement(joint, "axis", xyz=axis)
        limit = {"effort": "1000", "velocity": str(vel)}
        if lower is not None:
            limit.update(lower=f"{lower:.3f}", upper=f"{upper:.3f}")
        ET.SubElement(joint, "limit", **limit)
        wrist.append(joint)
    base_joint = ET.Element("joint", name="base_joint", type="fixed")
    ET.SubElement(base_joint, "parent", link="link6")
    ET.SubElement(base_joint, "child", link=hand_base)
    ET.SubElement(base_joint, "origin", xyz="0 0 0", rpy=" ".join(f"{v:.6f}" for v in args.base_rpy))
    wrist.append(base_joint)
    for i, el in enumerate(wrist):
        root.insert(i, el)
    root.set("name", f"dexhand021_{args.side}_sim")

    os.makedirs(args.output_dir, exist_ok=True)
    mesh_link = os.path.join(args.output_dir, "meshes")
    if not os.path.lexists(mesh_link):
        os.symlink(os.path.join(src_dir, "meshes"), mesh_link)
    out = os.path.join(args.output_dir, f"dexhand021_{args.side}.urdf")
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
    print(f"Saved {out} (from {os.path.basename(src)}, dropped {n_mimic} mimic tags)")
    out = os.path.join(args.output_dir, f"dexhand021_{args.side}_retarget.urdf")
    ET.ElementTree(retarget_root).write(out, encoding="utf-8", xml_declaration=True)
    print(f"Saved {out} (mimic kept)")


if __name__ == "__main__":
    main()
