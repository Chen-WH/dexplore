"""Mirror the GRAB object meshes for left-hand robots (RobotConfig.mirror).

convert_grab.mirror_interaction_data reflects the scene with M = diag(-1, 1, 1),
so object poses become (M R M, M t) and the object itself must become M v in
its local frame. This writes the mirrored meshes (x -> -x, faces flipped to
keep outward normals) and URDFs that reference them:

    <mesh_output_dir>/<name>/<name>.obj
    dexplore/data/assets/mjcf/mirror/<name>.urdf  ->  ../objects_mirror/<name>/<name>.obj
    dexplore/data/assets/mjcf/objects_mirror      ->  symlink to <mesh_output_dir>

Usage:
    python data_processing/make_mirrored_objects.py --mesh_output_dir ~/data/GRAB/grab_dir/objects_mirror
"""
import argparse
import os

import trimesh


def main():
    parser = argparse.ArgumentParser(description="Mirror GRAB object meshes and URDFs")
    parser.add_argument("--mjcf_dir", type=str, default="dexplore/data/assets/mjcf")
    parser.add_argument("--mesh_output_dir", type=str, required=True,
                        help="Where to write mirrored meshes (symlinked as <mjcf_dir>/objects_mirror)")
    args = parser.parse_args()

    mesh_out = os.path.abspath(os.path.expanduser(args.mesh_output_dir))
    obj_dir = os.path.join(args.mjcf_dir, "objects")
    urdf_out = os.path.join(args.mjcf_dir, "mirror")
    os.makedirs(urdf_out, exist_ok=True)

    names = sorted(n for n in os.listdir(obj_dir)
                   if os.path.exists(os.path.join(obj_dir, n, f"{n}.obj"))
                   and os.path.exists(os.path.join(args.mjcf_dir, f"{n}.urdf")))
    for name in names:
        mesh = trimesh.load(os.path.join(obj_dir, name, f"{name}.obj"), force="mesh", process=False)
        mesh.vertices[:, 0] *= -1
        mesh.faces = mesh.faces[:, ::-1]
        os.makedirs(os.path.join(mesh_out, name), exist_ok=True)
        mesh.export(os.path.join(mesh_out, name, f"{name}.obj"))

        with open(os.path.join(args.mjcf_dir, f"{name}.urdf")) as f:
            urdf = f.read()
        urdf = urdf.replace(f'filename="objects/{name}/', f'filename="../objects_mirror/{name}/')
        with open(os.path.join(urdf_out, f"{name}.urdf"), "w") as f:
            f.write(urdf)

    link = os.path.join(args.mjcf_dir, "objects_mirror")
    if not os.path.lexists(link):
        os.symlink(mesh_out, link)
    print(f"Mirrored {len(names)} objects -> {mesh_out}, URDFs in {urdf_out}")


if __name__ == "__main__":
    main()
