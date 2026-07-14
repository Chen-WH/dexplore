"""Minimal skeleton utilities for GRAB data processing.

Provides SkeletonTree (parsed from MuJoCo XML) and SkeletonState
(forward kinematics from local rotations to global poses).

Adapted from NVIDIA poselib (https://github.com/NVIDIA-Omniverse/IsaacGymEnvs).
Quaternion convention: [x, y, z, w].
"""
import xml.etree.ElementTree as ET

import numpy as np
import torch


# ── Quaternion utilities (xyzw convention) ──────────────────────────────────

def _quat_mul(a, b):
    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 + y1*w2 + z1*x2 - x1*z2,
        w1*z2 + z1*w2 + x1*y2 - y1*x2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dim=-1)


def _quat_conjugate(x):
    return torch.cat([-x[..., :3], x[..., 3:]], dim=-1)


def _quat_normalize(q):
    q = q / q.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
    # ensure positive real part
    q = q * (2.0 * (q[..., 3:] >= 0).float() - 1.0)
    return q


def _quat_rotate(rot, vec):
    other_q = torch.cat([vec, torch.zeros_like(vec[..., :1])], dim=-1)
    return _quat_mul(_quat_mul(rot, other_q), _quat_conjugate(rot))[..., :3]


def _quat_identity(shape):
    w = torch.ones(list(shape) + [1])
    xyz = torch.zeros(list(shape) + [3])
    return _quat_normalize(torch.cat([xyz, w], dim=-1))


# ── SkeletonTree ────────────────────────────────────────────────────────────

class SkeletonTree:
    """Rigid skeleton described as a tree with named nodes and local translations."""

    def __init__(self, node_names, parent_indices, local_translation):
        self.node_names = node_names
        self.parent_indices = parent_indices  # int32 tensor, -1 for root
        self.local_translation = local_translation  # (num_joints, 3) float tensor

    def __len__(self):
        return len(self.node_names)

    @property
    def num_joints(self):
        return len(self.node_names)

    @classmethod
    def from_mjcf(cls, path):
        """Parse a MuJoCo XML file and build the skeleton tree."""
        tree = ET.parse(path)
        root = tree.getroot()
        world_body = root.find("worldbody")
        body_root = world_body.find("body")

        node_names = []
        parent_indices = []
        local_translation = []

        def _add_node(xml_node, parent_index, node_index):
            node_names.append(xml_node.attrib.get("name"))
            pos = np.fromstring(xml_node.attrib.get("pos"), dtype=float, sep=" ")
            parent_indices.append(parent_index)
            local_translation.append(pos)
            curr = node_index
            node_index += 1
            for child in xml_node.findall("body"):
                node_index = _add_node(child, curr, node_index)
            return node_index

        _add_node(body_root, -1, 0)

        return cls(
            node_names,
            torch.from_numpy(np.array(parent_indices, dtype=np.int32)),
            torch.from_numpy(np.array(local_translation, dtype=np.float32)),
        )


# ── SkeletonState ───────────────────────────────────────────────────────────

class SkeletonState:
    """Static pose of a skeleton: local/global rotations + root translation."""

    def __init__(self, tensor, skeleton_tree, is_local):
        self._skeleton_tree = skeleton_tree
        self._is_local = is_local
        self.tensor = tensor.clone()

    @classmethod
    def from_rotation_and_root_translation(cls, skeleton_tree, r, t, is_local=True):
        """Construct state from per-joint rotations (xyzw quats) and root translation."""
        state_shape = r.shape[:-2]
        vr = r.reshape(*(state_shape + (-1,)))
        vt = t.broadcast_to(*state_shape + t.shape[-1:]).reshape(*(state_shape + (-1,)))
        state_vec = torch.cat([vr, vt], dim=-1)
        return cls(state_vec, skeleton_tree=skeleton_tree, is_local=is_local)

    @property
    def num_joints(self):
        return self._skeleton_tree.num_joints

    @property
    def skeleton_tree(self):
        return self._skeleton_tree

    @property
    def rotation(self):
        if not hasattr(self, "_rotation"):
            self._rotation = self.tensor[..., :self.num_joints * 4].reshape(
                *(self.tensor.shape[:-1] + (self.num_joints, 4)))
        return self._rotation

    @property
    def root_translation(self):
        if not hasattr(self, "_root_translation"):
            self._root_translation = self.tensor[..., self.num_joints * 4:self.num_joints * 4 + 3]
        return self._root_translation

    @property
    def local_rotation(self):
        if self._is_local:
            return self.rotation
        # Compute local from global
        if not hasattr(self, "_comp_local_rotation"):
            gr = self.global_rotation
            lr = _quat_identity(gr.shape[:-1]).to(gr)
            for i in range(self.num_joints):
                pi = self._skeleton_tree.parent_indices[i]
                if pi == -1:
                    lr[..., i, :] = gr[..., i, :]
                else:
                    lr[..., i, :] = _quat_normalize(
                        _quat_mul(_quat_conjugate(gr[..., pi, :]), gr[..., i, :]))
            self._comp_local_rotation = lr
        return self._comp_local_rotation

    @property
    def global_rotation(self):
        if not self._is_local:
            return self.rotation
        if not hasattr(self, "_comp_global_rotation"):
            self._comp_global_rotation = self._forward_kinematics()[0]
        return self._comp_global_rotation

    @property
    def global_translation(self):
        if not hasattr(self, "_global_translation"):
            self._global_translation = self._forward_kinematics()[1]
        return self._global_translation

    def _forward_kinematics(self):
        """Compute global rotations and translations via forward kinematics."""
        if hasattr(self, "_fk_cache"):
            return self._fk_cache

        lr = self.local_rotation if self._is_local else self.rotation
        lt = self._skeleton_tree.local_translation.broadcast_to(
            *(self.tensor.shape[:-1] + (self.num_joints,) + (3,))).clone()
        lt[..., 0, :] = self.root_translation

        parent_indices = self._skeleton_tree.parent_indices.numpy()
        global_rot = []
        global_trans = []

        for i in range(self.num_joints):
            pi = parent_indices[i]
            if pi == -1:
                global_rot.append(lr[..., i, :])
                global_trans.append(lt[..., i, :])
            else:
                gr = _quat_normalize(_quat_mul(global_rot[pi], lr[..., i, :]))
                gt = _quat_rotate(global_rot[pi], lt[..., i, :]) + global_trans[pi]
                global_rot.append(gr)
                global_trans.append(gt)

        gr_tensor = torch.stack(global_rot, dim=-2)
        gt_tensor = torch.stack(global_trans, dim=-2)
        self._fk_cache = (gr_tensor, gt_tensor)
        return self._fk_cache
