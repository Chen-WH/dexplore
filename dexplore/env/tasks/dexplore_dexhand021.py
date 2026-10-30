"""Dexplore task for the left DexHand021 (trained on mirrored GRAB data)."""
import torch

from env.tasks.base_dexplore_task import DexploreTask


# Isaac Gym DOF indices (6 wrist DOFs, then fingers 1=thumb .. 5=pinky, segments in order)
ACTIVE_FINGER_DOFS = [6, 7, 8, 10, 11, 12, 14, 15, 18, 19, 22, 23]
# Linear URDF mimic couplings: follower = multiplier * leader + offset
DIP_COUPLINGS = [(9, 8), (13, 12), (16, 15), (20, 19), (24, 23)]  # *_4 follows *_3
DIP_MULTIPLIER, DIP_OFFSET = 0.9023, -0.041
SPREAD_COUPLINGS = [(17, 10, 1.0, 0.0), (21, 10, 1.8162, 0.0181)]  # ring / pinky follow index spread


class Dexplore_Dexhand021(DexploreTask):
    # DexHand021: 25 DOFs (6 wrist + 19 finger, 12 of them active)
    DOF_VELOCITY = (7,) * 25
    DOF_STIFFNESS = (200,) * 6 + (100,) * 19
    DOF_DAMPING = (20,) * 6 + (10,) * 19

    def _get_robot_type(self):
        return "dexhand021/dexhand021_left.urdf"

    def _apply_collision_filter(self, env_ptr, humanoid_handle):
        # Same scheme as Inspire: distal finger links never collide with the table
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)
        shape_idx = self.gym.get_actor_rigid_body_shape_indices(env_ptr, humanoid_handle)
        names = self.gym.get_actor_rigid_body_names(env_ptr, humanoid_handle)
        for body, idx in zip(names, shape_idx):
            distal = body.endswith("_4") or body.endswith("_pad")
            for p_idx in range(idx.start, idx.start + idx.count):
                props[p_idx].filter = 3 if distal else 2
        self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, props)

    @staticmethod
    def _apply_couplings(q):
        for follower, leader in DIP_COUPLINGS:
            q[..., follower] = DIP_MULTIPLIER * q[..., leader] + DIP_OFFSET
        for follower, leader, mul, off in SPREAD_COUPLINGS:
            q[..., follower] = mul * q[..., leader] + off

    def _action_to_pd_targets(self, action):
        action[..., 6:] = (1 + action[..., 6:]) / 2
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        pd_tar[..., :6] = pd_tar[..., :6] + self._dof_pos[..., :6]
        self._apply_couplings(pd_tar)
        self.real_pd_tar = pd_tar[..., ACTIVE_FINGER_DOFS]
        return pd_tar

    def _set_env_state(self, env_ids, dof_pos, dof_vel):
        self._humanoid_root_states[:, :] = 0
        self._humanoid_root_states[:, 6:7] = 1

        q = dof_pos.clone()
        lo = self.dof_limits_lower[ACTIVE_FINGER_DOFS]
        hi = self.dof_limits_upper[ACTIVE_FINGER_DOFS]
        q[:, ACTIVE_FINGER_DOFS] = torch.max(torch.min(q[:, ACTIVE_FINGER_DOFS], hi), lo)
        self._apply_couplings(q)
        self._dof_pos[env_ids] = q
        self._dof_vel[env_ids] = dof_vel
