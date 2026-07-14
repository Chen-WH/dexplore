"""Dexplore task for the Inspire dexterous hand."""
import math
import torch

from env.tasks.base_dexplore_task import DexploreTask

from isaacgym import gymapi


class Dexplore_Inspire(DexploreTask):
    # Inspire hand: 18 DOFs (6 wrist + 12 finger)
    DOF_VELOCITY = (7,) * 18
    DOF_STIFFNESS = (200,) * 6 + (100,) * 12
    DOF_DAMPING = (20,) * 6 + (10,) * 12
    COLLISION_GROUP_SELF = 1

    def _get_robot_type(self):
        return f"{self.robot_name}_hand_new/{self.robot_name}_hand_right.urdf"

    def _apply_collision_filter(self, env_ptr, humanoid_handle):
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)
        names = self.gym.get_actor_rigid_body_names(env_ptr, humanoid_handle)
        for p_idx in range(len(props)):
            if 'thumb' in names[p_idx] and 'distal' in names[p_idx]:
                props[p_idx].filter = 3
            elif 'thumb' not in names[p_idx] and 'intermediate' in names[p_idx]:
                props[p_idx].filter = 3
            else:
                props[p_idx].filter = 2
        self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, props)

    def _action_to_pd_targets(self, action):
        action[..., 6:] = (1 + action[..., 6:]) / 2
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        pd_tar[..., :6] = pd_tar[..., :6] + self._dof_pos[..., :6]
        # Inspire-specific joint coupling
        pd_tar[..., 7] = pd_tar[..., 6] * 1.05
        pd_tar[..., 9] = pd_tar[..., 8] * 1.05
        pd_tar[..., 11] = pd_tar[..., 10] * 1.05
        pd_tar[..., 13] = pd_tar[..., 12] * 1.05
        pd_tar[..., 16] = pd_tar[..., 15] * 0.6
        pd_tar[..., 17] = pd_tar[..., 15] * 0.8
        self.real_pd_tar = pd_tar[..., [14, 15, 6, 8, 12, 10]]
        return pd_tar

    def _set_env_state(self, env_ids, dof_pos, dof_vel):
        self._humanoid_root_states[:, :] = 0
        self._humanoid_root_states[:, 6:7] = 1

        limits = {
            14: (0.0, 1.15),
            15: (0.0, 0.55),
            6: (0.0, 1.6),
            8: (0.0, 1.6),
            10: (0.0, 1.6),
            12: (0.0, 1.6),
        }
        self._dof_pos[env_ids] = dof_pos
        for idx, (lo, hi) in limits.items():
            self._dof_pos[env_ids, idx] = self._dof_pos[env_ids, idx].clamp(min=lo, max=hi)

        # Inspire-specific joint coupling at reset
        self._dof_pos[env_ids, 7] = self._dof_pos[env_ids, 6] * 1.05
        self._dof_pos[env_ids, 9] = self._dof_pos[env_ids, 8] * 1.05
        self._dof_pos[env_ids, 11] = self._dof_pos[env_ids, 10] * 1.05
        self._dof_pos[env_ids, 13] = self._dof_pos[env_ids, 12] * 1.05
        self._dof_pos[env_ids, 16] = self._dof_pos[env_ids, 15] * 0.6
        self._dof_pos[env_ids, 17] = self._dof_pos[env_ids, 15] * 0.8
        self._dof_vel[env_ids] = dof_vel
