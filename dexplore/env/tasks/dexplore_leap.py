"""Dexplore task for the LEAP dexterous hand."""
from env.tasks.base_dexplore_task import DexploreTask


class Dexplore_Leap(DexploreTask):
    # LEAP hand: 22 DOFs (6 wrist + 16 finger)
    DOF_VELOCITY = (10,) * 22
    DOF_STIFFNESS = (200,) * 6 + (100,) * 16
    DOF_DAMPING = (20,) * 6 + (10,) * 16
    COLLISION_GROUP_SELF = 2

    def _get_robot_type(self):
        return "leap_hand/leap_hand_right.urdf"

    def _apply_collision_filter(self, env_ptr, humanoid_handle):
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)
        names = self.gym.get_actor_rigid_body_names(env_ptr, humanoid_handle)
        body_shape_idxs = self.gym.get_actor_rigid_body_shape_indices(env_ptr, humanoid_handle)
        for b_idx, name in enumerate(names):
            f = 1 if 'fingertip' in name else 1
            idx_range = body_shape_idxs[b_idx]
            for s in range(idx_range.start, idx_range.start + idx_range.count):
                props[s].filter = f
        self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, props)

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        pd_tar[..., :6] = pd_tar[..., :6] + self._dof_pos[..., :6]
        self.real_pd_tar = pd_tar
        return pd_tar

    def _set_env_state(self, env_ids, dof_pos, dof_vel):
        self._humanoid_root_states[:, :] = 0
        self._humanoid_root_states[:, 6:7] = 1
        self._dof_pos[env_ids] = dof_pos
        self._dof_pos[env_ids][..., 6:] = 0
        self._dof_vel[env_ids] = self._dof_pos[env_ids] - self._dof_pos[env_ids]
