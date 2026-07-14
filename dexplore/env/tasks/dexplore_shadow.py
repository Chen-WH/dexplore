"""Dexplore task for the Shadow dexterous hand."""
from env.tasks.base_dexplore_task import DexploreTask


class Dexplore_Shadow(DexploreTask):
    # Shadow hand: 30 DOFs (6 wrist + 24 finger)
    DOF_VELOCITY = (10,) * 30
    DOF_STIFFNESS = (200,) * 6 + (100,) * 24
    DOF_DAMPING = (20,) * 6 + (10,) * 24
    COLLISION_GROUP_SELF = 2

    def _get_robot_type(self):
        return "shadow_hand/shadow_hand_right.urdf"

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
        # Shadow-specific multi-joint coupling
        pd_tar[..., 8] = pd_tar[..., 7] * 1.01511
        pd_tar[..., 9] = pd_tar[..., 7] * 1.44889
        pd_tar[..., 14] = pd_tar[..., 6] * 1
        pd_tar[..., 13] = pd_tar[..., 12] * 1.0450
        pd_tar[..., 25] = pd_tar[..., 24] * 1.0450
        pd_tar[..., 21] = pd_tar[..., 20] * 1.3588
        pd_tar[..., 22] = pd_tar[..., 20] * 1.42093
        pd_tar[..., 17] = pd_tar[..., 16] * 1.35880
        pd_tar[..., 18] = pd_tar[..., 16] * 1.42307
        pd_tar[..., 10] = pd_tar[..., 15] * 0.5
        pd_tar[..., 19] = pd_tar[..., 15] * 0.5
        self.real_pd_tar = pd_tar
        return pd_tar

    def _set_env_state(self, env_ids, dof_pos, dof_vel):
        self._humanoid_root_states[:, :] = 0
        self._humanoid_root_states[:, 6:7] = 1
        dof_pos[:, 6:] = 0
        dof_pos[:, 8] = dof_pos[:, 7] * 1.01511
        dof_pos[:, 9] = dof_pos[:, 7] * 1.44889
        dof_pos[:, 14] = dof_pos[:, 6] * 1
        dof_pos[:, 13] = dof_pos[:, 12] * 1.0450
        dof_pos[:, 25] = dof_pos[:, 24] * 1.0450
        dof_pos[:, 21] = dof_pos[:, 20] * 1.3588
        dof_pos[:, 22] = dof_pos[:, 20] * 1.42093
        dof_pos[:, 17] = dof_pos[:, 16] * 1.35880
        dof_pos[:, 18] = dof_pos[:, 16] * 1.42307
        dof_pos[:, 10] = dof_pos[:, 15] * 0.5
        dof_pos[:, 19] = dof_pos[:, 15] * 0.5
        self._dof_pos[env_ids] = dof_pos.clone()
        self._dof_vel[env_ids] = (self._dof_pos[env_ids] - self._dof_pos[env_ids]).clone()
