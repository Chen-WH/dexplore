"""Base task classes for Dexplore dexterous manipulation.

InterMimic: base class for motion-imitation tasks with Isaac Gym.
DexploreTask: extends InterMimic with motion loading, object handling,
              reward computation, and reference-scoped exploration.
"""
from enum import Enum
import numpy as np
import torch
import glob, os, random

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import torch.utils

from utils import torch_utils
import torch.nn.functional as F
from env.tasks.base_task import BaseTask
import trimesh
import math

PERTURB_OBJS = [
    ["small", 60],
]


class InterMimic(BaseTask):
    # Subclasses should override these for robot-specific DOF properties.
    DOF_VELOCITY = None       # tuple of per-DOF velocity limits
    DOF_STIFFNESS = None      # tuple of per-DOF stiffness values
    DOF_DAMPING = None        # tuple of per-DOF damping values
    COLLISION_GROUP_SELF = 1  # collision group for self-collision (1 for inspire, 2 for others)

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        # Control and physics parameters
        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # Observation and termination settings
        self.ref_contact_obs_size = len(self.cfg["env"]["contactBodies"]) * 3
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._root_height_obs = self.cfg["env"].get("rootHeightObs", True)
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        key_bodies = self.cfg["env"]["keyBodies"]
        self._setup_character_props(key_bodies)
        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        enable_camera_sensors = self.cfg["env"].get("enableCameraSensors", False)
        super().__init__(cfg=self.cfg, enable_camera_sensors=enable_camera_sensors)
        self.ref_hoi_obs_size = 392 + 2 * self.num_dof
        self.dt = self.control_freq_inv * sim_params.dt

        # Acquire GPU state tensors from Isaac Gym
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Root state tensors (position, rotation, velocity for each actor)
        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()

        self._humanoid_root_states = self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., 0, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:] = 0
        self._initial_humanoid_root_states[:, 6:7] = 1  # identity quaternion w=1

        self._humanoid_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

        # DOF state tensors (joint positions and velocities)
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 1]

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)

        # Rigid body state tensors (pos, rot, vel, ang_vel per body)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        self._rigid_body_pos = rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[..., :self.num_bodies, 10:13]

        # Contact force tensors
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., :self.num_bodies, :]

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self._build_termination_heights()

        # Build body ID tensors for key bodies (fingertips) and contact bodies
        contact_bodies = self.cfg["env"]["contactBodies"]
        self._key_body_ids = self._build_key_body_ids_tensor(key_bodies)
        self._contact_body_ids = self._build_contact_body_ids_tensor(contact_bodies)

        # Adaptive early termination: kappa=1.0 means use fixed thresholds (default)
        self._kappa = torch.ones(self.num_envs, device=self.device)

        if self.viewer is not None:
            self._init_camera()

        self.robot_index = self._key_body_ids

    def get_obs_size(self):
        return self._num_obs

    def get_action_size(self):
        return self._num_actions

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self._reset_envs(env_ids)

    def set_char_color(self, col, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.humanoid_handles[env_id]
            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))

    def _reset_envs(self, env_ids):
        if len(env_ids) > 0:
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)

    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _setup_character_props(self, key_bodies):
        self._dof_obs_size = self.cfg["env"]["numDof"]
        self._num_actions = self.cfg["env"]["numDof"]
        self._num_obs = self.cfg["env"]["numObs"]

    def _build_termination_heights(self):
        self._termination_heights = to_torch(0.3, device=self.device)

    def get_num_amp_obs(self):
        return self.ref_hoi_obs_size

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # Load robot URDF asset
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_path = os.path.join(asset_root, self.robot_type)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.fix_base_link = True
        asset_options.disable_gravity = gymapi.RIGID_BODY_DISABLE_GRAVITY

        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.num_humanoid_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_humanoid_shapes = self.gym.get_asset_rigid_shape_count(humanoid_asset)
        self.torso_index = 0
        self.num_bodies = self.num_humanoid_bodies
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        # Create all environments with aggregation for GPU pipeline
        max_agg_bodies = self.num_humanoid_bodies + 2  # robot + table + object
        max_agg_shapes = self.num_humanoid_shapes + 22

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            self._build_env(i, env_ptr, humanoid_asset)
            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)

        # Extract DOF limits
        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        if self._pd_control:
            self._build_pd_action_offset_scale()

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        segmentation_id = 1

        start_pose = gymapi.Transform()
        char_h = 0.89
        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, self.COLLISION_GROUP_SELF, segmentation_id)
        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        # Configure PD control with robot-specific DOF properties
        if self._pd_control:
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = (gymapi.DOF_MODE_POS,) * self.num_dof
            if self.DOF_VELOCITY is not None:
                dof_prop["velocity"] = self.DOF_VELOCITY
            if self.DOF_STIFFNESS is not None:
                dof_prop["stiffness"] = self.DOF_STIFFNESS
            if self.DOF_DAMPING is not None:
                dof_prop["damping"] = self.DOF_DAMPING
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        self._apply_collision_filter(env_ptr, humanoid_handle)
        self.humanoid_handles.append(humanoid_handle)

    def _apply_collision_filter(self, env_ptr, humanoid_handle):
        """Override in subclass for robot-specific collision filter logic."""
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)
        names = self.gym.get_actor_rigid_body_names(env_ptr, humanoid_handle)
        for p_idx in range(len(props)):
            props[p_idx].filter = 2
        self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, props)

    def _build_pd_action_offset_scale(self):
        """Build offset and scale for mapping actions to PD targets.
        Wrist DOFs (0-2) use unit scale, (3-5) use pi scale for rotations.
        Finger DOFs use the full joint range.
        """
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        self._pd_action_offset = lim_low - lim_low  # zeros
        self._pd_action_scale = lim_high - lim_low
        self._pd_action_scale[[0, 1, 2]] = 1      # wrist position: unit scale
        self._pd_action_scale[[3, 4, 5]] = np.pi   # wrist rotation: radian scale
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

    def _get_humanoid_collision_filter(self):
        return 1


    def _compute_reward(self, actions):
        # Compute acceleration terms for energy penalty (finite differences)
        hist_pos_vel = self._hist_obs[:,55:58]
        local_vel = (self._curr_obs[:,55:58] - hist_pos_vel)*self.fps_data
        pos_diffacc = (local_vel.view(-1, 1*3)*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)).clone()

        hist_dof_vel = self._hist_obs[:,392+self.num_dof:392+2*self.num_dof]
        local_vel = (self._curr_obs[:,392+self.num_dof:392+2*self.num_dof] - hist_dof_vel)*self.fps_data
        dof_diffacc = (local_vel.view(-1, self.num_dof)*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)).clone()

        hist_obj_vel = self._hist_obs[:,113:116]
        obj_diffacc = (self._curr_obs[:,113:116] - hist_obj_vel)*self.fps_data
        obj_diffacc = obj_diffacc*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)

        hist_obj_rot_vel = self._hist_obs[:,116:119]
        local_vel = (self._curr_obs[:,116:119] - hist_obj_rot_vel)*self.fps_data
        obj_rot_diffacc = local_vel.view(-1, 3)*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)

        env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self.rew_buf[:], ig_reset, contact_reset, kinematic_reset, metric_1, metric_2 = compute_humanoid_reward(
                                                  self._curr_ref_obs,
                                                  self._curr_obs,
                                                  self._contact_forces[:, self._contact_body_ids, :],
                                                  self._tar_contact_forces,
                                                  len(self._key_body_ids),
                                                  self.reward_weights,
                                                  pos_diffacc,
                                                  dof_diffacc,
                                                  self.object_points[self.object_id[self.data_id]],
                                                  obj_diffacc,
                                                  obj_rot_diffacc,
                                                  self.num_dof,
                                                  self.ball_size,
                                                  self.progress_buf-self.start_times>3,
                                                  self._kappa,
                                                  )
        self.contact_reset = (self.contact_reset + contact_reset) * contact_reset
        self._reset_ig = torch.logical_or(ig_reset, kinematic_reset)
        self.metric_1 = metric_1
        self.metric_2 = metric_2

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                    self.max_episode_length[self.data_id],
                                                   self._enable_early_termination, self.start_times, self.rollout_length, self._reset_ig, torch.any(self.contact_reset > 10, dim=-1)
                                                   )

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def _compute_task_obs(self, env_ids=None, ref_obs=None):
        """Compute object-relative observations (position, rotation, velocity diffs)."""
        if env_ids is None:
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            tar_states = self._target_states
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            tar_states = self._target_states[env_ids]

        obs, obj_points = compute_obj_observations(body_pos, body_rot, tar_states, self.object_points[self.object_id[self.data_id[env_ids]]] * self.ball_size, ref_obs)
        self.curr_obj_points = obj_points
        return obs, obj_points

    def _compute_observations_iter(self, env_ids=None, delta_t=1):
        """Compute observations at a future time offset delta_t.
        Returns body state + task obs + interaction graph (IG) features.
        """
        if env_ids is None:
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
            ts = self.progress_buf.clone()
            self._curr_ref_obs = self.hoi_data[self.data_id[env_ids], ts].clone()
            next_ts = torch.clamp(ts + delta_t, max=self.max_episode_length[self.data_id[env_ids]]-1)
            ref_obs = self.hoi_data[self.data_id[env_ids], next_ts].clone()
            obs = self._compute_humanoid_obs(env_ids, ref_obs, next_ts)
            task_obs, obj_points = self._compute_task_obs(env_ids, ref_obs)
            obs = torch.cat([obs, task_obs], dim=-1)
            key_body_pose = self._rigid_body_pos[:, self._key_body_ids[[0, 3, 6, 9, 12, 15]], :]
            ig = compute_sdf(key_body_pose, obj_points).view(-1, 3)
            heading_rot = torch_utils.calc_heading_quat_inv(self._rigid_body_rot[:, 7, :])
            heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, key_body_pose.shape[1], 1).view(-1, 4)
            ig = quat_rotate(heading_rot_extend, ig).view(env_ids.shape[0], -1, 3)
            ig = ig.view(env_ids.shape[0], -1)
            len_key_body_ids = len(self._key_body_ids)
            ref_ig = ref_obs[:, 119+len_key_body_ids*3+1+16:119+len_key_body_ids*3+1+16+len_key_body_ids*3].view(env_ids.shape[0], len_key_body_ids, 3)[:, [0, 3, 6, 9, 12, 15], :]
            ref_ig = ref_ig.view(env_ids.shape[0], -1)
            return torch.cat((obs,ig,ref_ig-ig),dim=-1)

        else:
            ts = self.progress_buf[env_ids].clone()
            ts = torch.clamp(ts, max=self.max_episode_length[self.data_id[env_ids]]-1)
            self._curr_ref_obs[env_ids] = self.hoi_data[self.data_id[env_ids], ts].clone()
            next_ts = torch.clamp(ts + delta_t, max=self.max_episode_length[self.data_id[env_ids]]-1)
            ref_obs = self.hoi_data[self.data_id[env_ids], next_ts].clone()
            obs = self._compute_humanoid_obs(env_ids, ref_obs, next_ts)
            task_obs, obj_points = self._compute_task_obs(env_ids, ref_obs)
            obs = torch.cat([obs, task_obs], dim=-1)
            key_body_pose = self._rigid_body_pos[env_ids][:, self._key_body_ids[[0, 3, 6, 9, 12, 15]], :]
            ig = compute_sdf(key_body_pose, obj_points).view(-1, 3)
            heading_rot = torch_utils.calc_heading_quat_inv(self._rigid_body_rot[env_ids][:, 7, :])
            heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, key_body_pose.shape[1], 1).view(-1, 4)
            ig = quat_rotate(heading_rot_extend, ig).view(env_ids.shape[0], -1, 3)
            ig = ig.view(env_ids.shape[0], -1)
            len_key_body_ids = len(self._key_body_ids)
            ref_ig = ref_obs[:, 119+len_key_body_ids*3+1+16:119+len_key_body_ids*3+1+16+len_key_body_ids*3].view(env_ids.shape[0], len_key_body_ids, 3)[:, [0, 3, 6, 9, 12, 15], :]
            ref_ig = ref_ig.view(env_ids.shape[0], -1)
            return torch.cat((obs,ig,ref_ig-ig),dim=-1)

    def _compute_observations(self, env_ids=None):
        """Concatenate short-horizon (delta_t=1) and long-horizon (delta_t=16) observations."""
        if env_ids is None:
            self.obs_buf[:] = torch.cat((self._compute_observations_iter(None, 1), self._compute_observations_iter(None, 16)), dim=-1)
        else:
            self.obs_buf[env_ids] = torch.cat((self._compute_observations_iter(env_ids, 1), self._compute_observations_iter(env_ids, 16)), dim=-1)

    def _compute_humanoid_obs(self, env_ids=None, ref_obs=None, next_ts=None):
        if env_ids is None:
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            contact_forces = self._contact_forces
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            contact_forces = self._contact_forces[env_ids]
        obs = compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs,
                                                self._root_height_obs,
                                                contact_forces, self._contact_body_ids, ref_obs, self._key_body_ids)

        return obs

    def _reset_actors(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        if self._pd_control:
            pd_tar = self._action_to_pd_targets(self.actions)
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pd_tar))
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(forces))

    def post_physics_step(self):
        self.progress_buf += 1
        self._refresh_sim_tensors()
        env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        # Compute HOI observations first (needed for correct imitation reward)
        self._compute_hoi_observations(env_ids)
        self._compute_observations(env_ids)
        self._compute_reward(self.actions)
        self._compute_reset()
        self.extras["terminate"] = self._terminate_buf

        if self.viewer and self.debug_viz:
            self._update_debug_viz()

    def render(self, sync_frame_time=False):
        if self.viewer:
            self._update_camera()
        super().render(sync_frame_time)

    def _build_key_body_ids_tensor(self, key_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []
        for body_name in key_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert body_id != -1, f"Body '{body_name}' not found in actor"
            body_ids.append(body_id)
        return to_torch(body_ids, device=self.device, dtype=torch.long)

    def _build_contact_body_ids_tensor(self, contact_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []
        for body_name in contact_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert body_id != -1, f"Contact body '{body_name}' not found in actor"
            body_ids.append(body_id)
        return to_torch(body_ids, device=self.device, dtype=torch.long)

    def _action_to_pd_targets(self, action):
        """Convert actions to PD targets. Override for robot-specific joint coupling."""
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        pd_tar[..., :6] = pd_tar[..., :6] + self._dof_pos[..., :6]
        self.real_pd_tar = pd_tar
        return pd_tar

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0],
                              self._cam_prev_char_pos[1] - 3.0,
                              1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _update_camera(self):
        pass

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)


class DexploreTask(InterMimic):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # State initialization strategy (Default, Start, Random, Hybrid)
        state_init = cfg["env"]["stateInit"]
        self._state_init = DexploreTask.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        # Environment configuration
        self.motion_file = cfg['env']['motion_file']
        self.play_dataset = cfg['env']['playdataset']
        self.projtype = cfg['env']['projtype']
        self.reward_weights = cfg["env"]["rewardWeights"]
        self.save_images = cfg['env']['saveImages']
        self.init_vel = cfg['env']['initVel']
        self.ball_size = cfg['env']['ballSize']
        self.more_rigid = cfg['env']['moreRigid']
        self.rollout_length = cfg['env']['rolloutLength']
        self.robot_name = cfg['env']['robot_name']
        self.num_envs = cfg["env"]["numEnvs"]

        # Resolve motion files from directory or explicit list
        if isinstance(self.motion_file, list):
            assert len(self.motion_file) > 0
        else:
            motion_dir = self.motion_file
            motion_file = [p for p in os.listdir(motion_dir) if os.path.isdir(os.path.join(motion_dir, p))]
            # Primary: all motions except doorknob
            motion_file_1 = sorted([os.path.join(motion_dir, p) for p in motion_file if 'doorknob' not in p])
            # Supplementary: oversample small/difficult objects
            hard_objects = ('pan', 'flute', 'knife', 'scissors', 'toothbrush', 'teapot', 'small', 'watch')
            motion_file_2 = sorted([os.path.join(motion_dir, p) for p in motion_file if any(obj in p for obj in hard_objects)])
            self.motion_file = motion_file_1 + motion_file_2

        # Load table data and filter valid motions
        self._load_table(self.motion_file)

        # Build object name -> ID mapping
        self.object_name = [motion_example.split('/')[-1].split('_')[1] for motion_example in self.motion_file]
        object_name_set = sorted(list(set(self.object_name)))
        self.object_id = to_torch([object_name_set.index(name) for name in self.object_name], dtype=torch.long).cuda()
        self.obj2motion = torch.stack([self.object_id == k for k in range(len(object_name_set))], dim=0)
        self.object_name = object_name_set

        self.robot_type = self._get_robot_type()

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

        # Load motion data and initialize observation buffers
        self.fps_data = 60 / cfg['env']['controlFrequencyInv']
        self._load_motion(self.motion_file)
        self._curr_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._curr_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._reset_ig = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)
        self._build_target_tensors()

        # Adaptive early termination tracking (paper Sec. 3.2)
        # Per-motion success rate: κ = N_fail / N_total — thresholds tighten as policy improves
        # Only active when stateInit == "Start" (all rollouts start from frame 0)
        self._term_total_count = torch.ones(self.num_motions, device=self.device)
        self._term_fail_count = torch.ones(self.num_motions, device=self.device)
        self._adaptive_kappa_enabled = (self._state_init == DexploreTask.StateInit.Start)

        # Collision filter curriculum: per-motion tracking, start with filter=2,
        # switch to filter=1 once that motion's success rate exceeds threshold
        self._table_col_upgraded_motions = torch.zeros(self.num_motions, device=self.device, dtype=torch.bool)

        if self.projtype in ("Mouse", "Auto"):
            self._build_proj_tensors()

    def _get_robot_type(self):
        """Override to specify URDF path for the robot hand."""
        return f"{self.robot_name}_hand_new/{self.robot_name}_hand_right.urdf"

    def _compute_reward(self, actions):
        super()._compute_reward(actions)

    def _compute_reset(self):
        super()._compute_reset()

    def post_physics_step(self):
        self.progress_buf += 1
        self._refresh_sim_tensors()
        env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self._update_hist_hoi_obs()
        self._compute_hoi_observations()
        self._compute_observations(env_ids)

        # Adaptive early termination: per-motion kappa from historical success rate
        if self._adaptive_kappa_enabled:
            self._kappa = (self._term_fail_count[self.data_id] /
                           self._term_total_count[self.data_id]).clamp(min=0.5)

        self._compute_reward(self.actions)
        self._compute_reset()

        # Update per-motion success statistics on episode end
        if self._adaptive_kappa_enabled:
            done = self.reset_buf.bool()
            if done.any():
                self._term_total_count.index_put_(
                    (self.data_id[done],),
                    torch.ones(done.sum(), device=self.device), accumulate=True)
                terminated = self._terminate_buf[done].bool()
                if terminated.any():
                    done_ids = done.nonzero(as_tuple=False).squeeze(-1)
                    fail_ids = self.data_id[done_ids[terminated]]
                    self._term_fail_count.index_put_(
                        (fail_ids,),
                        torch.ones(fail_ids.shape[0], device=self.device), accumulate=True)

        # Collision filter curriculum: per-motion upgrade once success rate is high enough
        if (self._adaptive_kappa_enabled and hasattr(self, '_table_handles')
                and not self._table_col_upgraded_motions.all()):
            per_motion_success = 1.0 - (self._term_fail_count / self._term_total_count)
            newly_upgraded = (per_motion_success > 0.5) & (~self._table_col_upgraded_motions)
            if newly_upgraded.any():
                self._table_col_upgraded_motions |= newly_upgraded
                for env_id in range(len(self.envs)):
                    motion_id = env_id % self.num_motions
                    if newly_upgraded[motion_id]:
                        props = self.gym.get_actor_rigid_shape_properties(
                            self.envs[env_id], self._table_handles[env_id])
                        for p in props:
                            p.filter = 1
                        self.gym.set_actor_rigid_shape_properties(
                            self.envs[env_id], self._table_handles[env_id], props)
                for mid in newly_upgraded.nonzero(as_tuple=False).squeeze(-1):
                    print(f"[Curriculum] Motion {mid.item()} table collision upgraded to 1 "
                          f"(success rate: {per_motion_success[mid]:.1%})")

        self.extras["terminate"] = self._terminate_buf

        if self.viewer and self.debug_viz:
            self._update_debug_viz()

    def _update_hist_hoi_obs(self, env_ids=None):
        self._hist_obs = self._curr_obs.clone()

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

    def _load_table(self, motion_file):
        """Load table poses and filter out motions with contact in the first frame."""
        self.table_data = []
        if not isinstance(motion_file, list):
            motion_file = [motion_file]
        motion_file_new = []
        for i, data_path in enumerate(motion_file):
            if len(motion_file_new) > self.num_envs:
                break
            loaded_dict = {}
            hoi_data = torch.load(data_path + f'/interaction_hand_{self.robot_name}.pt')
            loaded_dict['table_pos'] = hoi_data[0, 238:241].clone().detach().to('cuda')
            loaded_dict['table_rot'] = hoi_data[0, 241:245].clone().detach().to('cuda')
            contact_parts = torch.round(hoi_data[:, 206:206+32].clone())[:, :16]
            if (contact_parts > 0).any():
                continue
            motion_file_new.append(data_path)
            self.table_data.append(loaded_dict)
        self.motion_file = motion_file_new
        self.num_motions = len(motion_file_new)
        print('num_motions: ', self.num_motions)

    def _load_motion(self, motion_file):
        """Load HOI motion data and build reference observation tensors.
        Each motion contains: hand pose, object pose, contact labels, interaction graph.
        """
        self.hoi_data_dict = []
        hoi_datas = []
        hoi_refs = []
        if not isinstance(motion_file, list):
            motion_file = [motion_file]
        self.num_motions = len(motion_file)
        self.max_episode_length = []
        self.start_contact_idx = []
        for idx, data_path in enumerate(motion_file):
            loaded_dict = {}
            hoi_data = torch.load(data_path + f'/interaction_hand_{self.robot_name}.pt')
            loaded_dict['hoi_data'] = hoi_data.detach().to('cuda')

            self.max_episode_length.append(loaded_dict['hoi_data'].shape[0])

            loaded_dict['left_hand_pos'] = loaded_dict['hoi_data'][:, 0:3].clone()
            loaded_dict['left_hand_pos_vel'] = (loaded_dict['left_hand_pos'][1:,:].clone() - loaded_dict['left_hand_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['left_hand_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['left_hand_pos_vel'].shape[-1])).to('cuda'),loaded_dict['left_hand_pos_vel']),dim=0)

            loaded_dict['left_dof_pos'] = loaded_dict['hoi_data'][:, 3:51].clone()
            loaded_dict['left_hand_rot'] = torch_utils.exp_map_to_quat(loaded_dict['hoi_data'][:, 3:6].clone())

            loaded_dict['left_dof_pos_vel'] = []
            loaded_dict['left_dof_pos_vel'] = (loaded_dict['left_dof_pos'][1:].clone() - loaded_dict['left_dof_pos'][:-1].clone())*self.fps_data
            loaded_dict['left_dof_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['left_dof_pos_vel'].shape[-1])).to('cuda'),loaded_dict['left_dof_pos_vel']),dim=0)


            loaded_dict['right_hand_pos'] = loaded_dict['hoi_data'][:, 51:54].clone()
            loaded_dict['right_hand_pos_vel'] = (loaded_dict['right_hand_pos'][1:,:].clone() - loaded_dict['right_hand_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['right_hand_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['right_hand_pos_vel'].shape[-1])).to('cuda'),loaded_dict['right_hand_pos_vel']),dim=0)

            loaded_dict['right_dof_pos'] = loaded_dict['hoi_data'][:, 54:102].clone()
            loaded_dict['right_hand_rot'] = torch_utils.exp_map_to_quat(loaded_dict['hoi_data'][:, 54:57].clone())

            loaded_dict['right_dof_pos_vel'] = []
            loaded_dict['right_dof_pos_vel'] = (loaded_dict['right_dof_pos'][1:].clone() - loaded_dict['right_dof_pos'][:-1].clone())*self.fps_data
            loaded_dict['right_dof_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['right_dof_pos_vel'].shape[-1])).to('cuda'),loaded_dict['right_dof_pos_vel']),dim=0)

            loaded_dict['body_pos'] = loaded_dict['hoi_data'][:, 102: 102+32*3].clone().view(self.max_episode_length[-1],32,3)[:, 16:, :]
            loaded_dict['body_pos'][..., 6:] = torch.remainder(loaded_dict['body_pos'][..., 6:], 2 * math.pi)
            loaded_dict['key_body_pos'] = loaded_dict['body_pos'][:, :, :].view(self.max_episode_length[-1],-1).clone()
            loaded_dict['key_body_pos_vel'] = (loaded_dict['key_body_pos'][1:,:].clone() - loaded_dict['key_body_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['key_body_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['key_body_pos_vel'].shape[-1])).to('cuda'),loaded_dict['key_body_pos_vel']),dim=0)

            loaded_dict['obj_pos'] = loaded_dict['hoi_data'][:, 198:201].clone()
            loaded_dict['obj_pos_vel'] = (loaded_dict['obj_pos'][1:,:].clone() - loaded_dict['obj_pos'][:-1,:].clone())*self.fps_data
            if self.init_vel:
                loaded_dict['obj_pos_vel'] = torch.cat((loaded_dict['obj_pos_vel'][:1],loaded_dict['obj_pos_vel']),dim=0)
            else:
                loaded_dict['obj_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['obj_pos_vel'].shape[-1])).to('cuda'),loaded_dict['obj_pos_vel']),dim=0)

            loaded_dict['obj_rot'] = loaded_dict['hoi_data'][:, 201:205].clone()
            obj_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['obj_rot'])
            loaded_dict['obj_rot_vel'] = (obj_rot_exp_map[1:,:].clone() - obj_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['obj_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['obj_rot_vel'].shape[-1])).to('cuda'),loaded_dict['obj_rot_vel']),dim=0)

            obj_rot_extend = loaded_dict['obj_rot'].unsqueeze(1).repeat(1, self.object_points[self.object_id[idx]].shape[0], 1).view(-1, 4)
            object_points_extend = self.object_points[self.object_id[idx]].unsqueeze(0).repeat(loaded_dict['obj_rot'].shape[0], 1, 1).view(-1, 3)
            obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(loaded_dict['obj_rot'].shape[0], self.object_points[self.object_id[idx]].shape[0], 3) + loaded_dict['obj_pos'].unsqueeze(1)
            key_body_pose = loaded_dict['key_body_pos'][:,:].clone()
            ref_ig = compute_sdf(key_body_pose.view(loaded_dict['obj_rot'].shape[0],-1,3), obj_points).view(-1, 3)
            heading_rot = torch_utils.calc_heading_quat_inv(loaded_dict['right_hand_rot'])
            heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, key_body_pose.shape[1] // 3, 1).view(-1, 4)
            ref_ig = quat_rotate(heading_rot_extend, ref_ig).view(loaded_dict['obj_rot'].shape[0], -1)
            ref_ig_min = ref_ig.view(loaded_dict['obj_rot'].shape[0], -1, 3).norm(dim=-1).min(dim=-1)[0]
            idxs = torch.where(ref_ig_min < 0.2)[0]
            first_idx = idxs[0].item()
            self.start_contact_idx.append(first_idx)
            loaded_dict['contact'] = torch.round(loaded_dict['hoi_data'][:, 205:206].clone())
            loaded_dict['contact_parts'] = torch.round(loaded_dict['hoi_data'][:, 206:206+32].clone())[:, 16:]
            loaded_dict['table_pos'] = loaded_dict['hoi_data'][:, 238:241].clone()

            loaded_dict['table_rot'] = loaded_dict['hoi_data'][:, 241:245].clone()
            loaded_dict['human_rot'] = loaded_dict['hoi_data'][:, 245:245+32*4].clone()[:, 16*4:32*4]
            human_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['human_rot'].reshape(-1, 4)).view(-1, 16*3)
            loaded_dict['human_rot_vel'] = []
            loaded_dict['human_rot_vel'] = (human_rot_exp_map[1:,:].clone() - human_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['human_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['human_rot_vel'].shape[-1])).to('cuda'),loaded_dict['human_rot_vel']),dim=0)
            loaded_dict['robot_dof_pos'] = loaded_dict['hoi_data'][:, 245+32*4:245+32*4+self.num_dof].clone()
            loaded_dict['robot_dof_pos_vel'] = (loaded_dict['robot_dof_pos'][1:,:].clone() - loaded_dict['robot_dof_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['robot_dof_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['robot_dof_pos_vel'].shape[-1])).to('cuda'),loaded_dict['robot_dof_pos_vel']),dim=0)

            loaded_dict['hoi_data'] = torch.cat((   loaded_dict['right_hand_rot'].clone(),
                                                    loaded_dict['right_hand_pos'].clone(),
                                                    loaded_dict['right_dof_pos'].clone(),
                                                    loaded_dict['right_hand_pos_vel'].clone(),
                                                    loaded_dict['right_dof_pos_vel'].clone(),
                                                    loaded_dict['obj_pos'].clone(),
                                                    loaded_dict['obj_rot'].clone(),
                                                    loaded_dict['obj_pos_vel'].clone(),
                                                    loaded_dict['obj_rot_vel'].clone(),
                                                    loaded_dict['key_body_pos'][:,:].clone(),
                                                    loaded_dict['contact'].clone(),
                                                    loaded_dict['contact_parts'].clone(),
                                                    ref_ig.clone(),
                                                    loaded_dict['human_rot'].clone(),
                                                    loaded_dict['key_body_pos_vel'].clone(),
                                                    loaded_dict['human_rot_vel'].clone(),
                                                    loaded_dict['robot_dof_pos'].clone(),
                                                    loaded_dict['robot_dof_pos_vel'].clone(),
                                                    ),dim=-1)

            assert(self.ref_hoi_obs_size == loaded_dict['hoi_data'].shape[-1])
            self.hoi_data_dict.append(loaded_dict)
            hoi_datas.append(loaded_dict['hoi_data'])
            hoi_ref = torch.cat((
                                loaded_dict['right_hand_rot'].clone(),
                                loaded_dict['right_hand_pos'].clone(),
                                loaded_dict['right_dof_pos'].clone(),
                                loaded_dict['right_hand_pos_vel'].clone(),
                                loaded_dict['right_dof_pos_vel'].clone(),
                                loaded_dict['obj_pos'].clone(),
                                loaded_dict['obj_rot'].clone(),
                                loaded_dict['obj_pos_vel'].clone(),
                                loaded_dict['obj_rot_vel'].clone(),
                                loaded_dict['robot_dof_pos'].clone(),
                                loaded_dict['robot_dof_pos_vel'].clone(),
                                ),dim=-1)
            hoi_refs.append(hoi_ref)

        max_length = max(self.max_episode_length)
        self.max_episode_length = to_torch(self.max_episode_length, dtype=torch.long)
        self.start_contact_idx = to_torch(self.start_contact_idx, dtype=torch.long)
        self.hoi_data = []
        self.hoi_refs = []
        for i, data in enumerate(hoi_datas):
            pad_size = (0, 0, 0, max_length - data.size(0))
            padded_data = F.pad(data, pad_size, "constant", 0)
            self.hoi_data.append(padded_data)
            self.hoi_refs.append(F.pad(hoi_refs[i], pad_size, "constant", 0))
        self.hoi_data = torch.stack(self.hoi_data, dim=0)
        self.hoi_refs = torch.stack(self.hoi_refs, dim=0).unsqueeze(1).repeat(1, 1, 1, 1)
        self.ref_reward = torch.zeros((self.hoi_refs.shape[0], self.hoi_refs.shape[1], self.hoi_refs.shape[2])).to(self.hoi_refs.device)
        self.ref_reward[:, :, 0] = 1.0
        self.ref_index = torch.zeros((self.num_envs, )).long().to(self.hoi_refs.device)

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._target_handles = []
        self._table_handles = []
        self._load_table_asset()
        self._load_target_asset()
        super()._create_envs(num_envs, spacing, num_per_row)

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_table(env_id, env_ptr)
        self._build_target(env_id, env_ptr)
        if self.projtype in ("Mouse", "Auto"):
            self._build_proj(env_id, env_ptr)

    def _build_proj(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        for i, obj in enumerate(PERTURB_OBJS):
            default_pose = gymapi.Transform()
            default_pose.p.x = 200 + i
            default_pose.p.z = 1
            obj_type = obj[0]
            if (obj_type == "small"):
                proj_asset = self._small_proj_asset
            elif (obj_type == "large"):
                proj_asset = self._large_proj_asset

            proj_handle = self.gym.create_actor(env_ptr, proj_asset, default_pose, "proj{:d}".format(i), col_group, col_filter, segmentation_id)
            self._proj_handles.append(proj_handle)
            self.gym.set_actor_scale(env_ptr, proj_handle, 1)

    def _build_proj_tensors(self):
        self._proj_dist_min = 4
        self._proj_dist_max = 5
        self._proj_h_min = 0.25
        self._proj_h_max = 2
        self._proj_steps = 150
        self._proj_warmup_steps = 1
        self._proj_speed_min = 30
        self._proj_speed_max = 40

        num_actors = self.get_num_actors_per_env()
        num_objs = len(PERTURB_OBJS)
        self._proj_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., (num_actors - num_objs):, :]

        self._proj_actor_ids = num_actors * np.arange(self.num_envs)
        self._proj_actor_ids = np.expand_dims(self._proj_actor_ids, axis=-1)
        self._proj_actor_ids = self._proj_actor_ids + np.reshape(np.array(self._proj_handles), [self.num_envs, num_objs])
        self._proj_actor_ids = self._proj_actor_ids.flatten()
        self._proj_actor_ids = to_torch(self._proj_actor_ids, device=self.device, dtype=torch.int32)

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._proj_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., (num_actors - num_objs):, :]

        self._calc_perturb_times()

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "space_shoot")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")
        self.gym.subscribe_viewer_mouse_event(self.viewer, gymapi.MOUSE_LEFT_BUTTON, "mouse_shoot")

    def _load_proj_asset(self):
        asset_root = "dexplore/data/assets/mjcf/"

        small_asset_file = "block_projectile.urdf"
        small_asset_options = gymapi.AssetOptions()
        small_asset_options.angular_damping = 0.01
        small_asset_options.linear_damping = 0.01
        small_asset_options.max_angular_velocity = 100.0
        small_asset_options.density = 200.0
        small_asset_options.fix_base_link = True
        small_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._small_proj_asset = self.gym.load_asset(self.sim, asset_root, small_asset_file, small_asset_options)

    def _load_marker_asset(self):
        asset_root = "dexplore/data/assets/mjcf/"
        asset_file = "location_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 0.0
        asset_options.density = 0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    def _load_target_asset(self):
        """Load object meshes, create VHACD collision assets, and sample surface points."""
        asset_root = "dexplore/data/assets/mjcf/"
        # Left-hand robots train on mirrored GRAB data (data_processing/make_mirrored_objects.py)
        mirror = self.cfg["env"].get("objectMirror", False)
        self._target_asset = []
        self.object_points = []
        for i, object_name in enumerate(self.object_name):
            asset_file = ("mirror/" if mirror else "") + object_name + ".urdf"
            max_convex_hulls = 20
            density = 20
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01
            asset_options.density = density
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params.max_convex_hulls = max_convex_hulls
            asset_options.vhacd_params.max_num_vertices_per_ch = 16
            asset_options.vhacd_params.resolution = 50000

            self._target_asset.append(self.gym.load_asset(self.sim, asset_root, asset_file, asset_options))

            obj_file = asset_root + ('objects_mirror/' if mirror else 'objects/') + object_name + '/' + object_name + '.obj'
            mesh_obj = trimesh.load(obj_file, force='mesh')
            obj_verts = mesh_obj.vertices
            center = np.mean(obj_verts, 0)
            object_points, object_faces = trimesh.sample.sample_surface_even(mesh_obj, count=256, seed=2024)
            object_points = to_torch(object_points)
            while object_points.shape[0] < 256:
                object_points = torch.cat([object_points, object_points[:256 - object_points.shape[0]]], dim=0)
            self.object_points.append(to_torch(object_points))

        self.object_points = torch.stack(self.object_points, dim=0)

    def _load_table_asset(self):
        asset_root = "dexplore/data/assets/mjcf/"
        object_name = "table"
        asset_file = object_name + ".urdf"
        table_options = gymapi.AssetOptions()
        table_options.fix_base_link = True
        table_options.density = 100000
        table_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._table_asset = self.gym.load_asset(self.sim, asset_root, asset_file, table_options)

    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 2
        segmentation_id = 2

        default_pose = gymapi.Transform()

        target_handle = self.gym.create_actor(env_ptr, self._target_asset[self.object_id[env_id % self.num_motions]], default_pose, self.object_name[self.object_id[env_id % self.num_motions]], col_group, 0, segmentation_id)

        self._target_handles.append(target_handle)
        self.gym.set_actor_scale(env_ptr, target_handle, self.ball_size)

    def _build_table(self, env_id, env_ptr):
        col_group = env_id
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = self.table_data[env_id % self.num_motions]["table_pos"][0]
        default_pose.p.y = self.table_data[env_id % self.num_motions]["table_pos"][1]
        default_pose.p.z = self.table_data[env_id % self.num_motions]["table_pos"][2]

        default_pose.r.x = self.table_data[env_id % self.num_motions]["table_rot"][0]
        default_pose.r.y = self.table_data[env_id % self.num_motions]["table_rot"][1]
        default_pose.r.z = self.table_data[env_id % self.num_motions]["table_rot"][2]
        default_pose.r.w = self.table_data[env_id % self.num_motions]["table_rot"][3]

        table_col_filter = 1 if self.cfg["env"].get("is_test", False) else 2
        table_handle = self.gym.create_actor(env_ptr, self._table_asset, default_pose, "table", col_group, table_col_filter, segmentation_id)

        if not self.cfg["headless"]:
            self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL,
                                        gymapi.Vec3(0.7, 0.6, 0.3))

        self._table_handles.append(table_handle)
        self.gym.set_actor_scale(env_ptr, table_handle, 1)

    def _build_marker(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 2
        segmentation_id = 0

        default_pose = gymapi.Transform()

        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self._marker_handles.append(marker_handle)

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 2, :]
        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 2

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies+1, :]

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 2, :]
        self._marker_pos = self._marker_states[..., :3]
        self._marker_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 2

    def _reset_target(self, env_ids):
        self._target_states[env_ids, :3] = self.hoi_refs[self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids], 106:109]
        self._target_states[env_ids, 3:7] = self.hoi_refs[self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids], 109:113]
        self._target_states[env_ids, 7:10] = self.hoi_refs[self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids], 113:116]
        self._target_states[env_ids, 10:13] = self.hoi_refs[self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids], 116:119]

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super()._reset_envs(env_ids)

    def _reset_actors(self, env_ids):
        if self._state_init == DexploreTask.StateInit.Default:
            self._reset_default(env_ids)
        elif self._state_init in (DexploreTask.StateInit.Start, DexploreTask.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif self._state_init == DexploreTask.StateInit.Hybrid:
            self._reset_hybrid_state_init(env_ids)
        else:
            raise ValueError(f"Unsupported state initialization strategy: {self._state_init}")
        self._reset_target(env_ids)

    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids

    def _reset_ref_state_init(self, env_ids):
        """Reset environments to reference motion states (Start or Random init)."""
        num_envs = env_ids.shape[0]
        i = torch.tensor([env_id % self.num_motions for env_id in env_ids], device=self.device)
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if self._state_init in (DexploreTask.StateInit.Random, DexploreTask.StateInit.Hybrid, DexploreTask.StateInit.Start):
            motion_times = []
            for e in range(num_envs):
                if env_ids[e] not in ref_reset_ids:
                    n = max(1, self.start_contact_idx[i[e]])
                    weights = torch.arange(1, n+1, device=self.device, dtype=torch.float)
                    probs = weights / weights.sum()
                    motion_times.append(torch.multinomial(probs, 1))
                else:
                    motion_times.append(torch.zeros((1,), device=self.device, dtype=torch.long))
            motion_times = torch.cat(motion_times)

        self.ref_index[env_ids] = 0
        self.progress_buf[env_ids] = motion_times
        self.start_times[env_ids] = motion_times
        self.data_id[env_ids] = i
        self._hist_obs[env_ids] = 0
        self.contact_reset[env_ids] = 0
        self._set_env_state(env_ids=env_ids,
                            dof_pos=self.hoi_refs[i, 0, motion_times, 119:119+self.num_dof],
                            dof_vel=self.hoi_refs[i, 0, motion_times, 119+self.num_dof:119+self.num_dof*2])

    def cal_cdf(self, i, e):
        rewards = self.ref_reward[i[e],
                                :,
                                :max(1, self.max_episode_length[i[e]] - self.rollout_length)
                                ].clone()

        sum_rewards = rewards.sum(dim=0)

        positive_mask = sum_rewards > 0

        inv_weights = torch.zeros_like(sum_rewards)
        inv_weights[positive_mask] = 1.0 / (sum_rewards[positive_mask] + 1e-5)

        prob = inv_weights / (inv_weights.sum() + 1e-8)

        cdf = torch.cumsum(prob, dim=0)

        return cdf

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        i = torch.tensor([env_id % self.num_motions for env_id in env_ids], device=self.device)
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]

        motion_times = torch.cat([torch.searchsorted(self.cal_cdf(i, e), torch.rand(1).to(self.device)) if env_ids[e] not in ref_reset_ids else torch.zeros((1,), device=self.device, dtype=torch.long) for e in range(num_envs)])
        ref_reward = self.ref_reward[i, :, motion_times]
        assert (ref_reward.sum(dim=1) > 0).all().item()
        prob = ref_reward / ref_reward.sum(1, keepdim=True)
        cdf = torch.cumsum(prob, dim=1)
        idx = torch.searchsorted(cdf, torch.rand((cdf.shape[0], 1)).to(cdf.device)).squeeze(1)
        non_i = self.ref_reward[i, idx, motion_times] < 0.7
        idx[non_i] = 0
        motion_times[non_i] = 0
        self.ref_index[env_ids] = idx
        self.progress_buf[env_ids] = motion_times.clone()
        self.start_times[env_ids] = motion_times.clone()
        self.data_id[env_ids] = i
        self._hist_obs[env_ids] = 0
        self.contact_reset[env_ids] = 0
        self._set_env_state(env_ids=env_ids,
                            dof_pos=self.hoi_refs[i, idx, motion_times, 119:119+self.num_dof],
                            dof_vel=self.hoi_refs[i, idx, motion_times, 119+self.num_dof:119+self.num_dof*2])

    def _set_env_state(self, env_ids, dof_pos, dof_vel):
        """Set environment state. Override for robot-specific joint coupling at reset."""
        self._humanoid_root_states[:, :] = 0
        self._humanoid_root_states[:, 6:7] = 1

        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

    def _compute_hoi_observations(self, env_ids=None):
        """Build HOI observation vector from current body/object state (used for reward)."""
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        key_body_vel = self._rigid_body_vel[:, self._key_body_ids, :]
        key_body_rot = self._rigid_body_rot[:, self._key_body_ids, :]
        key_body_ang_vel = self._rigid_body_ang_vel[:, self._key_body_ids, :]
        if env_ids is None:
            self._curr_obs[:] = build_hoi_observations(self._rigid_body_pos[:, 6, :],
                                                               self._rigid_body_rot[:, 6, :],
                                                               self._rigid_body_vel[:, 6, :],
                                                               self._rigid_body_ang_vel[:, 6, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs,
                                                               self._dof_obs_size, self._target_states,
                                                               self._tar_contact_forces,
                                                               self._contact_forces[:, self._key_body_ids, :],
                                                               self.object_points[self.object_id[self.data_id]] * self.ball_size,
                                                               key_body_rot,
                                                               key_body_vel,
                                                               key_body_ang_vel
                                                               )
        else:
            self._curr_obs[env_ids] = build_hoi_observations(self._rigid_body_pos[env_ids][:, 6, :],
                                                                   self._rigid_body_rot[env_ids][:, 6, :],
                                                                   self._rigid_body_vel[env_ids][:, 6, :],
                                                                   self._rigid_body_ang_vel[env_ids][:, 6, :],
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                   self._local_root_obs, self._root_height_obs,
                                                                   self._dof_obs_size, self._target_states[env_ids],
                                                                   self._tar_contact_forces[env_ids],
                                                                   self._contact_forces[env_ids][:, self._key_body_ids, :],
                                                                   self.object_points[self.object_id[self.data_id[env_ids]]] * self.ball_size,
                                                                   key_body_rot[env_ids],
                                                                   key_body_vel[env_ids],
                                                                   key_body_ang_vel[env_ids]).float()

    def _calc_perturb_times(self):
        self._perturb_timesteps = []
        total_steps = 0
        for i, obj in enumerate(PERTURB_OBJS):
            curr_time = obj[1]
            total_steps += curr_time
            self._perturb_timesteps.append(total_steps)

        self._perturb_timesteps = np.array(self._perturb_timesteps)

    def _update_proj(self):
        if self.projtype == 'Auto':
            curr_timestep = self.progress_buf.cpu().numpy()[0]
            curr_timestep = curr_timestep % (self._perturb_timesteps[-1] + 1)
            perturb_step = np.where(self._perturb_timesteps == curr_timestep)[0]

            if (len(perturb_step) > 0):
                perturb_id = perturb_step[0]
                n = self.num_envs
                humanoid_root_pos = self._humanoid_root_states[..., 0:3]

                rand_theta = torch.rand([n], dtype=self._proj_states.dtype, device=self._proj_states.device)
                rand_theta *= 2 * np.pi
                rand_dist = (self._proj_dist_max - self._proj_dist_min) * torch.rand([n], dtype=self._proj_states.dtype, device=self._proj_states.device) + self._proj_dist_min
                pos_x = rand_dist * torch.cos(rand_theta)
                pos_y = -rand_dist * torch.sin(rand_theta)
                pos_z = (self._proj_h_max - self._proj_h_min) * torch.rand([n], dtype=self._proj_states.dtype, device=self._proj_states.device) + self._proj_h_min

                self._proj_states[..., perturb_id, 0] = humanoid_root_pos[..., 0] + pos_x
                self._proj_states[..., perturb_id, 1] = humanoid_root_pos[..., 1] + pos_y
                self._proj_states[..., perturb_id, 2] = pos_z
                self._proj_states[..., perturb_id, 3:6] = 0.0
                self._proj_states[..., perturb_id, 6] = 1.0

                tar_body_idx = np.random.randint(self.num_bodies)
                tar_body_idx = 1

                launch_tar_pos = self._rigid_body_pos[..., tar_body_idx, :]
                launch_dir = launch_tar_pos - self._proj_states[..., perturb_id, 0:3]
                launch_dir += 0.1 * torch.randn_like(launch_dir)
                launch_dir = torch.nn.functional.normalize(launch_dir, dim=-1)
                launch_speed = (self._proj_speed_max - self._proj_speed_min) * torch.rand_like(launch_dir[:, 0:1]) + self._proj_speed_min
                launch_vel = launch_speed * launch_dir
                launch_vel[..., 0:2] += self._rigid_body_vel[..., tar_body_idx, 0:2]
                self._proj_states[..., perturb_id, 7:10] = launch_vel
                self._proj_states[..., perturb_id, 10:13] = 0.0

                self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                             gymtorch.unwrap_tensor(self._proj_actor_ids),
                                                             len(self._proj_actor_ids))

        elif self.projtype == 'Mouse':
            # mouse control
            for evt in self.gym.query_viewer_action_events(self.viewer):

                if evt.action == "reset" and evt.value > 0:
                    self.gym.set_sim_rigid_body_states(self.sim, self._proj_states, gymapi.STATE_ALL)

                elif (evt.action == "space_shoot" or evt.action == "mouse_shoot") and evt.value > 0:
                    if evt.action == "mouse_shoot":
                        pos = self.gym.get_viewer_mouse_position(self.viewer)
                        window_size = self.gym.get_viewer_size(self.viewer)
                        xcoord = round(pos.x * window_size.x)
                        ycoord = round(pos.y * window_size.y)
                        print(f"Fired projectile with mouse at coords: {xcoord} {ycoord}")

                    cam_pose = self.gym.get_viewer_camera_transform(self.viewer, None)
                    cam_fwd = cam_pose.r.rotate(gymapi.Vec3(0, 0, 1))

                    spawn = cam_pose.p
                    speed = 25
                    vel = cam_fwd * speed

                    angvel = 1.57 - 3.14 * np.random.random(3)

                    self._proj_states[..., 0] = spawn.x
                    self._proj_states[..., 1] = spawn.y
                    self._proj_states[..., 2] = spawn.z
                    self._proj_states[..., 7] = vel.x
                    self._proj_states[..., 8] = vel.y
                    self._proj_states[..., 9] = vel.z

                    self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                            gymtorch.unwrap_tensor(self._proj_actor_ids),
                                                            len(self._proj_actor_ids))

    def play_dataset_step(self, time):
        """Step through the dataset for visualization (no policy rollout)."""
        t = time

        # Update object state
        self.data_id = to_torch([i % self.num_motions for i in range(self.num_envs)], device=self.device, dtype=torch.long)
        env_ids = to_torch([i for i in range(min(self.num_motions, self.num_envs)) if t < self.max_episode_length[i]], device=self.device, dtype=torch.long)

        self._target_states[env_ids, :3] = self.hoi_refs[self.data_id[env_ids], 0, t, 106:109]
        self._target_states[env_ids, 3:7] = self.hoi_refs[self.data_id[env_ids], 0, t, 109:113]
        self._target_states[env_ids, 7:10] = torch.zeros_like(self._target_states[env_ids, 7:10])
        self._target_states[env_ids, 10:13] = torch.zeros_like(self._target_states[env_ids, 10:13])

        # Update robot hand state
        self._humanoid_root_states[:, :] = 0
        self._humanoid_root_states[:, 6:7] = 1

        self._dof_pos[env_ids] = self.hoi_refs[self.data_id[env_ids], 0, t, 119:119+self.num_dof].clone()
        self._dof_vel[env_ids] = self.hoi_refs[self.data_id[env_ids], 0, t, 119+self.num_dof:119+2*self.num_dof].clone()

        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self._refresh_sim_tensors()

        human_contact = self.hoi_data[:, t, 119+len(self._key_body_ids)*3+1:119+len(self._key_body_ids)*3+1+16][..., [3, 6, 9, 12, 15]]
        for env_id, env_ptr in enumerate(self.envs):
            if env_id in env_ids:
                env_ptr = self.envs[env_id]
                handle = self._target_handles[env_id]

                handle = self.humanoid_handles[env_id]
                for j in range(self._contact_body_ids.shape[0]):
                    if human_contact[env_id, j] > 0.5:
                        self.gym.set_rigid_body_color(env_ptr, handle, self._contact_body_ids[j], gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(1., 0., 0.))
                    elif human_contact[env_id, j] > -0.5:
                        self.gym.set_rigid_body_color(env_ptr, handle, self._contact_body_ids[j], gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(0., 1., 0.))
                    else:
                        self.gym.set_rigid_body_color(env_ptr, handle, self._contact_body_ids[j], gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(0., 0., 1.))
        self.render(t=t)
        self.gym.simulate(self.sim)

    def _draw_task_play(self, t):
        """Draw reference skeleton lines in the viewer."""
        cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self.hoi_data_dict[0]['hoi_data'][t, :3]

        for i, env_ptr in enumerate(self.envs):
            for j in range(len(self._key_body_ids)):
                vec = self.hoi_data_dict[0]['key_body_pos'][t, j*3:j*3+3]
                vec = torch.cat([starts, vec], dim=-1).cpu().numpy().reshape([1, 6])
                self.gym.add_lines(self.viewer, env_ptr, 1, vec, cols)

    def render(self, sync_frame_time=False, t=0):
        super().render(sync_frame_time)
        if self.viewer:
            self._draw_task()
            if self.save_images:
                env_ids = 0
                frame_id = t if self.play_dataset else self.progress_buf[env_ids]
                dataname = self.motion_file[-1][len('dexplore/data/motions/'):]
                rgb_filename = f"dexplore/data/images/{dataname}/rgb_env{env_ids}_frame{frame_id:05d}.png"
                os.makedirs(f"dexplore/data/images/{dataname}", exist_ok=True)
                self.gym.write_viewer_image_to_file(self.viewer, rgb_filename)

    def _draw_task(self):
        pass


# ===================================================================
# Standalone / JIT-compiled functions
# ===================================================================

def build_hoi_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos,
                           local_root_obs, root_height_obs, dof_obs_size, target_states, target_contact_buf, contact_buf, object_points, body_rot, body_vel, body_rot_vel):
    """Build the HOI observation vector from current state (for reward computation).
    Includes: root state, object state, key body positions, contact flags, interaction graph.
    """
    contact = torch.any(torch.abs(contact_buf) > 0.1, dim=-1).float()
    target_contact = torch.any(torch.abs(target_contact_buf) > 0.1, dim=-1).float().unsqueeze(1)

    tar_pos = target_states[:, 0:3]
    tar_rot = target_states[:, 3:7]
    obj_rot_extend = tar_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
    object_points_extend = object_points.view(-1, 3)
    obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(tar_rot.shape[0], object_points.shape[1], 3) + tar_pos.unsqueeze(1)
    ig = compute_sdf(key_body_pos, obj_points).view(-1, 3)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, key_body_pos.shape[1], 1).view(-1, 4)
    ig = quat_rotate(heading_rot_extend, ig).view(tar_pos.shape[0], -1)
    obs = torch.cat((root_rot, root_pos, torch.zeros((root_pos.shape[0], 48), device=root_pos.device), root_vel, torch.zeros((root_pos.shape[0], 48), device=root_pos.device), target_states, key_body_pos.contiguous().view(-1,key_body_pos.shape[1]*key_body_pos.shape[2]), target_contact, contact, ig, body_rot.view(-1, key_body_pos.shape[1]*4), body_vel.view(-1,key_body_pos.shape[1]*key_body_pos.shape[2]), body_rot_vel.view(-1, key_body_pos.shape[1]*3), dof_pos, dof_vel), dim=-1)
    return obs


def compute_obj_observations(body_pos, body_rot, tar_states, object_points, ref_obs):
    """Compute object-centric observations: local object pose, velocity, and diffs from reference."""
    tar_pos = tar_states[:, 0:3]
    tar_rot = tar_states[:, 3:7]
    tar_vel = tar_states[:, 7:10]
    tar_ang_vel = tar_states[:, 10:13]
    root_pos = body_pos[:, 7]
    root_rot = body_rot[:, 7]
    ref_root_pos = ref_obs[:, 4:7]
    obj_rot_extend = tar_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
    object_points_extend = object_points.view(-1, 3)
    obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(tar_rot.shape[0], object_points.shape[1], 3) + tar_pos.unsqueeze(1)

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

    local_tar_pos = tar_pos - root_pos
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos)
    local_tar_vel = quat_rotate(heading_rot, tar_vel)
    local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)

    local_tar_rot = quat_mul(heading_rot, tar_rot)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

    _ref_obj_pos = ref_obs[:,106:109]
    diff_global_obj_pos = _ref_obj_pos - tar_pos
    diff_local_obj_pos_flat = torch_utils.quat_rotate(heading_rot, diff_global_obj_pos)

    local_ref_obj_pos = _ref_obj_pos - ref_root_pos
    local_ref_obj_pos = torch_utils.quat_rotate(heading_rot, local_ref_obj_pos)

    ref_obj_rot = ref_obs[:,109:113]
    diff_global_obj_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_obj_rot), tar_rot)
    diff_local_obj_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_rot, diff_global_obj_rot.view(-1, 4)), heading_inv_rot)
    diff_local_obj_rot_obs = torch_utils.quat_to_tan_norm(diff_local_obj_rot_flat)

    local_ref_obj_rot = torch_utils.quat_mul(heading_rot, ref_obj_rot)
    local_ref_obj_rot = torch_utils.quat_to_tan_norm(local_ref_obj_rot)

    ref_obj_vel = ref_obs[:, 113:116]
    diff_global_vel = ref_obj_vel - tar_vel
    diff_local_vel = torch_utils.quat_rotate(heading_rot, diff_global_vel)

    ref_obj_ang_vel = ref_obs[:, 116:119]
    diff_global_ang_vel = ref_obj_ang_vel - tar_ang_vel
    diff_local_ang_vel = torch_utils.quat_rotate(heading_rot, diff_global_ang_vel)

    obs = torch.cat([local_tar_pos, local_tar_rot_obs, local_tar_vel, local_tar_ang_vel, diff_local_obj_pos_flat, diff_local_obj_rot_obs, local_ref_obj_pos, local_ref_obj_rot, diff_local_vel, diff_local_ang_vel], dim=-1)
    return obs, obj_points


@torch.jit.script
def compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, contact_forces, contact_body_ids, ref_obs, key_body_ids):
    """Compute policy observation: body state in root-local frame + diffs from reference.
    Includes position, rotation, velocity diffs and contact diffs for all key bodies.
    """
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_pos = body_pos[:, 7, :]
    root_rot = body_rot[:, 7, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    len_keypos = len(key_body_ids)
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand_2 = heading_rot_expand.repeat((1, len_keypos, 1))
    flat_heading_rot_2 = heading_rot_expand_2.reshape(heading_rot_expand_2.shape[0] * heading_rot_expand_2.shape[1],
                                               heading_rot_expand_2.shape[2])

    heading_rot_expand = heading_rot_expand.repeat((1, len_keypos, 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                               heading_rot_expand.shape[2])

    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
    heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, len_keypos, 1))
    flat_heading_inv_rot = heading_inv_rot_expand.reshape(heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1],
                                               heading_inv_rot_expand.shape[2])

    _ref_body_pos = ref_obs[:,119:119+len_keypos*3].view(-1, len_keypos, 3)
    _body_pos = body_pos[:, key_body_ids, :]
    diff_global_body_pos = _ref_body_pos - _body_pos
    diff_local_body_pos_flat = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_body_pos.view(-1, 3)).view(-1, len_keypos * 3)

    local_ref_body_pos = _body_pos - root_pos.unsqueeze(1)
    local_ref_body_pos = torch_utils.quat_rotate(flat_heading_rot_2, local_ref_body_pos.view(-1, 3)).view(-1, len_keypos * 3)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos[:, key_body_ids, :] - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos
    flat_body_rot = body_rot[:, key_body_ids, :].reshape(body_rot.shape[0] * len_keypos, body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], len_keypos * flat_local_body_rot_obs.shape[1])

    ref_body_rot = ref_obs[:, 119+len_keypos*3+1+16+len_keypos*3: 119+len_keypos*3+1+16+len_keypos*3+16*4]
    diff_global_body_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot.reshape(-1, 4)), body_rot[:, key_body_ids, :].reshape(-1, 4))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(flat_heading_rot, diff_global_body_rot.view(-1, 4)), flat_heading_inv_rot)
    diff_local_body_rot_obs = torch_utils.quat_to_tan_norm(diff_local_body_rot_flat)
    diff_local_body_rot_obs = diff_local_body_rot_obs.view(body_rot.shape[0], len_keypos * diff_local_body_rot_obs.shape[-1])

    local_ref_body_rot = torch_utils.quat_mul(flat_heading_rot, ref_body_rot.reshape(-1, 4))
    local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot).view(ref_body_rot.shape[0], -1)

    ref_body_vel = ref_obs[:, 119+len_keypos*3+1+16+len_keypos*3+16*4:119+len_keypos*3+1+16+len_keypos*3+16*4+len_keypos*3].view(-1, len_keypos, 3)
    _body_vel = body_vel[:, key_body_ids, :]
    diff_global_vel = ref_body_vel - _body_vel
    diff_local_vel = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_vel.view(-1, 3)).view(-1, len_keypos * 3)

    ref_body_ang_vel = ref_obs[:, 119+len_keypos*3+1+16+len_keypos*3+16*4+len_keypos*3:119+len_keypos*3+1+16+len_keypos*3+16*4+len_keypos*3+16*3]
    diff_global_ang_vel = ref_body_ang_vel.view(-1, 16, 3) - body_ang_vel[:, key_body_ids, :]
    diff_local_ang_vel = torch_utils.quat_rotate(flat_heading_rot, diff_global_ang_vel.view(-1, 3)).view(-1, 16 * 3)

    if (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel[:, key_body_ids, :].reshape(body_vel.shape[0] * len_keypos, body_vel.shape[2])
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], len_keypos * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel[:, key_body_ids, :].reshape(body_ang_vel.shape[0] * len_keypos, body_ang_vel.shape[2])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], len_keypos * body_ang_vel.shape[2])

    body_contact_buf = contact_forces[:, contact_body_ids, :].clone()
    contact = torch.any(torch.abs(body_contact_buf) > 0.1, dim=-1).float()
    ref_body_contact = ref_obs[:,119+len_keypos*3+1:119+len_keypos*3+1+16][:, [3, 6, 9, 12, 15]]
    diff_body_contact = ref_body_contact * ((ref_body_contact + 1) / 2 - contact)

    root_pos = body_pos[:, 6, :]
    root_rot = body_rot[:, 6, :]
    ref_root = ref_obs[:, 4:7]
    ref_root_rot = ref_obs[:, :4]

    root_pos_vel = body_vel[:, 6, :]
    root_rot_vel = body_ang_vel[:, 6, :]

    diff_global_root_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_root_rot), root_rot)
    diff_global_root_pos = ref_root - root_pos
    diff_global_root_pos_flat = diff_global_root_pos.view(-1, 3)

    diff_global_root_rot_obs = torch_utils.quat_to_tan_norm(diff_global_root_rot.view(-1, 4))
    diff_global_root_rot_obs = diff_global_root_rot_obs.view(body_rot.shape[0], diff_global_root_rot_obs.shape[-1])

    diff_global_root_pos_vel_flat = root_pos_vel.view(-1, 3)

    diff_global_root_root_vel_flat = root_rot_vel.view(-1, 3)

    obs = torch.cat((diff_global_root_pos_flat, diff_global_root_rot_obs, diff_global_root_pos_vel_flat, diff_global_root_root_vel_flat, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, contact, diff_local_body_pos_flat, diff_local_body_rot_obs, diff_body_contact, local_ref_body_pos, local_ref_body_rot, diff_local_vel, diff_local_ang_vel), dim=-1)
    return obs


def huber_loss(diff, sigma=1.0):
    beta = 1. / (sigma ** 2)
    diff = torch.abs(diff)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss


def compute_humanoid_reward(hoi_ref, hoi_obs, contact_buf, tar_contact_forces, len_keypos, w, pos_actions, actions, object_points, object_pos_action, object_rot_action, num_dof, ball_size, t_sat, kappa):
    """Compute the task-agnostic HOI imitation reward.
    Combines: body tracking (position + rotation), object tracking,
    interaction graph matching, contact matching, and energy penalties.
    kappa: adaptive early termination factor (1.0=loose, 0.1=strict).
    Returns: (reward, ig_reset, contact_reset, kinematic_reset, metric_1, metric_2)
    """
    # --- Extract simulated states ---
    root_pos = hoi_obs[:,4:7]
    root_rot = hoi_obs[:,:4]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    dof_pos = hoi_obs[:,392:392+num_dof]
    dof_pos_vel = hoi_obs[:,392+num_dof:392+num_dof*2]
    obj_pos = hoi_obs[:,106:109]
    obj_rot = hoi_obs[:,109:113]

    local_obj_pos = obj_pos - root_pos
    local_obj_pos = quat_rotate(heading_rot, local_obj_pos)

    local_obj_rot = quat_mul(heading_rot, obj_rot)

    obj_rot_extend = obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
    object_points_extend = object_points.view(-1, 3)
    obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend * ball_size).view(obj_rot.shape[0], object_points.shape[1], 3) + obj_pos.unsqueeze(1)

    obj_pos_vel = hoi_obs[:,113:116]
    obj_rot_vel = hoi_obs[:,116:119]
    key_pos = hoi_obs[:,119:119+len_keypos*3]
    local_key_pos = key_pos.clone().view(-1, len_keypos, 3)
    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_pos[:, :, 2:3] = local_key_pos[:, :, 2:3] - root_pos_expand[:, :, 2:3]
    local_key_pos = local_key_pos.view(-1, (len_keypos)*3)
    body_rot = hoi_obs[:, 119+len_keypos*3+1+16+len_keypos*3:119+len_keypos*3+1+16+len_keypos*3+len_keypos*4].view(-1, len_keypos, 4)
    body_vel = hoi_obs[:, 119+len_keypos*3+1+16+len_keypos*3+len_keypos*4:119+len_keypos*3+1+16+len_keypos*3+len_keypos*4+len_keypos*3].view(-1, len_keypos, 3)
    ig = key_pos.view(-1,len_keypos,3)[:, [3, 6, 9, 12, 15, 3, 3, 15, 15], :].unsqueeze(2) - obj_points.unsqueeze(1)
    ig_2 = torch.cross(key_pos.view(-1,len_keypos,3)[:, 0, :] - key_pos.view(-1,len_keypos,3)[:, 1, :], key_pos.view(-1,len_keypos,3)[:, 0, :] - key_pos.view(-1,len_keypos,3)[:, 7, :], dim=-1)
    ig_2 = ig_2 / (ig_2.norm(dim=-1, keepdim=True) + 1e-6)
    rot_id = [[0, 1], [0, 4], [0, 7], [0, 10], [0, 13], [1, 3], [4, 6], [7, 9], [10, 12], [13, 15]]
    local_key_rot = [(key_pos.view(-1,len_keypos,3)[:, idx[0]] - key_pos.view(-1,len_keypos,3)[:, idx[1]]).unsqueeze(1) for idx in rot_id]
    local_key_rot = torch.cat(local_key_rot, dim=1)
    local_key_rot = (local_key_rot / (local_key_rot.norm(dim=-1, keepdim=True) + 1e-5))

    # --- Extract reference states ---
    ref_root_pos = hoi_ref[:,4:7]
    ref_root_rot = hoi_ref[:,:4]

    ref_heading_rot = torch_utils.calc_heading_quat_inv(ref_root_rot)

    ref_dof_pos = hoi_ref[:,392:392+num_dof]
    ref_dof_pos_vel = hoi_ref[:,392+num_dof:392+num_dof*2]
    ref_obj_pos = hoi_ref[:,106:109]
    ref_obj_rot = hoi_ref[:,109:113]

    ref_local_obj_pos = ref_obj_pos - ref_root_pos
    ref_local_obj_pos = quat_rotate(ref_heading_rot, ref_local_obj_pos)

    ref_local_obj_rot = quat_mul(ref_heading_rot, ref_obj_rot)

    ref_obj_rot_extend = ref_obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
    ref_obj_points = torch_utils.quat_rotate(ref_obj_rot_extend, object_points_extend).view(obj_rot.shape[0], object_points.shape[1], 3) + ref_obj_pos.unsqueeze(1)

    ref_obj_pos_vel = hoi_ref[:,113:116]
    ref_obj_rot_vel = hoi_ref[:,116:119]
    ref_key_pos = hoi_ref[:,119:119+len_keypos*3]
    ref_obj_contact = hoi_ref[:,119+len_keypos*3:119+len_keypos*3+1]
    ref_human_contact = hoi_ref[:,119+len_keypos*3+1:119+len_keypos*3+1+16]

    ref_local_key_pos = ref_key_pos.clone().view(-1, len_keypos, 3)
    ref_root_pos_expand = ref_root_pos.unsqueeze(-2)
    ref_local_key_pos[:, :, 2:3] = ref_local_key_pos[:, :, 2:3] - ref_root_pos_expand[:, :, 2:3]
    ref_local_key_pos = ref_local_key_pos.view(-1, (len_keypos)*3)
    ref_body_rot = hoi_ref[:, 119+len_keypos*3+1+16+len_keypos*3:119+len_keypos*3+1+16+len_keypos*3+16*4]
    ref_body_vel = hoi_ref[:, 119+len_keypos*3+1+16+len_keypos*3+16*4:119+len_keypos*3+1+16+len_keypos*3+16*4+len_keypos*3].view(-1, len_keypos, 3)
    ref_ig = ref_key_pos.view(-1,len_keypos,3)[:, [3, 6, 9, 12, 15, 3, 3, 15, 15], :].unsqueeze(2) - ref_obj_points.unsqueeze(1)
    ref_ig_2 = torch.cross(ref_key_pos.view(-1,len_keypos,3)[:, 0, :] - ref_key_pos.view(-1,len_keypos,3)[:, 1, :], ref_key_pos.view(-1,len_keypos,3)[:, 0, :] - ref_key_pos.view(-1,len_keypos,3)[:, 7, :], dim=-1)
    ref_ig_2 = ref_ig_2 / (ref_ig_2.norm(dim=-1, keepdim=True) + 1e-6)
    rot_id = [[0, 1], [0, 4], [0, 7], [0, 10], [0, 13], [1, 3], [4, 6], [7, 9], [10, 12], [13, 15]]
    ref_local_key_rot = [(ref_key_pos.view(-1,len_keypos,3)[:, idx[0]] - ref_key_pos.view(-1,len_keypos,3)[:, idx[1]]).unsqueeze(1) for idx in rot_id]
    ref_local_key_rot = torch.cat(ref_local_key_rot, dim=1)
    ref_local_key_rot = (ref_local_key_rot / (ref_local_key_rot.norm(dim=-1, keepdim=True) + 1e-5))

    # Adaptive early termination: thresholds scaled by kappa (tightens as policy improves)
    human_reset = (ref_key_pos - key_pos).view(-1, len_keypos, 3).norm(dim=-1).max(dim=-1)[0] > 0.8 * kappa
    object_reset = (obj_points - ref_obj_points).norm(dim=-1).max(dim=-1)[0] > 0.6 * kappa
    # --- Interaction graph (IG) reward ---
    w_ig = 10
    weight_1 = (1 / torch.clamp((ig**2).sum(dim=-1), min=0.0001))
    weight_1 = weight_1 / weight_1.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
    weight_2 = (1 / torch.clamp((ref_ig**2).sum(dim=-1), min=0.0001))
    weight_2 = weight_2 / weight_2.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)

    eig = (((ig - ref_ig).norm(dim=-1))**2 + (ig.norm(dim=-1) - ref_ig.norm(dim=-1))**2).abs() * (weight_2)
    rig = torch.exp(-w_ig * (eig.sum(dim=-1).sum(dim=-1)) * (1 - (torch.clamp(ref_ig.norm(dim=-1).min(dim=-1)[0].min(dim=-1)[0], max=1.0))) )

    kinematic_reset = torch.logical_or(human_reset, object_reset)
    reset_ig_1 = (((ig - ref_ig)**2).sum(dim=-1).sqrt() / torch.clamp((ig**2).sum(dim=-1).sqrt(), min=0.25)).max(dim=-1)[0].max(dim=-1)[0] > 2.0 * kappa
    reset_ig_2 = (((ig - ref_ig)**2).sum(dim=-1).sqrt() / torch.clamp((ref_ig**2).sum(dim=-1).sqrt(), min=0.25)).max(dim=-1)[0].max(dim=-1)[0] > 2.0 * kappa

    reset_ig = torch.logical_or(reset_ig_1, reset_ig_2)

    # --- Body tracking reward ---

    height = ref_key_pos.view(-1, len_keypos, 3)[:, :, 2].min(dim=-1)[0] - 0.01 - key_pos.view(-1, len_keypos, 3)[:, :, 2].min(dim=-1)[0]
    height_er = height.abs() * (height >= 0)
    # Body position reward (fingertip tracking)
    ep = (torch.mean(torch.sum((ref_key_pos - key_pos).view(-1, len_keypos, 3)[:, [3, 6, 9, 12, 15, 3, 15, 15, 15], :]**2, dim=-1),dim=-1)) * (torch.clamp(ref_ig.norm(dim=-1).min(dim=-1)[0].min(dim=-1)[0], max=1.0, min=0.01))
    rp = torch.exp(-ep*10)

    er = torch.mean(((ref_local_key_rot - local_key_rot)**2).sum(dim=-1),dim=-1) * (torch.clamp(ref_ig.norm(dim=-1).min(dim=-1)[0].min(dim=-1)[0], max=1.0, min=0.01))
    eig_2 = (((ig_2 - ref_ig_2)**2).sum(dim=-1)).abs()  * (torch.clamp(ref_ig.norm(dim=-1).min(dim=-1)[0].min(dim=-1)[0], max=1.0))
    rig_2 = torch.exp(-1 * eig_2)
    rr = torch.exp(-1 * er) * rig_2

    # Body velocity and energy rewards
    epv = torch.mean(((ref_body_vel - body_vel)**2).sum(dim=-1)[:, :],dim=-1)
    rpv = torch.exp(-epv*w['pv'])
    erv = torch.mean((dof_pos_vel)**2,dim=-1)
    rrv = torch.exp(-erv*w['rv'])
    energy = actions[..., :6].pow(2).mean(dim=-1).mul(-w['eg1']).exp()
    energy_pos = pos_actions.pow(2).mean(dim=-1).mul(-w['eg2']).exp()
    rb = rp*rr*rpv*rrv*energy_pos*energy

    # --- Object tracking reward ---
    eop = torch.mean((ref_obj_pos - obj_pos)**2,dim=-1)
    rop = torch.exp(-eop*w['op'])
    diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_obj_rot), obj_rot)
    diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
    diff = diff_angle.view(-1, 1)

    eor = torch.mean(huber_loss(diff, 3),dim=-1)
    ror = torch.exp(-eor*w['or'])

    eopv = torch.mean((ref_obj_pos_vel - obj_pos_vel)**2,dim=-1)
    ropv = torch.exp(-eopv*w['opv'])
    eorv = torch.mean((ref_obj_rot_vel - obj_rot_vel)**2,dim=-1)
    rorv = torch.exp(-eorv*w['orv'])
    obj_energy = (object_pos_action.pow(2).mean(dim=-1).mul(-w['eg2']).exp()) * (object_rot_action.pow(2).mean(dim=-1).mul(-w['eg2']).exp())
    ro = rop*ror*ropv*rorv*obj_energy

    # --- Contact matching reward ---
    contact_thres = 0.01
    obj_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > contact_thres, dim=-1).float()

    right_contact_hand_ids = [3, 6, 9, 12, 15]
    ref_right_contact_hand = ref_human_contact[:, right_contact_hand_ids]
    ref_right_contact_hand_any = torch.any(ref_right_contact_hand > contact_thres, dim=-1).float()
    right_hand_contact_buf = contact_buf.clone()
    right_hand_contact = torch.any(torch.abs(right_hand_contact_buf) > contact_thres, dim=-1).float()
    w_cg_right = 1.0
    ecg_right = (((ref_right_contact_hand > contact_thres) * torch.abs(right_hand_contact - ref_right_contact_hand))[:, [0, 4, 0, 4, 1, 2, 3]].sum(dim=-1))
    ig = key_pos.view(-1,len_keypos,3)[:, [3, 6, 9, 12, 15, 3, 15, 15, 15], :].unsqueeze(2) - obj_points.unsqueeze(1)
    eig_right_hand = (F.relu(ig.norm(dim=-1).min(dim=-1)[0][..., :5] - 0.01) * (ref_right_contact_hand > contact_thres))[..., [0, 1, 2, 3, 4, 0, 4, 0, 4]]
    rig_right_hand = torch.exp(-10 * (eig_right_hand.mean(dim=-1)))
    reset_ig_hand = (F.relu(ig.norm(dim=-1).min(dim=-1)[0][..., :5] - 0.07) * (ref_right_contact_hand > contact_thres))[..., [0, 4]].sum(dim=-1) > 1e-5
    reset_ig = torch.logical_or(reset_ig, reset_ig_hand)
    ecg_reset = torch.mean((dof_pos[..., [15, 15, 15, 14, 6, 6, 8, 12, 10]]**2),dim=-1)
    d = ref_ig.norm(dim=-1).min(dim=-1)[0].min(dim=-1)[0]
    mu = 0.1
    sigma = 0.05

    # Gaussian weighting around mu
    weight = (d / mu) * torch.exp(-((d - mu) ** 2) / (2 * sigma ** 2))
    weight = torch.clamp(weight, min=0.0, max=1.0)
    ecg_reset = weight * ecg_reset

    rcg_reset = torch.exp(-0.2 * ecg_reset)
    rcg_right = 0.5 * (torch.exp(-ecg_right*w_cg_right) + rig_right_hand) * (ref_right_contact_hand_any) + (1 - ref_right_contact_hand_any) * rcg_reset
    rcg3 = rcg_right

    # Contact mismatch flag (same value broadcast to 5 contact body slots)
    contact_mismatch = torch.abs(ref_right_contact_hand_any - torch.any(torch.abs(right_hand_contact) > contact_thres, dim=-1).float()) * (ref_right_contact_hand_any > contact_thres)
    contact_reset = contact_mismatch.unsqueeze(-1).repeat(1, 5)

    contact_all = contact_buf[:,:, :].clone().abs().sum(dim=-1).sum(dim=-1)
    contact_energy = contact_all.pow(2).mul(-w['eg3']).exp()
    rcg = rcg3*contact_energy

    # --- Final task-agnostic HOI imitation reward ---
    reward = rb*ro*rig*rcg
    metric_1 = (ref_key_pos - key_pos).view(-1, len_keypos, 3).norm(dim=-1).mean(dim=-1)
    metric_2 = (obj_points - ref_obj_points).norm(dim=-1).mean(dim=-1)
    return reward, reset_ig, contact_reset, kinematic_reset, metric_1, metric_2


def compute_humanoid_reset(reset_buf, progress_buf,
                           max_episode_length, enable_early_termination, start_times, rollout_length, reset_ig, contact_reset):
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        reset_ig *= (progress_buf > 5 + start_times)
        contact_reset *= (progress_buf > 5 + start_times)
        terminated = torch.where(torch.logical_or(reset_ig, contact_reset), torch.ones_like(reset_buf), terminated)
    reset = torch.where(torch.logical_or(progress_buf >= max_episode_length-1, progress_buf - start_times >= rollout_length-1), torch.ones_like(reset_buf), terminated)
    return reset, terminated

@torch.jit.script
def compute_sdf(points1, points2):
    """Compute closest-point displacement vectors from points1 to points2 (batched)."""
    # type: (Tensor, Tensor) -> Tensor
    dis_mat = points1.unsqueeze(2) - points2.unsqueeze(1)
    dis_mat_lengths = torch.norm(dis_mat, dim=-1)
    min_length_indices = torch.argmin(dis_mat_lengths, dim=-1)
    B_indices, N_indices = torch.meshgrid(torch.arange(points1.shape[0]), torch.arange(points1.shape[1]), indexing='ij')
    min_dis_mat = dis_mat[B_indices, N_indices, min_length_indices].contiguous()
    return min_dis_mat