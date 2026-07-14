import time
import torch
from env.tasks.dexplore_inspire import Dexplore_Inspire

from isaacgym.torch_utils import *
from rl_games.algos_torch import torch_ext
import torch.nn as nn
from isaacgym import gymapi
from isaacgym import gymtorch
# from phc.learning.pnn import PNN
# from phc.learning.network_loader import load_mcp_mlp, load_pnn
from learning import dexplore_network_builder, dexplore_models
from torch.func import vmap
from functorch import make_functional
import os
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from torch_cluster import fps
from utils import torch_utils

def get_all_paths(dir_path):
    paths = []
    for root, dirs, files in os.walk(dir_path):
        for name in files:
            paths.append(os.path.join(root, name))
    return paths

class Dexplore_Distill(Dexplore_Inspire):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        self.action_buf = torch.zeros(
            (self.num_envs, 18), device=self.device, dtype=torch.float)
        self.mu_buf = torch.zeros(
            (self.num_envs, 18), device=self.device, dtype=torch.float)
        self.student_obs_buf = torch.zeros(
            (self.num_envs, 2102), device=self.device, dtype=torch.float)
        self.distill_keep_prob = 1.0  # wrist delta masking, updated by agent
        # Ablation flags (read from config, default=True to preserve original behavior)
        self.pointcloud_dr = self.cfg["env"].get("pointcloud_dr", True)
        self.obs_noise = self.cfg["env"].get("obs_noise", True)
        self.wrist_masking = self.cfg["env"].get("wrist_masking", True)
        # Loss-side ablation knobs (read by agent via task reference)
        self.aux_loss_coef = float(self.cfg["env"].get("aux_loss_coef", 0.1))
        self.kld_floor = float(self.cfg["env"].get("kld_floor", 0.001))
        self.kld_max = float(self.cfg["env"].get("kld_max", 0.01))
        self.kld_ramp_start = int(self.cfg["env"].get("kld_ramp_start", 0))
        self.kld_ramp_epochs = int(self.cfg["env"].get("kld_ramp_epochs", 3000))
        self.dagger_start_epoch = int(self.cfg["env"].get("dagger_start_epoch", 100))
        self.dagger_decay_epochs = int(self.cfg["env"].get("dagger_decay_epochs", 2000))
        self.zero_vae_noise = bool(self.cfg["env"].get("zero_vae_noise", False))
        self.mask_mimic_dims = bool(self.cfg["env"].get("mask_mimic_dims", False))
        self.contact_loss_alpha = float(self.cfg["env"].get("contact_loss_alpha", 0.0))
        self.vae_latent_dim = int(self.cfg["env"].get("vae_latent_dim", 64))
        self.models = []
        self.running_means = []
        self.running_vars = []
        obs_shape = cfg["env"]["numObs"]
        config = {
            'actions_num' : 18,
            'input_shape' : (obs_shape, ),
            'num_seqs' : cfg["env"]["numEnvs"] * 1,
            'value_size': 1,
        }
        print(config)
        network = dexplore_network_builder.DexploreBuilder()
        params = {
            "model": {
                "name": "dexplore"
            },
            "network": {
                "name": "dexplore",
                "separate": True,
                "space": {
                    "continuous": {
                        "mu_activation": "None",
                        "sigma_activation": "None",
                        "mu_init": {
                            "name": "default"
                        },
                        "sigma_init": {
                            "name": "const_initializer",
                            "val": -2.9
                        },
                        "fixed_sigma": True,
                        "learn_sigma": False
                    }
                },
                "mlp": {
                    "units": [1024, 1024, 1024, 512],
                    "activation": "relu",
                    "d2rl": False,
                    "initializer": {
                        "name": "default"
                    },
                    "regularizer": {
                        "name": "None"
                    }
                }
            }
        }
        network.load(params['network'])
        network = dexplore_models.ModelDexploreContinuous(network)
        model_path = cfg["env"]["modelZoo"]
        ck = torch_ext.load_checkpoint(model_path)
        model = network.build(config)
        model.to(self.device)
        model.load_state_dict(ck['model'])
        self.model = model
        running_mean, running_var = ck['running_mean_std']['running_mean'], ck['running_mean_std']['running_var']
        self.running_mean = running_mean
        self.running_var = running_var
        q_r = torch.tensor(
            [0.6710458873429609,
            0.6428864217629108,
            -0.2806992224997054,
            -0.24000502553288294],
            dtype=torch.float32
        )   # [x,y,z,w]
        t_r = torch.tensor([0.8897372841541342,
                            0.028012640860735998,
                            0.4977260755439195],
                        dtype=torch.float32)

        # precompute base height & Euler (xyz) for jitter
        # use torch_events or convert via math?
        # here we’ll convert once to numpy via scipy, then back:

        _base_euler = R.from_quat(q_r.numpy()).as_euler('xyz', degrees=True)
        self.base_height = t_r[2].item()
        self.base_euler  = torch.tensor(_base_euler, dtype=torch.float32)  # [pitch, roll, yaw]
        self.height_table = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self.origin_table = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self.frame_yaw = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)  # world-frame alignment DR
        camera_pose = [[-0.12, -0.78, 1.4], [0.0, 0.8, np.pi, 1]]
        camera_config_depth = {
            "name": "fix_camera_depth",
            "is_body_camera": False,  # set to True to have camera move with hand
            "actor_name": "humanoid",
            "image_size": [128, 128], # [2160, 3840],
            # "image_type": "rgb",
            'image_type': 'depth',
            "horizontal_fov": 80.96,
            # "camera_pose": [[0, 0.05, 0], [0.0, 0.0, 0.0, 1]],
            "camera_pose": camera_pose,
            # "near_plane": self.znear,
            "use_collision_geometry": False,
            # "attach_link_name": "middle_tip",
        }

        camera_config_rgb = {
            "name": "fix_camera_rgb",
            "is_body_camera": False,  # set to True to have camera move with hand
            "actor_name": "humanoid",
            "image_size": [128, 128], # [2160, 3840],
            # "image_type": "rgb",
            'image_type': 'rgb',
            "horizontal_fov": 80.96,
            # "camera_pose": [[0, 0.05, 0], [0.0, 0.0, 0.0, 1]],
            "camera_pose": camera_pose,
            # "near_plane": self.znear,
            "use_collision_geometry": False,
            # "attach_link_name": "middle_tip",
        }

        camera_config_seg = {
            "name": "fix_camera_seg",
            "is_body_camera": False,  # set to True to have camera move with hand
            "actor_name": "humanoid",
            "image_size": [128, 128], # [2160, 3840],
            # "image_type": "rgb",
            'image_type': 'seg',
            "horizontal_fov": 80.96,
            # "camera_pose": [[0, 0.05, 0], [0.0, 0.0, 0.0, 1]],
            "camera_pose": camera_pose,
            # "near_plane": self.znear,
            "use_collision_geometry": False,
            # "attach_link_name": "middle_tip",
        }
        
        # camera_config = OmegaConf.create(camera_config)
        self.camera_spec_dict = {camera_config_depth["name"]: camera_config_depth, camera_config_seg["name"]: camera_config_seg}
        self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.pointCloudDownsampleNum = 768
        self.camera_u, self.camera_v = torch.meshgrid(torch.arange(camera_config_depth['image_size'][1], dtype=torch.float32, device=self.device), torch.arange(camera_config_depth['image_size'][0], dtype=torch.float32, device=self.device))
        self.camera_u = self.camera_u.reshape(-1)
        self.camera_v = self.camera_v.reshape(-1)
        # self.camera_u = torch.arange(0, camera_config_depth['image_size'][0], device=self.device)
        # self.camera_v = torch.arange(0, camera_config_depth['image_size'][1], device=self.device)
        if self.camera_spec_dict:
            # tactile cameras created along with other cameras in create_camera_actors
            self.create_camera_actors()
        return

    def _build_table(self, env_id, env_ptr):
        """Override to allow table-hand contact during distillation (filter=1)."""
        col_group = env_id
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = self.table_data[env_id % self.num_motions]["table_pos"][0]
        default_pose.p.y = self.table_data[env_id % self.num_motions]["table_pos"][1]
        default_pose.p.z = self.table_data[env_id % self.num_motions]["table_pos"][2]

        default_pose.r.x = self.table_data[env_id % self.num_motions]["table_rot"][0]
        default_pose.r.y = self.table_data[env_id % self.num_motions]["table_rot"][1]
        default_pose.r.z = self.table_data[env_id % self.num_motions]["table_rot"][2]
        default_pose.r.w = self.table_data[env_id % self.num_motions]["table_rot"][3]

        table_handle = self.gym.create_actor(env_ptr, self._table_asset, default_pose, "table", col_group, 1, segmentation_id)

        if not self.cfg["headless"]:
            self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL,
                                        gymapi.Vec3(0.7, 0.6, 0.3))

        self._table_handles.append(table_handle)
        self.gym.set_actor_scale(env_ptr, table_handle, 1)

    def create_camera_actors(self):
        if hasattr(self, "camera_handles_list"):
            return
        self.camera_handles_list = []
        self.camera_tensors_list = []
        self.camera_vinv_list = []
        self.camera_proj_list = []
        camera_name = list(self.camera_spec_dict.keys())[0]
        for i in range(self.num_envs):
            env_ptr = self.envs[i]
            env_camera_handles = self.setup_env_cameras(env_ptr, self.camera_spec_dict)
            self.camera_handles_list.append(env_camera_handles)

            env_camera_tensors, env_camera_vinv, env_camera_proj = self.create_tensors_for_env_cameras(
                env_ptr, env_camera_handles, self.camera_spec_dict
            )
            self.camera_tensors_list.append(env_camera_tensors)
            self.camera_vinv_list.append(env_camera_vinv)
            self.camera_proj_list.append(env_camera_proj)
            origin = self.gym.get_env_origin(env_ptr)
            # print(origin)
            self.env_origin[i][0] = origin.x
            self.env_origin[i][1] = origin.y
            self.env_origin[i][2] = origin.z
            # if i == 0:
            #     cameraViewTransform = self.gym.get_camera_view_matrix(self.sim, env_ptr, env_camera_handles[camera_name])
            #     proj_matrix = self.gym.get_camera_proj_matrix(self.sim, env_ptr, env_camera_handles[camera_name])
            #     self.camera_spec_dict[camera_name]["camera_view_transform"] = cameraViewTransform
            #     self.camera_spec_dict[camera_name]["proj_matrix"] = proj_matrix

    def setup_env_cameras(self, env_ptr, camera_spec_dict):
        camera_handles = {}
        for name, camera_spec in camera_spec_dict.items():
            camera_properties = gymapi.CameraProperties()
            camera_properties.height = camera_spec["image_size"][0]
            camera_properties.width = camera_spec["image_size"][1]
            camera_properties.enable_tensors = True
            camera_properties.horizontal_fov = camera_spec["horizontal_fov"]
            # if "near_plane" in camera_spec:
            #     camera_properties.near_plane = camera_spec["near_plane

            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_properties)
            # print(env_ptr, camera_handle)
            camera_handles[name] = camera_handle

            if camera_spec["is_body_camera"]:
                actor_handle = self.gym.find_actor_handle(
                    env_ptr, camera_spec["actor_name"]
                )
                robot_body_handle = self.gym.find_actor_rigid_body_handle(
                    env_ptr, actor_handle, camera_spec["attach_link_name"]
                )
                print('handle', robot_body_handle)
                self.gym.attach_camera_to_body(
                    camera_handle,
                    env_ptr,
                    robot_body_handle,
                    gymapi.Transform(
                        gymapi.Vec3(*camera_spec["camera_pose"][0]),
                        gymapi.Quat(*camera_spec["camera_pose"][1]),
                    ),
                    gymapi.FOLLOW_TRANSFORM,
                )
            else:
                transform = gymapi.Transform(
                    gymapi.Vec3(*camera_spec["camera_pose"][0]),
                    gymapi.Quat(*camera_spec["camera_pose"][1]),
                )
                self.gym.set_camera_transform(camera_handle, env_ptr, transform)
        return camera_handles


    # def look_at_quat(self, cam_pos, target=torch.tensor([0.,0.,0.]), up=torch.tensor([0.,0.,1.])):
    #     """
    #     cam_pos: (3,) torch.Tensor, camera position in table-frame.
    #     target:  (3,) torch.Tensor, point to look at (here origin).
    #     up:      (3,) torch.Tensor, world up vector.
    #     Returns:
    #     q: (4,) torch.Tensor quaternion [x,y,z,w]
    #     """
    #     # 1) forward vector f = normalize(target - cam_pos)
    #     f = (target - cam_pos)
    #     f = f / f.norm()

    #     # 2) right vector r = normalize(cross(up, f))
    #     r = torch.cross(up, f)
    #     r = r / r.norm()

    #     # 3) true up u = cross(f, r)
    #     u = torch.cross(f, r)

    #     # 4) rotation matrix: columns are [r, u, f]
    #     R_mat = torch.stack([r, u, f], dim=1).numpy()  # shape (3,3)

    #     # 5) convert to quaternion via scipy (returns [x,y,z,w])
    #     q = R.from_matrix(R_mat).as_quat()
    #     return torch.from_numpy(q).float()
    

    def sample_camera_rel_pos(
        self,
        radius: float = 20.0,
        height: float = 20.0,
    ):
        """
        Returns:
        t_c : torch.Tensor (3,)   random translation in table–frame
        q_c : torch.Tensor (4,)   random quaternion [x,y,z,w]
        """
        az  = torch.rand(1).item() * 2*math.pi
        rho = 50 + (torch.rand(1).item() - 0.5)*radius
        x   = rho * math.cos(az)
        y   = rho * math.sin(az)
        z   = 50 + (torch.rand(1).item() - 0.5)*height

        t_c = torch.tensor([x, y, z], dtype=torch.float32, device=self.device) / 100

        return t_c
    
    def sample_camera_rel_target(
        self,
        radius: float = 5.0,
    ):
        """
        Returns:
        t_c : torch.Tensor (3,)   random translation in table–frame
        q_c : torch.Tensor (4,)   random quaternion [x,y,z,w]
        """
        az  = torch.rand(1).item() * 2*math.pi
        rho = (torch.rand(1).item())*radius
        x   = rho * math.cos(az)
        y   = rho * math.sin(az)
        z   = 0

        t_c = torch.tensor([x, y, z], dtype=torch.float32, device=self.device) / 100

        return t_c

    def reset_env_cameras(self, env_id, camera_spec_dict):
        env_ptr = self.envs[env_id]
        camera_handles = self.camera_handles_list[env_id]
        
        
        e_rel = self.sample_camera_rel_pos()
        camera_eye = e_rel + self.table_data[env_id % self.num_motions]["table_pos"]
        t_rel = self.sample_camera_rel_target()
        camera_lookat = t_rel + self.table_data[env_id % self.num_motions]["table_pos"]
        for name, camera_spec in camera_spec_dict.items():
            camera_handle = camera_handles[name]
            self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(*camera_eye), gymapi.Vec3(*camera_lookat))
            cam_vinv = torch.inverse(torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle))).to(self.device)
            cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle),device=self.device,)
            self.camera_vinv_list[env_id][name] = cam_vinv
            self.camera_proj_list[env_id][name] = cam_proj


    def create_tensors_for_env_cameras(
        self, env_ptr, env_camera_handles, camera_spec_dict
    ):
        env_camera_tensors = {}
        env_camera_vinv = {}
        env_camera_proj = {}
        for name in env_camera_handles:
            camera_handle = env_camera_handles[name]
            if camera_spec_dict[name]["image_type"] == "rgb":
                # obtain camera tensor
                camera_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR
                )
            elif camera_spec_dict[name]["image_type"] == "depth":
                # obtain camera tensor
                camera_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH
                )
            elif camera_spec_dict[name]["image_type"] == "seg":
                # obtain camera tensor
                # print(name, camera_spec_dict[name]["image_type"])
                camera_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_ptr, camera_handle, gymapi.IMAGE_SEGMENTATION
                )
            else:
                raise NotImplementedError(
                    f"Camera type {camera_spec_dict[name]['image_type']} not supported"
                )

            # wrap camera tensor in a pytorch tensor
            torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
            # store references to the tensor that gets updated when render_all_camera_sensors
            env_camera_tensors[name] = torch_camera_tensor
            view = self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle)
            view = torch.tensor(view, dtype=torch.float32, device=self.device).view(4, 4)
            # print(view)
            cam_vinv = torch.linalg.inv(view)
            cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle),device=self.device,)
            env_camera_vinv[name] = cam_vinv
            env_camera_proj[name] = cam_proj
        return env_camera_tensors, env_camera_vinv, env_camera_proj
    
    def get_camera_image_tensors_dict(self):
        # transforms and information must be communicated from the physics simulation into the graphics system
        if self.device != "cpu":
            self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        camera_image_tensors_dict = dict()
        camera_vinv_tensors_dict = dict()
        camera_proj_tensors_dict = dict()
        for name in self.camera_spec_dict:
            camera_spec = self.camera_spec_dict[name]
            if camera_spec["image_type"] == "rgb":
                num_channels = 3
                camera_images = torch.zeros(
                        (
                            self.num_envs,
                            camera_spec["image_size"][0],
                            camera_spec["image_size"][1],
                            num_channels,
                        ),
                        device=self.device,
                        dtype=torch.uint8,
                    )
                for id in np.arange(self.num_envs):
                    camera_images[id] = self.camera_tensors_list[id][name][
                        :, :, :num_channels
                    ].clone()
            elif camera_spec["image_type"] == "depth":
                num_channels = 1
                camera_images = torch.zeros(
                        (
                            self.num_envs,
                            camera_spec["image_size"][0],
                            camera_spec["image_size"][1],
                        ),
                        device=self.device,
                        dtype=torch.float,
                    )
                for id in np.arange(self.num_envs):
                    # Note that isaac gym returns negative depth
                    # (see the Isaac Gym graphics documentation on camera image types)
                    camera_images[id] = (
                        self.camera_tensors_list[id][name][:, :].clone() # * -1.0
                    )
                    camera_images[id][camera_images[id] == np.inf] = 0.0
            elif camera_spec["image_type"] == "seg":
                num_channels = 1
                camera_images = torch.zeros(
                        (
                            self.num_envs,
                            camera_spec["image_size"][0],
                            camera_spec["image_size"][1],
                        ),
                        device=self.device,
                        dtype=torch.int32,
                    )
                for id in np.arange(self.num_envs):
                    # Note that isaac gym returns negative depth
                    # (see the Isaac Gym graphics documentation on camera image types)
                    camera_images[id] = self.camera_tensors_list[id][name][
                        :, :
                    ].clone()
            else:
                print(f'Image type {camera_spec["image_type"]} not supported!')
            camera_image_tensors_dict[name] = camera_images
            camera_vinv = torch.zeros(
                        (
                            self.num_envs,
                            4,
                            4,
                        ),
                        device=self.device,
                        dtype=torch.float,
                    )
            camera_proj = torch.zeros(
                        (
                            self.num_envs,
                            4,
                            4,
                        ),
                        device=self.device,
                        dtype=torch.float,
                    )
            for id in np.arange(self.num_envs):
                # Note that isaac gym returns negative depth
                # (see the Isaac Gym graphics documentation on camera image types)
                camera_vinv[id] = self.camera_vinv_list[id][name].clone()
                camera_proj[id] = self.camera_proj_list[id][name].clone()
            camera_vinv_tensors_dict[name] = camera_vinv
            camera_proj_tensors_dict[name] = camera_proj
        return camera_image_tensors_dict, camera_vinv_tensors_dict, camera_proj_tensors_dict

    # def _get_pointcloud(self, camera_spec, depth):
    #             # Parse camera parameters
    #     # width, height = (
    #     #     camera_spec["image_size"][0],
    #     #     camera_spec["image_size"][1],
    #     # )
    #     # # camera_config_2 = {
    #     # #     "name": "fix_camera",
    #     # #     "is_body_camera": False,  # set to True to have camera move with hand
    #     # #     "actor_name": "humanoid",
    #     # #     "image_size": [256, 256],
    #     # #     # "image_type": "rgb",
    #     # #     'image_type': 'depth',
    #     # #     "horizontal_fov": 60,
    #     # #     # "camera_pose": [[0, 0.05, 0], [0.0, 0.0, 0.0, 1]],
    #     # #     "camera_pose": [[0, -1.0, 1.0], [0.0, 0.0, np.pi, 1]],
    #     # #     # "near_plane": self.znear,
    #     # #     "use_collision_geometry": True,
    #     # #     # "attach_link_name": "middle_tip",
    #     # # }
    #     # # Get the camera properties
    #     # fov = camera_spec["horizontal_fov"]  # Horizontal FOV in degrees

    #     # # Convert horizontal FOV to radians
    #     # fov_rad = math.radians(fov)

    #     # # Calculate focal length (in pixels)
    #     # focal_length = width / (2 * math.tan(fov_rad / 2))

    #     # # Calculate the horizontal aperture (field of view in radians)
    #     # horiz_aperture = fov_rad  # Horizontal aperture in radians
    #     # # vert_aperture might be buggy?
    #     # # vert_aperture = camera_params["cameraAperture"][1]
    #     # vert_aperture = height / width * horiz_aperture
    #     # focal_y = height * focal_length / vert_aperture
    #     # focal_x = width * focal_length / horiz_aperture
    #     # center_y = height * 0.5
    #     # center_x = width * 0.5

    #     # fx, fy, cx, cy = focal_x, focal_y, center_x, center_y

    #     # # Compute point cloud in camera frame
    #     # pointcloud_camera = depth2points(depth, fx, fy, cx, cy, rgb=None)
    #     # trans = np.eye(4)
    #     # trans[1, 1] *= -1
    #     # trans[2, 2] *= -1

    #     # # https://towardsdatascience.com/7071b72fb8ec
    #     # # The extrinsic matrix is a transformation matrix from the world coordinate
    #     # # system to the camera coordinate system
    #     # cam_world_to_local = np.array(camera_spec["camera_view_transform"]).reshape(4, 4)
    #     # cam_world_to_local = np.matmul(cam_world_to_local, trans)
    #     # cam_local_to_world = np.linalg.inv(cam_world_to_local)

    #     # pointcloud_camera_homo = np.pad(
    #     #     pointcloud_camera["xyz"][pointcloud_camera["index"]],
    #     #     ((0, 0), (0, 1)),
    #     #     constant_values=1,
    #     # )
    #     # pointcloud_camera_to_world = np.matmul(pointcloud_camera_homo, cam_local_to_world)[:, :3]
    #     # return pointcloud_camera, pointcloud_camera_to_world

    #     vinv = np.linalg.inv(np.matrix(camera_spec["camera_view_transform"]))

    #     # Get the camera projection matrix and get the necessary scaling
    #     # coefficients for deprojection
    #     proj = camera_spec["proj_matrix"]
    #     fu = 2/proj[0, 0]
    #     fv = 2/proj[1, 1]

    #     # Ignore any points which originate from ground plane or empty space
    #     depth_buffer[seg_buffer == 0] = -10001
    #     # print(depth_buffer.shape)
    #     cam_width = camera_spec_dict[name]["image_size"][0]
    #     cam_height = camera_spec_dict[name]["image_size"][1]
    #     centerU = cam_width/2
    #     centerV = cam_height/2
    #     for i in range(cam_width):
    #         for j in range(cam_height):
    #             if depth_buffer[j, i] < -10000:
    #                 continue
    #             if seg_buffer[j, i] > 0:
    #                 u = -(i-centerU)/(cam_width)  # image-space coordinate
    #                 v = (j-centerV)/(cam_height)  # image-space coordinate
    #                 d = depth_buffer[j, i]  # depth buffer value
    #                 X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
    #                 p2 = X2*vinv  # Inverse camera view to get world coordinates
    #                 points.append([p2[0, 2], p2[0, 0], p2[0, 1]])
    #                 color.append(c)
    #     # # use pptk to visualize the 3d point cloud created above
    #     # v = pptk.viewer(points, color)
    #     # v.color_map(color_map)
    #     # # Sets a similar view to the gym viewer in the PPTK viewer
    #     # v.set(lookat=[0, 0, 0], r=5, theta=0.4, phi=0.707)
    #     # print("Point Cloud Complete")
    #     return points

    def depth_image_to_point_cloud_GPU(self, depth_buffer, seg_buffer, camera_view_matrix_inv, camera_proj_matrix, v, u, width:float, height:float, depth_bar:float, device:torch.device):
        vinv = camera_view_matrix_inv
        proj = camera_proj_matrix
        fu = 2/proj[:, 0:1, 0]
        fv = 2/proj[:, 1:2, 1]

        centerU = width/2
        centerV = height/2

        Z = depth_buffer.view(self.num_envs, -1)
        # print((-(u-centerU)/width).shape, Z.shape, fu.shape)
        X = -(u-centerU)/width * Z * fu
        Y = (v-centerV)/height * Z * fv

        Z[seg_buffer.view(self.num_envs, -1) == 0] = -depth_bar - 1
        Z = Z.view(self.num_envs, -1)
        valid = Z > -depth_bar
        X = X.view(self.num_envs, -1)
        Y = Y.view(self.num_envs, -1)
        # print(valid)
        # print(X.shape)
        position = torch.stack((X, Y, Z, torch.ones(X.shape, device=device)), dim=-1) #[:, valid]
        # print(position.shape)
         # position = position.permute(0, 2, 1)
        # print(position.shape, vinv.shape)
        position = position@vinv
        position = position[:, :, :3] - self.env_origin.unsqueeze(1)
        valid_points = [position[i, valid[i]][:, :] for i in range(self.num_envs)]
        # for points in valid_points:
        #     print(points.mean(dim=0))
        # points = position[:, :, 0:3]

        return valid_points

    def rand_row(self, tensor, dim_needed):  
        row_total = tensor.shape[0]
        return tensor[torch.randint(low=0, high=row_total, size=(dim_needed,)),:]

    def sample_points(self, points, sample_num=1000, sample_mathed='furthest'):
        if points.shape[1] > 0:
            eff_points = points[points[:, 2]>0.04]
            if eff_points.shape[0] < sample_num :
                eff_points = points
            # if sample_mathed == 'random':
            sampled_points = self.rand_row(eff_points, sample_num)
            # elif sample_mathed == 'furthest':
            #     sampled_points_id = pointnet2_utils.furthest_point_sample(eff_points.reshape(1, *eff_points.shape), sample_num)
            #     sampled_points = eff_points.index_select(0, sampled_points_id[0].long())
            return sampled_points
        else:
            return points


    def post_physics_step(self):
        self.progress_buf += 1
        # print(torch.where((self.max_episode_length[self.data_id] - self.progress_buf + 1) < 0))
        # assert ((self.max_episode_length[self.data_id] - self.progress_buf + 1) >= 0).all() == True
        self._refresh_sim_tensors()
        env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        #extra calc of self._curr_hoi_obs_buf, for correct calculate of imitation reward
        self._compute_hoi_observations(env_ids)
        self._compute_observations(env_ids)
        self._compute_observations_distill(env_ids)
        self._compute_reward(self.actions)
        self._compute_reset()
        
        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def _reset_envs(self, env_ids):
        # print(env_ids)
        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            for i in env_ids:
                self.reset_env_cameras(i.item(), self.camera_spec_dict)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
            self.origin_table[env_ids] = self.curr_obj_points[..., :2].mean(dim=1)
            self.height_table[env_ids] = self.curr_obj_points[..., 2].min(dim=-1)[0]
            if self.obs_noise:
                n = env_ids.shape[0]
                az  = torch.rand(n, device=self.device) * 2*math.pi
                rho = torch.rand(n, device=self.device) * 0.1
                xy  = torch.stack([rho * torch.cos(az), rho * torch.sin(az)], dim=-1)
                z   = (torch.rand(n, device=self.device) - 0.5) * 0.05
                self.origin_table[env_ids] = self.origin_table[env_ids] + xy
                self.height_table[env_ids] = self.height_table[env_ids] + z
                # World-frame alignment DR: ±~3° yaw per episode (applied consistently to
                # hand positions and point cloud after origin subtraction).
                self.frame_yaw[env_ids] = (torch.rand(n, device=self.device) - 0.5) * 0.1
            else:
                self.frame_yaw[env_ids] = 0.0
            self._compute_observations_distill(env_ids)
        return
        
    def _compute_observations_distill(self, env_ids=None, delta_t=16):

        cameras, camera_vinv, camera_proj = self.get_camera_image_tensors_dict()
        camera_names = list(self.camera_spec_dict.keys())
        name = "fix_camera_depth"
        cam_width = self.camera_spec_dict[name]["image_size"][0]
        cam_height = self.camera_spec_dict[name]["image_size"][1]
        # print(self.camera_u, self.camera_v)
        
        # depth_image_to_point_cloud_GPU(self, depth_buffer, seg_buffer, camera_view_matrix_inv, camera_proj_matrix, u, v, width:float, height:float, depth_bar:float, device:torch.device):

        points = self.depth_image_to_point_cloud_GPU(cameras["fix_camera_depth"], cameras["fix_camera_seg"], camera_vinv[name], camera_proj[name], self.camera_u, self.camera_v, cam_width, cam_height, 10, self.device)
        # # print(points[0][:, 0].max(), points[0][:, 1].max(), points[0][:, 2].max())
        # selected_points = self.sample_points(points, sample_num=self.pointCloudDownsampleNum, sample_mathed='random')
        # print(points.shape, selected_points.shape)
        # obs_images = cameras[camera_names[0]]
        # print(obs_images.max(), obs_images.min())
        # pointcloud_camera_to_world = self._get_pointcloud(self.camera_spec_dict[camera_names[0]], obs_images[0].cpu().numpy())
        # print(pointcloud_camera_to_world.shape)
        
        # points = self.depth_image_to_point_cloud_GPU(self.camera_tensors[i], self.camera_view_matrixs[i], self.camera_proj_matrixs[i], self.camera_u2, self.camera_v2, self.camera_props.width, self.camera_props.height, 10, self.device)
        # selected_points = self.sample_points(points, sample_num=self.pointCloudDownsampleNum, sample_mathed='random')
        # print(pointcloud_camera_to_world)
        # # Add coordicates
        visualize = False
        env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        ts = self.progress_buf.clone()
        self._curr_ref_obs = self.hoi_data[self.data_id[env_ids], ts].clone() 
        next_ts = torch.clamp(ts + delta_t, max=self.max_episode_length[self.data_id[env_ids]]-1)
        ref_obs = self.hoi_data[self.data_id[env_ids], next_ts].clone()
        local_obs, global_obs = self._compute_humanoid_obs_pro(env_ids, ref_obs)
        global_obs = global_obs.view(env_ids.shape[0], -1, 3)
        
        # if visualize:
        #     _, curr_obj_points = self._compute_task_obs(ref_obs=self.hoi_data[self.data_id, 0].clone())
        #     visualizer3d = Visualizer3D(
        #         view_config="o3d_view_conf.json", overwrite_view_config=True, non_blocking=True
        #     )

        #     visualizer3d.create_coordinate_frame(size=0.2)
        #     # camera_frame = True
        #     # if camera_frame:
        #     #     # Add point cloud from camera to world
        #     #     visualizer3d.create_pointcloud(
        #     #         pointcloud_camera["xyz"][pointcloud_camera["index"]],
        #     #         rgb=None,
        #     #         name="pointcloud_camera",
        #     #         radius=0.1,
        #     #     )
        #     # else:
        #     #     # Add point cloud from camera to world
        #     visualizer3d.create_pointcloud(
        #         points[0].cpu().numpy(),
        #         color=[0.5, 0.5, 0.5],
        #         name="pointcloud_camera_to_world",
        #         radius=0.1,
        #     )

        #     visualizer3d.create_pointcloud(
        #         curr_obj_points[0].cpu().numpy(),
        #         color=[0, 1.0, 0],
        #         name="oracle",
        #         radius=0.1,
        #     )
            
        #     visualizer3d.create_pointcloud(
        #         global_obs[0].cpu().numpy(),
        #         color=[0, 0, 1.0],
        #         name="hand",
        #         radius=1.0,
        #     )
        #     # Visualize the scene
        #     visualizer3d.draw_geometries()

        # Collate raw point cloud (no hand kp). Force exactly N_PC=512 points so that
        # combined with 6 hand keypoints we always get 518 points.
        points_padded, valid = self.collate(points)
        N_PC = 512
        B, P_cur, _ = points_padded.shape
        if P_cur < N_PC:
            # Upsample by repeating existing points
            extra = N_PC - P_cur
            idx = torch.randint(0, P_cur, (B, extra), device=self.device)
            extra_pts = torch.gather(points_padded, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            extra_valid = torch.gather(valid, 1, idx)
            points_padded = torch.cat([points_padded, extra_pts], dim=1)
            valid = torch.cat([valid, extra_valid], dim=1)
        elif P_cur > N_PC:
            points_padded = points_padded[:, :N_PC, :]
            valid = valid[:, :N_PC]

        # Point cloud augmentation for sim-to-real robustness (applied to point cloud only,
        # not hand keypoints — hand kp are deterministic references)
        if self.pointcloud_dr:
            # Gaussian noise (σ=0.01) on valid points
            noise = 0.01 * torch.randn_like(points_padded)
            points_padded = points_padded + noise * valid.unsqueeze(-1)
            # Point dropout: zero out 15% of points
            dropout = (torch.rand(valid.shape, device=self.device) > 0.15).float()
            valid = valid * dropout
            # Random outliers: replace 5% of points with random positions
            outlier = torch.rand(valid.shape, device=self.device) < 0.05
            rand_pts = torch.rand_like(points_padded) * 0.6 - 0.3
            points_padded = torch.where(outlier.unsqueeze(-1), rand_pts, points_padded)

        # Prepend hand keypoint validities (value 2.0 to mark them as semantic flag,
        # distinguishing from point cloud which uses 1.0/0.0)
        valid = torch.cat([torch.ones([global_obs.shape[0], global_obs.shape[1]], device=self.device)+1, valid], dim=-1)
        # Concat hand keypoints (6) onto point cloud (512) → [B, 518, 3]
        global_obs = torch.cat([global_obs, points_padded], dim=1)

        # Apply coordinate frame transformation to all 518 points (hand + point cloud)
        global_obs[:, :, :2] = global_obs[:, :, :2] - self.origin_table.unsqueeze(1)
        global_obs[:, :, 2] = global_obs[:, :, 2] - self.height_table.unsqueeze(1)

        # Apply per-env yaw DR around the (already-translated) origin, consistently to
        # both hand points and point cloud so their relative geometry is preserved.
        cos_y = torch.cos(self.frame_yaw).unsqueeze(-1)  # (B, 1)
        sin_y = torch.sin(self.frame_yaw).unsqueeze(-1)
        x = global_obs[:, :, 0]
        y = global_obs[:, :, 1]
        global_obs[:, :, 0] = cos_y * x - sin_y * y
        global_obs[:, :, 1] = sin_y * x + cos_y * y
        
        # if visualize:
        #     _, curr_obj_points = self._compute_task_obs(ref_obs=self.hoi_data[self.data_id, 0].clone())
        #     visualizer3d = Visualizer3D(
        #         view_config="o3d_view_conf.json", overwrite_view_config=True, non_blocking=True
        #     )

        #     visualizer3d.create_coordinate_frame(size=0.2)
        #     # camera_frame = True
        #     # if camera_frame:
        #     #     # Add point cloud from camera to world
        #     #     visualizer3d.create_pointcloud(
        #     #         pointcloud_camera["xyz"][pointcloud_camera["index"]],
        #     #         rgb=None,
        #     #         name="pointcloud_camera",
        #     #         radius=0.1,
        #     #     )
        #     # else:
        #     #     # Add point cloud from camera to world
        #     visualizer3d.create_pointcloud(
        #         global_obs[5, 6:].cpu().numpy(),
        #         color=[0.5, 0.5, 0.5],
        #         name="pointcloud_camera_to_world",
        #         radius=0.1,
        #     )
        #     curr_obj_points[:, :, :2] = curr_obj_points[:, :, :2] - self.origin_table.unsqueeze(1)
        #     curr_obj_points[:, :, 2] = curr_obj_points[:, :, 2] - self.height_table.unsqueeze(1)
        #     visualizer3d.create_pointcloud(
        #         curr_obj_points[5].cpu().numpy(),
        #         color=[0, 1.0, 0],
        #         name="oracle",
        #         radius=0.1,
        #     )
            
        #     visualizer3d.create_pointcloud(
        #         global_obs[5, :6].cpu().numpy(),
        #         color=[0, 0, 1.0],
        #         name="hand",
        #         radius=10.0,
        #     )
        #     # Visualize the scene
        #     visualizer3d.draw_geometries()
        # Construct student observation: [local_obs (30D) | global_obs (1554D = 518*3) | valid (518D)] = 2102D
        # global_obs is the 6 hand keypoints + 512 point cloud = 518 points (matches cluster training)
        self.student_obs_buf = torch.cat([local_obs, global_obs.view(self.num_envs, 518 * 3), valid], dim=-1)


    def collate(self, batch, N_lo=256, N_hi=512, training=True):
        pcs, masks = [], []
        EMPTY_TOKEN = torch.zeros(1, 3).to(self.device)
        for pc in batch:                     # pc : (P,3) float32
            P = pc.shape[0]

            # 0-point case -----------------------------------------------------
            if P == 0:
                pc = EMPTY_TOKEN.clone()
                mask = torch.tensor([0])   # invalid → will be ignored
            else:
                mask = torch.ones(min(P, N_hi), dtype=torch.bool)

                # up-sample ----------------------------------------------------
                if P < N_lo:
                    idx = torch.randint(0, P, (N_lo-P,))
                    pc  = torch.cat([pc, pc[idx] + 1e-3*torch.randn_like(pc[idx])])

                # down-sample --------------------------------------------------
                elif P > N_hi:
                    sel = fps(pc, ratio=N_hi/P)
                    pc  = pc[sel]

            if training:
                pc += 1e-3*torch.randn_like(pc)  # tiny noise aug

            pcs.append(pc)        # ragged list
            masks.append(mask)

        # Ragged → padded tensor + mask
        N_max = max(p.size(0) for p in pcs)
        padded = torch.stack([torch.nn.functional.pad(p, (0,0,0,N_max-p.size(0)))
                            for p in pcs])                      # (B,N_max,3)
        valid  = torch.stack([torch.nn.functional.pad(m, (0,N_max-m.size(0)),
                                                    value=0) for m in masks]).to(padded.device)
        return padded, valid

    def calculate_intersection_ratio(self, obj_points, pointcloud_camera_to_world, distance_threshold=0.02):
        """
        Calculate the ratio of points from `obj_points` that are within a certain distance
        of any point in `pointcloud_camera_to_world`.
        
        Parameters:
        - obj_points: A tensor of shape (N, 3), representing the first point cloud (object points).
        - pointcloud_camera_to_world: A tensor of shape (M, 3), representing the second point cloud (camera-to-world points).
        - distance_threshold: The maximum distance to consider a point as intersected (default is 0.01).
        
        Returns:
        - intersection_ratio: The ratio of intersected points.
        """
        # Ensure obj_points and pointcloud_camera_to_world are tensors
        obj_points = obj_points.float()
        pointcloud_camera_to_world = pointcloud_camera_to_world.float()
        
        # Expand the dimensions of the point clouds to compute pairwise distances
        obj_points_expanded = obj_points.unsqueeze(1)  # Shape: (N, 1, 3)
        pointcloud_expanded = pointcloud_camera_to_world.unsqueeze(0)  # Shape: (1, M, 3)
        
        # Compute pairwise squared Euclidean distances
        distances = torch.norm(obj_points_expanded - pointcloud_expanded, dim=2)  # Shape: (N, M)
        
        # Find the minimum distance for each point in `obj_points` to any point in `pointcloud_camera_to_world`
        min_distances, _ = torch.min(distances, dim=1)  # Shape: (N,)
        # print(min_distances)
        
        # Count how many points are within the threshold distance
        intersected_points = (min_distances <= distance_threshold).sum().item()
        
        # Calculate the ratio of intersected points
        intersection_ratio = intersected_points / obj_points.shape[0]
        
        return intersection_ratio
    
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        # self._num_actions = self.num_prim
        return

    # def get_task_obs_size_detail(self):
    #     task_obs_detail = super().get_task_obs_size_detail()
    #     task_obs_detail['num_prim'] = self.num_prim
    #     return task_obs_detail


    def step(self, weights):

        self.pre_physics_step(weights)

        # step physics and render each frame
        self._physics_step()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        with torch.no_grad():
            curr_obs = ((self.obs_buf - self.running_mean.float().to(self.device)) / torch.sqrt(self.running_var.float().to(self.device) + 1e-05))
            curr_obs = torch.clamp(curr_obs, min=-5.0, max=5.0)
            self.model.eval()
            input_dict = {
                'is_train': False,
                'prev_actions': None, 
                'obs' : curr_obs,
                'rnn_states' : None
            }
            res_dict = self.model(input_dict)
            teacher_action = torch.clamp(res_dict['actions'], min=-1.0, max=1.0)
            mu = res_dict['mus']
            self.action_buf = teacher_action     
            self.mu_buf = mu
            # print('env', self.obs_buf.shape)
        

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)


    def reset(self, env_ids=None):
        super().reset(env_ids=env_ids)
        with torch.no_grad():
            curr_obs = ((self.obs_buf - self.running_mean.float().to(self.device)) / torch.sqrt(self.running_var.float().to(self.device) + 1e-05))
            curr_obs = torch.clamp(curr_obs, min=-5.0, max=5.0)
            self.model.eval()
            input_dict = {
                'is_train': False,
                'prev_actions': None, 
                'obs' : curr_obs,
                'rnn_states' : None
            }
            res_dict = self.model(input_dict)
            # print(res_dict)
            teacher_action = torch.clamp(res_dict['actions'], min=-1.0, max=1.0)
            mu = res_dict['mus']
            self.action_buf = teacher_action     
            self.mu_buf = mu    

        return
    
    def render(self, sync_frame_time=False, t=0):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
            
            if self.save_images:
                env_ids = 0
                if self.play_dataset:
                    frame_id = t
                else:
                    frame_id = self.progress_buf[env_ids]

                dataname = self.motion_file[-1][len('dexplore/data/motions/'):]
                os.makedirs("dexplore/data/depth/" + dataname, exist_ok=True)
                os.makedirs("dexplore/data/color/" + dataname, exist_ok=True)
                rgb_filename = "dexplore/data/images/" + dataname + "/rgb_env%d_frame%05d.png" % (env_ids, frame_id)
                os.makedirs("dexplore/data/images/" + dataname, exist_ok=True)
                self.gym.write_viewer_image_to_file(self.viewer,rgb_filename)
                rgb_filename = "dexplore/data/depth/" + dataname + "/hand%05d.png" % (t)
                self.gym.write_camera_image_to_file(self.sim, self.envs[0], 0, gymapi.IMAGE_DEPTH, rgb_filename)

                # rgb_filename = "dexplore/data/depth/" + dataname + "/fix%05d.png" % (t)
                # self.gym.write_camera_image_to_file(self.sim, self.envs[0], 1, gymapi.IMAGE_DEPTH, rgb_filename)

                rgb_filename = "dexplore/data/color/" + dataname + "/hand%05d.png" % (t)
                self.gym.write_camera_image_to_file(self.sim, self.envs[0], 2, gymapi.IMAGE_COLOR, rgb_filename)

                # rgb_filename = "dexplore/data/color/" + dataname + "/fix%05d.png" % (t)
                # self.gym.write_camera_image_to_file(self.sim, self.envs[0], 3, gymapi.IMAGE_COLOR, rgb_filename)
        return
    
    def _compute_humanoid_obs_pro(self, env_ids=None, ref_obs=None):
        keep_prob = self.distill_keep_prob if self.wrist_masking else 1.0
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            contact_forces = self._contact_forces
            dof_pos = self._dof_pos
            local_obs, global_obs = compute_humanoid_observations_pro(body_pos, body_rot, dof_pos,
                                                contact_forces, self._contact_body_ids, ref_obs, self._key_body_ids,
                                                self.progress_buf < self.start_contact_idx[self.data_id],
                                                keep_prob, self.obs_noise)

        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            contact_forces = self._contact_forces[env_ids]
            dof_pos = self._dof_pos[env_ids]
            local_obs, global_obs = compute_humanoid_observations_pro(body_pos, body_rot, dof_pos,
                                                contact_forces, self._contact_body_ids, ref_obs, self._key_body_ids,
                                                self.progress_buf[env_ids] < self.start_contact_idx[self.data_id[env_ids]],
                                                keep_prob, self.obs_noise)

        return local_obs, global_obs
    
def compute_humanoid_observations_pro(body_pos, body_rot, dof_pos, contact_forces, contact_body_ids, ref_obs, key_body_ids, at_begin, keep_prob=1.0, obs_noise=True):
    root_pos = body_pos[:, 7, :]
    root_rot = body_rot[:, 7, :]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    len_keypos = len(key_body_ids)
    heading_rot_expand = heading_rot.unsqueeze(-2)


    heading_rot_expand = heading_rot_expand.repeat((1, len_keypos, 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                               heading_rot_expand.shape[2])


    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos[:, key_body_ids, :] - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    body_contact_buf = contact_forces[:, contact_body_ids, :].clone()
    contact = torch.any(torch.abs(body_contact_buf) > 0.1, dim=-1).float()
    ref_body_contact = torch.any((ref_obs[:,119+len_keypos*3+1:119+len_keypos*3+1+16][:, [3, 6, 9, 12, 15]] + 1) / 2 > 0.5, dim=-1)
    todo = torch.logical_or(at_begin, ref_body_contact).unsqueeze(1)

    root_pos = body_pos[:, 6, :]
    root_rot = body_rot[:, 6, :]

    # Wrist reference delta: translation (3D) + rotation (6D) in local frame
    ref_root_pos = ref_obs[:, 4:7]
    ref_root_rot = ref_obs[:, 0:4]
    wrist_trans_delta = ref_root_pos - root_pos
    wrist_trans_delta_local = quat_rotate(heading_rot, wrist_trans_delta)  # 3D
    rot_delta = torch_utils.quat_mul(torch_utils.quat_inverse(root_rot), ref_root_rot)
    rot_delta_6d = torch_utils.quat_to_tan_norm(rot_delta)  # 6D

    # Apply masking curriculum: zero out wrist delta with probability (1 - keep_prob)
    wrist_mask = (torch.rand(root_pos.shape[0], 1, device=root_pos.device) < keep_prob).float()
    wrist_trans_delta_local = wrist_trans_delta_local * wrist_mask
    rot_delta_6d = rot_delta_6d * wrist_mask

    local_obs = torch.cat((dof_pos[:, 3:], contact, todo, wrist_trans_delta_local, rot_delta_6d), dim=-1)
    # local_obs: 15 + 5 + 1 + 3 + 6 = 30D
    global_obs = torch.cat((root_pos, body_pos[:, key_body_ids, :][:, [3, 6, 9, 12, 15], :].view(-1, 5*3)), dim=-1)
    if obs_noise:
        global_obs = global_obs + 1e-2*torch.randn_like(global_obs)
    return local_obs, global_obs