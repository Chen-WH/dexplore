"""Per-robot hand configurations for GRAB data conversion.

Each robot defines:
  - robot_name: name used for dex_retargeting RobotName enum and output filenames
  - urdf_subpath: path relative to dex_retargeting's robot hands directory
  - num_dof: number of DOFs (including 6 dummy wrist DOFs)
  - hand_joint_reorder: reordering of SMPL-X hand joints before retargeting (if needed)
  - sapien2isaac: mapping from SAPIEN joint order to Isaac Gym joint order
  - retarget_config: custom dex-retargeting config (for hands dex_retargeting does not know)
  - mirror: mirror the right-hand GRAB data into a left hand (see convert_grab.mirror_interaction_data)
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RobotConfig:
    robot_name: str
    urdf_subpath: str
    num_dof: int
    # Reorder hand joints before retargeting (None = no reorder)
    hand_joint_reorder: Optional[List[int]] = None
    # Isaac Gym DOF dictionary: joint_name -> isaac_index
    isaac_dof_dict: dict = field(default_factory=dict)
    # SAPIEN DOF ordering (after retarget2sapien mapping)
    sapien_dof_list: List[str] = field(default_factory=list)
    # dex-retargeting config path relative to data_processing/ (None = dex_retargeting default)
    retarget_config: Optional[str] = None
    # Retarget the mirrored right hand onto a left robot hand
    mirror: bool = False

    def get_sapien2isaac(self):
        """Compute SAPIEN -> Isaac Gym joint reordering."""
        if not self.isaac_dof_dict or not self.sapien_dof_list:
            return None
        isaac_dof_list = sorted(self.isaac_dof_dict.keys(), key=lambda k: self.isaac_dof_dict[k])
        return [self.sapien_dof_list.index(dof) for dof in isaac_dof_list]


ROBOT_CONFIGS = {
    "inspire": RobotConfig(
        robot_name="inspire",
        urdf_subpath="inspire_hand/inspire_hand_right.urdf",
        num_dof=18,
        # Inspire uses a hardcoded joint reorder after retargeting
        hand_joint_reorder=[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
        # Direct index mapping (no sapien2isaac dict needed)
        isaac_dof_dict={},
        sapien_dof_list=[],
    ),

    "leap": RobotConfig(
        robot_name="leap",
        urdf_subpath="leap_hand/leap_hand_right_glb.urdf",
        num_dof=22,
        isaac_dof_dict={
            "0": 7, "1": 6, "10": 20, "11": 21, "12": 10, "13": 11,
            "14": 12, "15": 13, "2": 8, "3": 9, "4": 15, "5": 14,
            "6": 16, "7": 17, "8": 19, "9": 18,
            "dummy_x_translation_joint": 0, "dummy_y_translation_joint": 1,
            "dummy_z_translation_joint": 2, "dummy_x_rotation_joint": 3,
            "dummy_y_rotation_joint": 4, "dummy_z_rotation_joint": 5,
        },
        sapien_dof_list=[
            "dummy_x_translation_joint", "dummy_y_translation_joint",
            "dummy_z_translation_joint", "dummy_x_rotation_joint",
            "dummy_y_rotation_joint", "dummy_z_rotation_joint",
            "1", "5", "9", "12", "0", "4", "8", "13",
            "2", "6", "10", "14", "3", "7", "11", "15",
        ],
    ),

    "shadow": RobotConfig(
        robot_name="shadow",
        urdf_subpath="shadow_hand/shadow_hand_right_glb.urdf",
        num_dof=30,
        isaac_dof_dict={
            "FFJ1": 11, "FFJ2": 10, "FFJ3": 9, "FFJ4": 8,
            "LFJ1": 16, "LFJ2": 15, "LFJ3": 14, "LFJ4": 13, "LFJ5": 12,
            "MFJ1": 20, "MFJ2": 19, "MFJ3": 18, "MFJ4": 17,
            "RFJ1": 24, "RFJ2": 23, "RFJ3": 22, "RFJ4": 21,
            "THJ1": 29, "THJ2": 28, "THJ3": 27, "THJ4": 26, "THJ5": 25,
            "WRJ1": 7, "WRJ2": 6,
            "dummy_x_translation_joint": 0, "dummy_y_translation_joint": 1,
            "dummy_z_translation_joint": 2, "dummy_x_rotation_joint": 3,
            "dummy_y_rotation_joint": 4, "dummy_z_rotation_joint": 5,
        },
        sapien_dof_list=[
            "dummy_x_translation_joint", "dummy_y_translation_joint",
            "dummy_z_translation_joint", "dummy_x_rotation_joint",
            "dummy_y_rotation_joint", "dummy_z_rotation_joint",
            "WRJ2", "WRJ1",
            "FFJ4", "MFJ4", "RFJ4", "LFJ5", "THJ5",
            "FFJ3", "MFJ3", "RFJ3", "LFJ4", "THJ4",
            "FFJ2", "MFJ2", "RFJ2", "LFJ3", "THJ3",
            "FFJ1", "MFJ1", "RFJ1", "LFJ2", "THJ2",
            "LFJ1", "THJ1",
        ],
    ),

    "allegro": RobotConfig(
        robot_name="allegro",
        urdf_subpath="allegro_hand/allegro_hand_right_glb.urdf",
        num_dof=22,
        isaac_dof_dict={
            "joint_0": 6, "joint_1": 7, "joint_2": 8, "joint_3": 9,
            "joint_4": 10, "joint_5": 11, "joint_6": 12, "joint_7": 13,
            "joint_8": 14, "joint_9": 15, "joint_10": 16, "joint_11": 17,
            "joint_12": 18, "joint_13": 19, "joint_14": 20, "joint_15": 21,
            "dummy_x_translation_joint": 0, "dummy_y_translation_joint": 1,
            "dummy_z_translation_joint": 2, "dummy_x_rotation_joint": 3,
            "dummy_y_rotation_joint": 4, "dummy_z_rotation_joint": 5,
        },
        sapien_dof_list=[
            "dummy_x_translation_joint", "dummy_y_translation_joint",
            "dummy_z_translation_joint", "dummy_x_rotation_joint",
            "dummy_y_rotation_joint", "dummy_z_rotation_joint",
            "joint_0", "joint_1", "joint_2", "joint_3",
            "joint_4", "joint_5", "joint_6", "joint_7",
            "joint_8", "joint_9", "joint_10", "joint_11",
            "joint_12", "joint_13", "joint_14", "joint_15",
        ],
    ),
}

_DUMMY_JOINTS = [
    "dummy_x_translation_joint", "dummy_y_translation_joint", "dummy_z_translation_joint",
    "dummy_x_rotation_joint", "dummy_y_rotation_joint", "dummy_z_rotation_joint",
]
# DexHand021 finger joints: fingers 1 (thumb) .. 5 (pinky), segments 1..4; l_f_joint3_1 is fixed
_DEXHAND021_FINGER_JOINTS = [f"l_f_joint{f}_{s}" for f in range(1, 6) for s in range(1, 5) if (f, s) != (3, 1)]

ROBOT_CONFIGS["dexhand021"] = RobotConfig(
    robot_name="dexhand021",
    urdf_subpath=None,  # taken from retarget_config
    num_dof=25,
    hand_joint_reorder=ROBOT_CONFIGS["inspire"].hand_joint_reorder,  # SMPL-X -> MANO order
    # Isaac Gym loads DOFs depth-first, finger by finger
    isaac_dof_dict={name: i for i, name in enumerate(_DUMMY_JOINTS + _DEXHAND021_FINGER_JOINTS)},
    # SAPIEN orders the finger joints breadth-first, segment by segment
    sapien_dof_list=_DUMMY_JOINTS + sorted(_DEXHAND021_FINGER_JOINTS, key=lambda n: (n[-1], n)),
    retarget_config="retarget_configs/dexhand021_left.yml",
    mirror=True,
)

# Inspire uses a direct hardcoded reorder instead of sapien2isaac dict
INSPIRE_RETARGET_REORDER = [0, 1, 2, 3, 4, 5, 7, 12, 8, 13, 10, 15, 9, 14, 6, 11, 16, 17]
