"""SMPL-X body model constants for data processing.

Defines bone orderings, body segments, and vertex segmentation used
to convert GRAB dataset sequences into simulation-ready formats.
"""
import json
import os
import urllib.request

SMPLX_VERT_SEGMENTATION_URL = (
    "https://raw.githubusercontent.com/Meshcapade/wiki/main/"
    "assets/SMPL_body_segmentation/smplx/smplx_vert_segmentation.json"
)

SMPLH_BONE_ORDER_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee", "Spine",
    "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe", "Neck",
    "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    "L_Index1", "L_Index2", "L_Index3",
    "L_Middle1", "L_Middle2", "L_Middle3",
    "L_Pinky1", "L_Pinky2", "L_Pinky3",
    "L_Ring1", "L_Ring2", "L_Ring3",
    "L_Thumb1", "L_Thumb2", "L_Thumb3",
    "R_Index1", "R_Index2", "R_Index3",
    "R_Middle1", "R_Middle2", "R_Middle3",
    "R_Pinky1", "R_Pinky2", "R_Pinky3",
    "R_Ring1", "R_Ring2", "R_Ring3",
    "R_Thumb1", "R_Thumb2", "R_Thumb3",
]

# Maps SMPLH joint names to SMPL-X mesh segment names (for vertex segmentation lookup)
SMPLH_SEGMENT = {
    "Pelvis": "hips", "L_Hip": "leftUpLeg", "R_Hip": "rightUpLeg",
    "Torso": "spine", "L_Knee": "leftLeg", "R_Knee": "rightLeg",
    "Spine": "spine1", "L_Ankle": "leftFoot", "R_Ankle": "rightFoot",
    "Chest": "spine2", "L_Toe": "leftToeBase", "R_Toe": "rightToeBase",
    "Neck": "neck", "L_Thorax": "leftShoulder", "R_Thorax": "rightShoulder",
    "Head": "head", "L_Shoulder": "leftArm", "R_Shoulder": "rightArm",
    "L_Elbow": "leftForeArm", "R_Elbow": "rightForeArm",
    "L_Wrist": "leftHand", "R_Wrist": "rightHand",
    "L_Index1": "leftHandIndex1", "L_Index2": "leftHandIndex1", "L_Index3": "leftHandIndex1",
    "L_Middle1": "leftHandIndex1", "L_Middle2": "leftHandIndex1", "L_Middle3": "leftHandIndex1",
    "L_Pinky1": "leftHandIndex1", "L_Pinky2": "leftHandIndex1", "L_Pinky3": "leftHandIndex1",
    "L_Ring1": "leftHandIndex1", "L_Ring2": "leftHandIndex1", "L_Ring3": "leftHandIndex1",
    "L_Thumb1": "leftHand", "L_Thumb2": "leftHand", "L_Thumb3": "leftHand",
    "R_Index1": "rightHandIndex1", "R_Index2": "rightHandIndex1", "R_Index3": "rightHandIndex1",
    "R_Middle1": "rightHandIndex1", "R_Middle2": "rightHandIndex1", "R_Middle3": "rightHandIndex1",
    "R_Pinky1": "rightHandIndex1", "R_Pinky2": "rightHandIndex1", "R_Pinky3": "rightHandIndex1",
    "R_Ring1": "rightHandIndex1", "R_Ring2": "rightHandIndex1", "R_Ring3": "rightHandIndex1",
    "R_Thumb1": "rightHand", "R_Thumb2": "rightHand", "R_Thumb3": "rightHand",
}

# SMPL joint reordering: SMPL-X joint indices -> MuJoCo skeleton ordering
SMPL_2_MUJOCO = [
    0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
    14, 17, 19, 21,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
]

# Joint indices for extracting hand keypoints from SMPL-X output joints
SMPL_2_MUJOCO_JOINTS = [
    20, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    21, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
]


def load_smplx_vert_segmentation():
    """Load SMPL-X vertex segmentation from JSON.

    The segmentation file is published by Meshcapade alongside the SMPL-X model
    and is not redistributed with this repo (see SMPL-X license). On first call,
    it is downloaded from the upstream URL and cached locally.
    """
    json_path = os.path.join(os.path.dirname(__file__), "smplx_vert_segmentation.json")
    if not os.path.exists(json_path):
        try:
            urllib.request.urlretrieve(SMPLX_VERT_SEGMENTATION_URL, json_path)
        except Exception as e:
            raise RuntimeError(
                f"Could not download SMPL-X vertex segmentation from "
                f"{SMPLX_VERT_SEGMENTATION_URL}: {e}\n"
                f"Manually download that file and place it at:\n  {json_path}"
            ) from e
    with open(json_path) as f:
        return json.load(f)
