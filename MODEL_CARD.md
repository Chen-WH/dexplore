# Dexplore Overview

## Description:
Dexplore is a unified optimization framework that combines retargeting and tracking into a single learning loop for dexterous manipulation — using human demonstrations as soft guidance with adaptive spatial scopes to train policies across diverse robot hands. The release comprises (a) the open-source Python training and inference codebase, (b) two pretrained policy checkpoints for the Inspire dexterous hand: `inspire.pth` (state-based PPO teacher) and `inspire_distill.pth` (DAgger-distilled PointNet++/VAE vision student). Dexplore was developed by NVIDIA Research (Seattle Robotics Lab).
_This model is for non-commercial use only._


### License/Terms of Use:
[NVIDIA OneWay Noncommercial License (22 Mar 2022)](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf). Full text is included in the [`LICENSE`](LICENSE) file at the root of this repository.

### Deployment Geography:
Global

### Use Case:
Researchers and developers building AI controllers for dexterous robot hand manipulation. The framework enables training policies across diverse robot hands using human demonstration data, supporting applications such as robotic manipulation research, motion retargeting research, embodied-AI research, and academic reproduction of the accompanying CoRL 2025 paper. The released checkpoints are intended for use in simulation only; they have not been validated for deployment on physical robot hardware.


### Release Date:
GitHub 06/01/2026 via https://github.com/NVlabs/dexplore


## Reference(s):
- Xu, S., Chao, Y.-W., Bian, L., Mousavian, A., Wang, Y.-X., Gui, L.-Y., Yang, W. (2025). *Dexplore: Scalable Neural Control for Dexterous Manipulation from Reference-Scoped Exploration.* In Conference on Robot Learning (CoRL). https://arxiv.org/abs/2509.09671
- Project page: https://sirui-xu.github.io/dexplore/
- NVIDIA Research blog: https://developer.nvidia.com/blog/r2d2-three-neural-breakthroughs-transforming-robot-learning-from-nvidia-research/


## Model Architecture:
**Architecture Type:** Reinforcement-learning policy network. Two checkpoints with different architectures.
**Network Architecture:**
- `inspire.pth` (teacher): Gaussian MLP actor-critic for Proximal Policy Optimization (PPO), implemented via rl_games. Actor and critic MLP units [1024, 1024, 1024, 512] with ReLU activations (see `dexplore/data/cfg/train/rlg/inspire.yaml`).
- `inspire_distill.pth` (student): two single-scale set-abstraction PointNet++ layers + global max-pool over a sampled object point cloud, fused with proprioceptive features through a conditional Variational Autoencoder (VAE) with 64-dim latent (prior conditioned on student observation; encoder conditioned on teacher's privileged observation during training only), followed by an MLP action head with units [1024, 1024, 512] and ReLU activations (see `dexplore/data/cfg/train/rlg/inspire_distill.yaml`). Trained via DAgger with β-annealing against the PPO teacher.

**Number of model parameters:**
- `inspire.pth` (teacher): 8,213,029 parameters (~8.2 M).
- `inspire_distill.pth` (student): 8,175,712 parameters (~8.2 M).

## Input:
**Input Type(s):**
- Teacher: state vector (proprioceptive + reference signals).
- Student: state vector + 3D point cloud of the manipulated object.

**Input Format(s):**
- State: float32 tensor.
- Point cloud (student only): float32 tensor of sampled object surface points.

**Input Parameters:** 1D vector
- Teacher observation: 1442-dim proprioceptive + privileged reference vector (robot-hand state, target hand pose and object pose at the next timestep, contact targets; configured in `dexplore/data/cfg/inspire.yaml`).
- Student observation: 2102-dim — 30-dim local proprioceptive state + 1554-dim point cloud (518 points × 3D coordinates: 6 hand keypoints + 512 sampled object-surface points) + 518-dim validity mask. The student does not use the teacher's privileged reference signals (configured in `dexplore/data/cfg/inspire_distill.yaml`).

**Other Properties Related to Input:** Not applicable. No resolution, characters, context length, image range, alpha channel, or bit-depth parameters apply.


## Output:
**Output Type(s):** Continuous action vector — PD-controller targets for the simulated Inspire robotic hand.
**Output Format:** float32 tensor, range [-1, 1] before being mapped to joint-angle targets.
**Output Parameters:** 1D vector
- Action dimension: 18 (6 wrist DOFs — 3 prismatic translation + 3 revolute rotation — plus 12 finger joints for the Inspire hand).
- Actions are interpreted as PD targets and converted to robot-hand joint commands via `_action_to_pd_targets()` in `dexplore/env/tasks/dexplore_inspire.py`, which also handles the mechanically-coupled joint pairs specific to the Inspire hand.

**Other Properties Related to Output:** Not applicable. No resolution, characters, context length, image range, alpha channel, or bit-depth parameters apply.


Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:
**Runtime Engine(s):** PyTorch 2.2.2+cu121; rl_games 1.1.4; NVIDIA Isaac Gym Preview 4 (runtime simulator, installed separately by the user — not redistributed by Dexplore).
**Supported Hardware Microarchitecture Compatibility:**
* NVIDIA Ampere (verified on RTX 3090)
* NVIDIA Hopper
* NVIDIA Lovelace
* NVIDIA Turing
Requires a single GPU; multi-GPU training optionally available via Horovod.

**Supported Operating System(s):** Linux (verified on Ubuntu 22.04 with kernel 6.8).


The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.


## Model Version(s):
v1.0.0 (initial release; covers both `inspire.pth` teacher and `inspire_distill.pth` student for the Inspire dexterous hand)


## Training, Testing, and Evaluation Datasets:


## Training Dataset:

**Link:** https://grab.is.tue.mpg.de/
**Data Modality:** Multimodal Data (time-series motion capture + 3D mesh geometry + hand–object contact maps)
**Other Training Data Size:** Approximately 6 GB total raw GRAB download (~3.5 GB GRAB parameters + ~2.3 GB raw VICON mocap, summed across the 10 per-subject archives at https://grab.is.tue.mpg.de/, plus small tools/object-mesh archives). Dexplore uses 658 of the ~1,269 GRAB sequences; after Dexplore's preprocessing pipeline (InterAct extraction + dex-retargeting), the per-robot training data subset is approximately 1.9 GB for the Inspire hand release. The original raw GRAB data is not redistributed; users download it directly from MPI.
**Data Collection Method by dataset:** Hybrid: Human, Synthetic
**Labeling Method by dataset:** Not Applicable
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** The GRAB dataset (Taheri et al., ECCV 2020) contains approximately 1,269 motion-capture sequences of full-body human interactions with 51 everyday objects, recorded by 10 subjects (s1–s10) at the Max Planck Institute for Intelligent Systems (Tübingen). Capture used a VICON optical motion-capture system. Each sequence includes SMPL-X body parameters, 6-DoF object pose trajectories, hand-object contact maps, and per-object mesh geometry. Dexplore uses 658 of these sequences for training. In addition, both teacher and student policies are trained on simulated rollouts generated in NVIDIA Isaac Gym, with reference motions retargeted from the GRAB sequences. The rollouts are synthetic experience traces produced by the simulator during training. No additional external dataset and no human labeling were used.

### Testing Dataset:
**Link:** Not applicable; no dedicated held-out testing split is defined for the released checkpoints. Reference data comes from the GRAB dataset (https://grab.is.tue.mpg.de/).
**Data Collection Method by dataset:** Human and simulated rollouts
**Labeling Method by dataset:** Not Applicable
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** Dexplore does not use a conventional training/test split for the teacher or student checkpoints. During inference/evaluation, policies encounter randomized camera views and randomized state initialization that are unseen during training, while the reference-data object categories remain from the same GRAB object set. No held-out GRAB object or sequence split is specified.

### Evaluation Dataset:
**Link:** https://grab.is.tue.mpg.de/

**Benchmark Score:** Smoke-test evaluation of `inspire_distill.pth` on the public Inspire-hand configuration (8 episodes via `dexplore/eval_distill.py`): success rate 50.0% (4/8 episodes survived), mean reward 52.05 ± 37.13, mean steps 104.9. Full paper-reported evaluation numbers are in Section 4 of the CoRL 2025 paper.
**Data Collection Method by dataset:** Human
**Labeling Method by dataset:** Not Applicable
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** Evaluation uses GRAB-derived reference motions and simulated rollouts in NVIDIA Isaac Gym. The paper reports tracking-success-rate, mean-reward, and mean-survival-steps metrics in Section 4, but does not define a dedicated held-out GRAB train/test split for these checkpoints.


## Inference:
**Acceleration Engine:** PyTorch with CUDA acceleration; deployed within NVIDIA Isaac Gym for simulation rollout.
**Test Hardware:** NVIDIA RTX 3090 (Ampere generation, 24 GB VRAM) — verified during pre-release reproduction. Measured peak GPU memory at inference (num_envs=4, headless): teacher ~17.6 GB, student ~17.9 GB. The footprint is dominated by Isaac Gym's pre-allocated physics buffers, not the model weights themselves (which are ~32 MB each). A GPU with at least 24 GB VRAM is recommended; smaller GPUs may work for very small env counts but have not been validated.

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
