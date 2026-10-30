# Dexplore → UR10 + DexHand021 真机实施计划

目标：用 Dexplore 框架（GRAB 人手示范 → 重定向 + 跟踪一体化 RL → 视觉蒸馏），
为实验室的 UR10 + DexHand021（左手）真机系统训练可部署的灵巧操作策略。

作者发布的 `checkpoint/inspire.pth` / `inspire_distill.pth` 只适用于仿真中的
Inspire 手（动作/观测维度与 Inspire 运动学绑定，MODEL_CARD 明确未在真机验证），
不直接用于本系统；它们只用于验证数据流程与训练流程。

---

## 0. 已完成（2026-10-29 ~ 10-30）

- [x] 环境：conda env `dexplore`（py3.8, Isaac Gym Preview 4, torch 2.4.1+cu121,
      rl-games 1.1.4, dex-retargeting 0.4.6, sapien 3.0.0b1, numpy 1.23.5, trimesh 4.2.4）。
- [x] GRAB 预处理（公开 InterAct 不产出 `motion.npz`，自行补齐）：
  - `data_processing/prepare_grab_sequences.py`：raw GRAB → `sequences/*/{object,human,motion}.npz`
  - `data_processing/make_grab_skeletons.py`：生成 `dexplore/data/assets/smplx/smplx_grab_s{N}.xml`
- [x] `convert_grab.py` 修复：接触标签原先在 Rx(+90°) 旋转后的人体与 y-up 物体之间计算，
      坐标系不一致导致手部接触标签全为 -1，左手 `*_pass_1` 序列未被过滤并使加载崩溃。
      现改为 y-up 下计算；距离计算放到 GPU（与 CPU 结果逐位一致，10 s → 1.2 s/条）；
      新增 `--contact_only` 只刷新已有输出的接触列。修复后加载器保留 657 条右手序列，
      与 MODEL_CARD 的 658 条吻合。
- [x] Inspire 重定向：1335 条全部完成（`~/data/GRAB/dexplore_motions`，软链接为 `data/`）。
- [x] 作者 teacher 评估（64 env，前 65 条均为 s10）：成功率 67.3%，手误差 0.072 m，物体误差 0.043 m。
- [x] 本机吞吐基准（RTX 4070 Ti SUPER 16GB）：8192/4096 env 显存不足；3072 env 可跑，
      约 2.17 万步/秒（含 PPO 更新），每 epoch ≈ 9 s。作者 checkpoint 对应 31.5 亿环境步，
      本机约 40 h；建议在 AutoDL 租 RTX 4090 24GB 或 48GB 卡（L20/A40）训练。

---

## 1. 硬件与资产

资产仓库：`~/ros2_ws/src/robot-assets`（git: Chen-WH/robot-assets）

| 项目 | 路径 / 说明 |
|---|---|
| DexHand021 | `dexrobot_021/urdf/dexhand021_{left,right}_{convex_mimic,primitive_mimic,convex_collision,full_visual}.urdf`；RL 用 `*_convex_mimic` 或 `*_primitive_mimic` |
| DexHand021 关节 | 左手 19 个转动关节（`l_f_joint3_1` 固定），7 个 mimic（各指 `_4` 跟随 `_3`：0.9023x−0.041；`4_1`、`5_1` 跟随 `2_1`），**12 个主动自由度**；五个 `*_pad` link 作为接触/触觉锚点 |
| DIP/PIP 真实耦合 | `dexrobot_021/params.txt`、`末端指节映射关系.csv`：真实耦合是多项式，URDF mimic 是线性近似 |
| UR10 | `ur10/`；整机场景 `scene/urdf/ur10_dexhand.urdf`（UR10 + mount + **左手** + 腕部 D435i） |
| 手眼/安装标定 | `scene/calibration/ur10_dexhand_20261011.json`（`T_tcp_hand`，适用于当前左手安装；支架 + 法兰约 15 mm、基准不确定） |
| 外部相机标定 | `scene/calibration/external_cameras_20261011_025.json`（`T_base_external_d435`、`T_base_azure`） |
| 桌面 | `scene/calibration/table_boundary_20261012.json`：70×70 cm，中心在 base 系 (0.027, −0.860, 0.112) m |
| 相机 | 外部 D435、腕部 D435i、Azure Kinect DK |
| 带 AprilTag 的物体 | `reorientation_cube/`、`reorientation_cylinder/`、`tabletop_cardboard/` |

---

## 2. 关键决策

> 2026-10-30 已确认：D1 选 A（预处理镜像成左手），D2 保留浮动手腕，D3 用外部 D435，
> D4 需要对齐；如果对不齐，可以调整真实实验场景（桌子和相机的摆放）来配合。

### D1 左右手：GRAB 是右手数据，真机装的是左手
- **A（推荐）预处理时镜像**：把右手关键点、手腕位姿、物体位姿和桌子关于人体矢状面镜像，
  物体网格同步镜像，再重定向到 `dexhand021_left_*`。仿真与真机完全一致，部署时不需要额外变换。
  镜像只涉及重定向用到的量（手部关键点、腕/物体/桌子位姿、接触标签左右互换），不需要镜像整个 SMPL-X。
- B 仿真用右手训练，部署时对观测和动作做镜像：改动少，但部署接口容易出错，且仿真里的手与真机不一致。
- C 改装右手：需要重新做安装标定。

### D2 手腕：仿真是 6 自由度浮动手腕，真机是 UR10
- **推荐**：保留浮动手腕（框架不变），并加入：
  (1) 把参考轨迹对齐到真实桌面，剔除 UR10 不可达或姿态不可行的序列；
  (2) 训练时随机化手腕跟踪滞后和噪声；
  (3) 限制手腕速度和加速度，使其匹配 UR10（servoJ/RTDE）。
  部署前先在 MuJoCo 中用 `ur10_dexhand` 跑"IK 跟踪手腕目标"的闭环回放来验证。
- 备选：仿真中加入完整 UR10、以臂关节作为动作。更贴近真机，但改动框架较大，留作第二轮。

### D3 学生策略用哪台相机
- 仿真里是固定相机：`dexplore_distill.py:136`，位姿 `[-0.12, -0.78, 1.4]`，128×128，FOV 80.96°。
- **推荐外部 D435**：已有 `T_base_external_d435` 标定，把它变换到仿真世界系后替换上述位姿，
  并将 FOV 和分辨率按 D435 深度流设置。Kinect 留作对照或融合；腕部 D435i 后续可用
  `is_body_camera` 试验。
- 真实点云需要分割出物体：去桌面平面、按机器人模型去掉手和臂，必要时用 SAM2。

### D4 世界系对齐
- 每条序列做平移和绕 z 轴旋转，使 GRAB 桌面与真实桌面重合（中心、高度），
  再在 UR10 base 系下检查可达性。GRAB 桌子尺寸与真实 70×70 cm 不同，需要检查物体初始位置是否落在桌内。

---

## 3. 分阶段任务

### 阶段 0：仓库整理
- [ ] 提交 GRAB 预处理脚本和 `convert_grab.py` 修复（可考虑向 upstream 提 PR）。
- [x] `data_processing/make_dexhand021_urdf.py`：从 `dexhand021_left_convex_mimic.urdf` 生成
      `dexplore/data/assets/dexhand021/dexhand021_left.urdf`（前置 6 自由度浮动手腕，与 Inspire 相同；
      去掉 mimic 标签）；`meshes/` 软链接到 robot-assets（不复制、不入库）。
      `--base_rpy`（手腕系到手掌基座的旋转）暂为 0，阶段 2 标定。

### 阶段 1：DexHand021 接入仿真（本机即可）
- [x] Isaac Gym（CPU）加载通过：25 DOF（6 手腕 + 19 手指）、33 个刚体、26 个碰撞体（含 5 个 pad）、
      总质量 1.01 kg，限位与 URDF 一致。Isaac Gym 中的 DOF 顺序与联动关系：

      | 手指 | DOF 索引 | 主动 | 从动（URDF 线性 mimic） |
      |---|---|---|---|
      | 拇指 1 | 6–9 | 6, 7, 8 | 9 = 0.9023·[8] − 0.041 |
      | 食指 2 | 10–13 | 10（侧摆）, 11, 12 | 13 = 0.9023·[12] − 0.041 |
      | 中指 3 | 14–16 | 14, 15 | 16 = 0.9023·[15] − 0.041 |
      | 无名指 4 | 17–20 | 18, 19 | 17 = [10]；20 = 0.9023·[19] − 0.041 |
      | 小指 5 | 21–24 | 22, 23 | 21 = 1.8162·[10] + 0.0181；24 = 0.9023·[23] − 0.041 |

      主动自由度共 12 个（6, 7, 8, 10, 11, 12, 14, 15, 18, 19, 22, 23）。
- [x] 联动关系评估：Isaac Gym 导入 URDF 时忽略 mimic，从动关节作为独立 DOF 加载。Inspire 的做法是在
      `_action_to_pd_targets` / `_set_env_state` 中用主动关节的目标值覆盖从动关节（作者用的系数，
      如 ×1.05、×0.6、×0.8，与 URDF 中的 mimic 系数并不相同），属于 PD 软耦合。DexHand021 照此实现，
      系数直接取 URDF 的 mimic 值。
      对照 `末端指节映射关系.csv`：在 URDF 限位内（PIP ≤ 1.3 rad ≈ 74.5°），线性近似的 DIP 误差为
      −3.2°～+2.1°；实测 PIP 最大可到 86.7°，超出 URDF 上限，该段误差最大 9°（仿真中到不了）。
      真机只发送 12 个主动关节指令，联动由机械结构完成，因此线性近似只影响仿真与真机的一致性。
      如果之后发现深度弯曲的抓取在真机上表现差，再考虑放宽 PIP 上限或改用多项式耦合。
      Isaac Gym 不支持 URDF mimic：从动关节按独立 DOF 加载，在 `_action_to_pd_targets`
      中由主动关节计算（仿照 `dexplore_inspire.py:32`），优先使用 `params.txt` 的多项式耦合。
- [ ] PD 增益、力矩和速度上限参照真机 SDK（做简单的系统辨识：阶跃响应）。

### 阶段 2：重定向数据

设计要点（2026-10-30）：
- **手腕系不用额外标定**：`convert_grab.py` 开了 `add_dummy_free_joint`，手腕 6 个自由度和手指在
  重定向中一起优化，手腕 qpos 定义在重定向所用 URDF 的手掌基座系下。重定向和仿真用同一份
  DexHand021 URDF（同一个 `left_hand_base`），两边的 dummy 链顺序都是 xyz 平移加 xyz 旋转，
  因此 `--base_rpy` 保持 0 即可。Inspire 的 `rpy="3.14 0 1.57"` 是因为它的仿真 URDF 和
  dex-urdf 版本的基座系不同。
- **重定向配置**：dex-retargeting 的 `RobotName` 枚举里没有 DexHand021，改为直接加载自定义 yml：
  - `urdf_path` 用 `dexhand021_left_convex_mimic.urdf`，`ignore_mimic_joint: False`，
    这样只优化 12 个主动关节，从动关节由 mimic 推出，和真机耦合一致；
  - 目标 link：五个 `*_pad` 对应 MANO 指尖 [4, 8, 12, 16, 20]；再加 DIP/IP 处的 `l_f_link{1..5}_4`
    对应 [3, 7, 11, 15, 19]，参照 Leap 配置，用来约束手腕姿态。
- **镜像（D1-A）**：在 z-up 的最终数据系中取反射 M = diag(−1, 1, 1)，整个场景一起镜像：
  - 位置：p' = M p，包括手部关键点（重定向输入）、手腕、物体、桌子、`body_pos`；
  - 全局旋转：R' = M R M（仍是正常旋转）；exp-map 形式为 v' = −M v；
  - 物体：世界点 M(R v + t) = (M R M)(M v) + M t，所以网格也要在物体局部系里做 x → −x 镜像。
    需要生成镜像物体的网格和 URDF（`objects_mirror/`），镜像后注意三角面绕序要翻转；
  - 镜像后的右手就成了左手，仍放在"受控手"槽位（`51:102`、`body_pos[16:]`、`contact[16:]`），
    原来的左手数据镜像后放进另一个槽位，"左手有接触则过滤"的规则保持不变；
  - 输出文件名 `interaction_hand_dexhand021.pt`，与 Inspire 数据并存。
- 待查：任务观测和奖励是否用到 `dof_smpl`（6:51、57:102）这类局部关节角；
  如果用到，也要按 v' = −M v 镜像。

- [x] 重定向配置 `data_processing/retarget_configs/dexhand021_left.yml`：只优化 12 个主动关节（mimic 生效），
      目标 = 五指 pad→MANO 指尖、远节 link→DIP/IP、拇指 link1_3→MCP（不加 MCP 时拇指 IP 偏差约 3 cm，加后约 6 mm）。
      `robot_configs.py` 新增 `dexhand021`（`mirror=True`，SAPIEN↔Isaac 关节顺序由关节名生成）。
- [x] 修复原版 bug：重定向用的手部关键点取了 `joints[:, 50:65]`，SMPL-X 右手实为 40–54
      （Inspire 只用指尖顶点，不受影响；Leap 等用到 DIP 的配置原本会拿到脸部/脚部的点）。
- [x] 镜像实现：`convert_grab.mirror_interaction_data`；镜像物体由 `make_mirrored_objects.py` 生成
      （网格放在 `~/data/GRAB/grab_dir/objects_mirror`，URDF 在 `mjcf/mirror/`），任务通过 `objectMirror: True` 选用。
- [x] 验证（3 条序列）：镜像后的位置、旋转、接触与 Inspire 数据逐列一致；人手到物体的距离镜像前后完全相同；
      四指 DIP 平均偏差 5–15 mm、拇指 5–7 mm；接触帧中五指 pad 到物体表面的中位距离多在 2–11 mm。
- [x] 全量转换：1335 条全部成功（`~/data/GRAB/convert_dexhand021.log`）；去掉左手接触和门把手后剩 657 条，
      加上难物体的重复采样，训练时加载 782 条。
- [x] D4 可达性扫描（结果在 `~/data/GRAB/ur10_reachability/`，文件名 `y{挪远}_z{降低}.json`，单位 m，
      为相对当前标定桌面中心的偏移）：

      | 桌面偏移 | 完整可行 | ≥90% 帧可行 | 需放慢（中位） | 有桌面碰撞的序列 | 有 IK 失败的序列 |
      |---|---|---|---|---|---|
      | 当前位置 (0, 0) | 195/657 (30%) | 48% | 1.8× | 62% | 26% |
      | 降低 20 cm | 214 (33%) | 51% | 1.9× | 59% | 31% |
      | 降低 30 cm | 249 (38%) | 57% | 1.8× | 49% | 32% |
      | 降低 40 cm | 272 (41%) | 56% | 1.8× | 44% | 39% |
      | 挪远 15 cm、降低 40 cm | 276 (42%) | 55% | 1.5× | 43% | 36% |
      | 挪远 30 cm | ≤191 | ≤42% | 1.5× | ~49% | ~50% |

      结论：桌面相对 UR10 降低 30–40 cm 收益最大（30% → 40% 完整可行），再挪远 15 cm 可把需要的放慢倍数从 1.8× 降到 1.5×；
      挪远 30 cm 开始够不着。IK 失败主要是把物体举到面前的动作（eat/drink/see/inspect）。
      （`y-0.15_z-0.2` 这组在运行中途段错误，待查；不影响趋势。）
- [ ] 训练集取舍（待定）：全部 657 条 vs 只用在选定桌面摆放下可行的子集；以及是否按 1.5–1.8× 放慢参考动作。

### 阶段 3：仿真任务
- [x] `dexplore/env/tasks/dexplore_dexhand021.py`（`Dexplore_Dexhand021`，已在 `utils/parse_task.py` 注册）：
      联动按上表在 `_action_to_pd_targets` / `_set_env_state` 中实现；碰撞过滤与 Inspire 相同
      （远节 `_4` 和 pad 为 3，其余为 2）；PD 增益和速度暂时与 Inspire 相同，待系统辨识后再调。
- [x] `dexplore/data/cfg/dexhand021.yaml` + `train/rlg/dexhand021.yaml`：`numDof: 25`、`objectMirror: True`；
      `numObs` 与自由度数无关，所有机器人都是 1442。keyBodies 按人手顺序：手腕，食指、中指、小指、无名指的
      `_2/_3/_4`，拇指 `_2/_3/_4`；contactBodies 为五个远节 `_4`。
      仿真 URDF 去掉了 pad 的 1 mm 碰撞片，让指尖接触落在远节上。
- [x] 冒烟测试：256 env、107 条动作，46 个 epoch 内平均奖励从 4.3 涨到 13.8，训练流程正常。
- [ ] 运动学回放参考轨迹（`playdataset`），目视检查手和镜像物体是否对齐。
- [ ] 加域随机化：摩擦、物体质量和尺度、PD 增益、动作延迟（1–3 步）、手腕跟踪滞后和噪声、观测噪声。

### 阶段 4：训练（AutoDL）

本机实测（2026-10-30，RTX 4070 Ti SUPER 16GB，DexHand021，782 条动作）：
- 3072 env、接触对上限 8M、缓冲倍数 10：约 2.06 万步/秒（含 PPO 更新，与 Inspire 的 2.17 万相当），
  但第 9 个 epoch 时 PhysX 接触缓冲溢出（`compressContactStage1 fail`）。
- 接触对提到 16M 后，3072 和 2048 env 都显存不足。PyTorch 一侧约 6 GB（主要是 782 条参考动作的数据，
  与环境数无关），其余为 PhysX 缓冲。结论：本机无法稳定训练 DexHand021，需上云。

时间估计（以作者 Inspire checkpoint 的 31.5 亿环境步为基准；云端速度按 CUDA 核心数和更大的环境数推算，未实测）：

| 方案 | RTX 4090 24GB（约 4096 env，缓冲需调小） | L20/A40 48GB（8192 env，原版缓冲） |
|---|---|---|
| Teacher，参考动作原速 | 约 19–25 h | 约 24–35 h |
| Teacher，参考动作放慢 1.6×（回合变长，步数约 ×1.6） | 约 30–40 h | 约 40–55 h |
| Student 蒸馏（作者约 2500 万步，需要相机渲染） | 约 2–6 h | 约 2–6 h |

- 只用可行子集训练时，动作数约减半，收敛可能更快，但预算仍建议按上表准备，用评估成功率决定何时停。
- 风险：student 蒸馏需要 Isaac Gym 相机渲染，云容器若不支持 Vulkan/EGL 图形渲染会跑不了；
  蒸馏只用 784 env，可考虑回本机做（需先确认显存）。
- [ ] 打包上传：`data/`（只含 dexhand021 数据）、`dexplore/data/assets`（跟随软链接）、代码。
- [ ] 租卡后先跑 5 分钟测吞吐和显存，再决定环境数、缓冲大小和租期。
- [ ] Teacher：每 500 epoch 存档并用 `evaluate.py` 评估，收敛即停。
- [ ] Student 蒸馏：相机换成 D435 的标定位姿，加入 D435 深度噪声模型和点云丢点。

### 阶段 5：真机（每步都先做碰撞和状态检查；改场景模型不等于允许运动）
- [ ] ROS2 接口：UR10（ur_robot_driver / RTDE servoJ）、DexHand021 SDK、
      realsense2_camera / Azure Kinect 驱动，用 TF 发布标定结果。
- [ ] 安全层：工作空间盒、关节速度和加速度限幅、桌面碰撞余量（考虑约 15 mm 安装不确定度）、急停。
- [ ] 5.1 不带物体，开环回放重定向轨迹（UR10 IK + 手关节）：验证标定、可达性和速度。
- [ ] 5.2 带物体开环回放：检查抓取几何。
- [ ] 5.3 Teacher 策略 + AprilTag 物体位姿（用带标签的方块或圆柱）：在不涉及视觉的情况下验证动力学迁移。
- [ ] 5.4 Student 策略 + 外部 D435 点云。

---

## 4. 风险
- GRAB 动作幅度大（物体可被举到约 1.7 m 高）：需要按 UR10 工作空间筛选或缩放，可用序列会明显减少。
- URDF mimic 是线性近似，真实 DIP/PIP 是多项式耦合；手指零位和 SDK 映射有残差（见标定文件 `limitations`）。
- 手眼和安装标定基于视觉配准，不是机械计量，存在毫米级误差。
- Dexplore 基于 Isaac Gym（已停止维护）；robot-assets 里的 USD 是给 IsaacLab 用的，不能直接复用，
  本计划沿用 Isaac Gym + URDF。
- 显存：本机只能开 3072 env，与作者的 8192 env 训练设置不同。

---

## 5. 路径索引
- GRAB 原始数据和预处理结果：`~/data/GRAB/grab_dir/{raw,sequences,objects,tools}`
- 重定向输出：`~/data/GRAB/dexplore_motions`（仓库内 `data/`）
- SMPL-X：`~/data/SMPL-X/models/smplx`
- InterAct：`~/Git/InterAct`；dex-urdf：`~/Git/dex-urdf/robots/hands`
- 日志：`~/data/GRAB/{convert_inspire,contact_fix,eval_inspire}.log`
