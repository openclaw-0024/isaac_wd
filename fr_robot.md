# FR 人形机器人（fr / fr_urdf）

本文档汇总当前工程中 **FR 双足人形** 在 Isaac Lab 扩展内的资源路径、代码入口，以及 **URDF 关节限位**（与仿真/控制一致时的参考）。

---

## 1. 资源与路径

| 项目 | 路径 |
|------|------|
| URDF 源文件（当前使用版本） | `/home/rob/work/urdf/0408/fr.urdf`（`robot` 名：`fr02_urdf_02_0404`） |
| 导入后的 USD（需含 `configuration/` 等引用） | `isaaclab_assets/data/Robots/Custom/FrUrdf/fr_robot.usd`（通过 `ISAACLAB_ASSETS_DATA_DIR` 解析） |

网格与配置目录应与 URDF 中 `meshes/`、`configuration/` 引用保持一致，否则加载会失败或缺碰撞。

---

## 2. 代码入口（本扩展 `isaac_wd`）

| 内容 | 文件 |
|------|------|
| 机器人 `ArticulationCfg`、关节名常量、足部/躯干 link 名 | `source/isaac_wd/isaac_wd/robots/my_robot.py` |
| 行走/站立 MDP：场景、指令、奖励、终止等 | `source/isaac_wd/isaac_wd/tasks/manager_based/isaac_wd/isaac_wd_env_cfg.py` |
| skrl PPO 训练超参 | `source/isaac_wd/isaac_wd/tasks/manager_based/isaac_wd/agents/skrl_ppo_cfg.yaml` |

**关节分组（27 个转动关节）**

- 腿 12：`l/r_hip_pitch|roll|yaw`、`l/r_knee_pitch`、`l/r_ankle_pitch|roll`
- 躯干 3：`waist_yaw`、`chest_roll`、`chest_pitch`
- 手臂 10：左右肩俯仰/翻滚、上臂 yaw、肘、腕滚转
- 头 2：`head_yaw`、`head_pitch`

**常用 Gym 任务名**：`Template-Isaac-Wd-v0`（以 `isaac_wd` 扩展注册为准）。

**脚本示例**（需在已安装 Isaac Lab / Isaac Sim 的 Python 环境中执行）：

- 训练：`python3 scripts/skrl/train.py --task Template-Isaac-Wd-v0 --headless --num_envs <N>`
- 播放：`python3 scripts/skrl/play.py --task Template-Isaac-Wd-v0 --num_envs 4`
- 站立零动作测试：`python3 scripts/stand_test.py`（若仓库中保留）

---

## 3. 仿真中的限位说明

`my_robot.py` 中设置了 `soft_joint_pos_limit_factor=0.9`，Isaac Lab 会在关节软限位上相对 URDF 再收缩，**实际控制/奖励用的有效范围**略小于下表 raw 限位。

---

## 4. URDF 关节限位（弧度 rad）

以下数据摘自 `/home/rob/work/urdf/0408/fr.urdf` 中各 `<limit lower="..." upper="..." .../>`。**effort** 单位为 N·m，**velocity** 为 rad/s。

### 4.1 腿部（12）

| 关节名 | lower (rad) | upper (rad) | effort | velocity |
|--------|-------------|-------------|--------|----------|
| `l_hip_pitch_joint` | -2.4 | 2.4 | 1.6 | 6.283 |
| `l_hip_roll_joint` | -0.2 | 1.46 | 1.6 | 6.283 |
| `l_hip_yaw_joint` | -1.57 | 1.57 | 0.4 | 10.472 |
| `l_knee_pitch_joint` | 0 | 1.95 | 1.6 | 6.283 |
| `l_ankle_pitch_joint` | -0.37 | 0.52 | 1.6 | 6.283 |
| `l_ankle_roll_joint` | -0.52 | 0.47 | 1.2 | 6.283 |
| `r_hip_pitch_joint` | -2.4 | 2.4 | 1.6 | 6.283 |
| `r_hip_roll_joint` | -1.46 | 0.2 | 1.6 | 6.283 |
| `r_hip_yaw_joint` | -1.57 | 1.57 | 0.4 | 10.472 |
| `r_knee_pitch_joint` | 0 | 1.95 | 1.6 | 6.283 |
| `r_ankle_pitch_joint` | -0.37 | 0.52 | 1.6 | 6.283 |
| `r_ankle_roll_joint` | -0.47 | 0.52 | 1.2 | 6.283 |

说明：左右 **hip_roll** 限位为镜像关系；左右 **ankle_roll** 上下界略有差异，以 URDF 为准。

### 4.2 躯干（3）

| 关节名 | lower (rad) | upper (rad) | effort | velocity |
|--------|-------------|-------------|--------|----------|
| `waist_yaw_joint` | -0.6 | 0.6 | 1.2 | 6.283 |
| `chest_roll_joint` | -0.27 | 0.27 | 1.2 | 6.283 |
| `chest_pitch_joint` | 0 | 0.82 | 1.2 | 6.283 |

### 4.3 手臂（10）

| 关节名 | lower (rad) | upper (rad) | effort | velocity |
|--------|-------------|-------------|--------|----------|
| `l_shoulder_pitch_joint` | -0.8 | 3 | 0.4 | 5.236 |
| `l_shoulder_roll_joint` | -1.47 | 0.62 | 0.4 | 5.236 |
| `l_upper_arm_yaw_joint` | -1.4 | 1.4 | 0.2 | 8.378 |
| `l_elbow_pitch_joint` | 0 | 2.2 | 0.4 | 5.236 |
| `l_wrist_roll_joint` | -2.1 | 2.1 | 0.14 | 4.189 |
| `r_shoulder_pitch_joint` | -3 | 0.8 | 0.4 | 5.236 |
| `r_shoulder_roll_joint` | -0.62 | 1.47 | 0.4 | 5.236 |
| `r_upper_arm_yaw_joint` | -1.4 | 1.4 | 0.2 | 8.378 |
| `r_elbow_pitch_joint` | -2.2 | 0 | 0.4 | 5.236 |
| `r_wrist_roll_joint` | -2.1 | 2.1 | 0.14 | 4.189 |

说明：左右肩/肘区间因轴镜像与命名约定不同，数值上不对称是正常现象。

### 4.4 头部（2）

| 关节名 | lower (rad) | upper (rad) | effort | velocity |
|--------|-------------|-------------|--------|----------|
| `head_yaw_joint` | -1 | 1 | 0.2 | 8.378 |
| `head_pitch_joint` | -0.87 | 0.52 | 0.4 | 6.283 |

---

## 5. 角度对照（可选）

近似换算：`deg ° = rad × 180 / π`。例如 `1.57 rad ≈ 90°`；`0.82 rad ≈ 47°`。

---

## 6. 变更记录

- 文档与 `fr.urdf`（0408 版本）及 `my_robot.py` 中的关节列表对齐；若你更新 URDF，请同步更新本文件中的限位表与路径。
