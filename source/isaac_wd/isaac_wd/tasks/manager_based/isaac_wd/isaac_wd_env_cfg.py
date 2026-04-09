# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the fr_urdf humanoid robot (low-speed walking task).

Robot: fr_urdf -- 27-DOF biped
  Legs  (12): hip pitch/roll/yaw, knee pitch, ankle pitch/roll (bilateral)
  Torso  (3): waist yaw, chest roll, chest pitch
  Arms  (10): shoulder pitch/roll, upper-arm yaw, elbow pitch, wrist roll (bilateral)
  Head   (2): head yaw, head pitch

Task: follow a low-speed velocity command (max 0.5 m/s forward).
  Actions  : joint position targets for the 12 leg joints
  Obs      : velocity command + base state + leg joint pos/vel + last action
  Reward   : velocity tracking + gait (feet_air_time) + termination penalty + efficiency
  Done     : timeout + illegal_contact (torso/head touching ground)
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

# 基础 MDP（Isaac Lab 内置）
from . import mdp

# 步态专用 MDP（feet_air_time_positive_biped / feet_slide）
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp

##
# Robot config + joint name constants
##

from isaac_wd.robots import (  # isort:skip
    FR_ROBOT_CFG,
    FR_LEG_JOINTS,
    FR_TORSO_JOINTS,
    FR_ARM_JOINTS,
    FR_FOOT_LINKS,
    FR_TORSO_LINKS,
)


##
# Scene
##


@configclass
class FrHumanoidSceneCfg(InteractiveSceneCfg):
    """Scene: fr_urdf humanoid on flat ground with contact sensors."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = FR_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 接触力传感器：覆盖所有刚体，用于足部空气时间 + 非期望接触惩罚
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


IsaacWdSceneCfg = FrHumanoidSceneCfg


##
# MDP -- Commands（速度指令）
##


@configclass
class CommandsCfg:
    """发给机器人的速度指令：低速前进为主，少量横移与转向。"""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


##
# MDP -- Actions
##


@configclass
class ActionsCfg:
    """12 条腿关节的位置控制指令。"""

    leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=FR_LEG_JOINTS,
        scale=0.25,
        use_default_offset=True,
    )


##
# MDP -- Observations
##


@configclass
class ObservationsCfg:
    """策略输入：速度指令 + 基座状态 + 腿部关节状态 + 上一步动作。"""

    @configclass
    class PolicyCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        base_lin_vel  = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel  = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=FR_LEG_JOINTS)},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=FR_LEG_JOINTS)},
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# MDP -- Events
##


@configclass
class EventCfg:
    """Reset events following H1/G1 pattern: reset base + joints with zero velocity."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


##
# MDP -- Rewards
##


@configclass
class RewardsCfg:
    """Reward terms aligned with H1/G1 reference: velocity tracking as primary driver."""

    # ── 倒地终止惩罚 ──────────────────────────────────────────────────
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # ── 速度跟踪（主要正向奖励，必须主导学习信号）───────────────────
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # ── 步态奖励：双足交替（适中权重，辅助而非主导）─────────────────
    feet_air_time = RewTerm(
        func=loco_mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FR_FOOT_LINKS),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=loco_mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FR_FOOT_LINKS),
            "asset_cfg": SceneEntityCfg("robot", body_names=FR_FOOT_LINKS),
        },
    )

    # ── 姿态保持（高权重：保持直立是前提）─────────────────────────────
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={"target_height": 0.30, "asset_cfg": SceneEntityCfg("robot")},
    )

    # ── 稳定性正则 ────────────────────────────────────────────────────
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # ── 动作平滑（参考 H1 量级，不压制运动）──────────────────────────
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # ── 脚踝约束 ──────────────────────────────────────────────────────
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle.*"])},
    )
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle_roll.*"])},
    )

    # ── 关节偏差 ──────────────────────────────────────────────────────
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_yaw.*", ".*hip_roll.*"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=FR_ARM_JOINTS)},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=FR_TORSO_JOINTS)},
    )


##
# MDP -- Terminations
##


@configclass
class TerminationsCfg:
    """Timeout + illegal_contact (torso/head touching ground)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base_link", ".*waist.*", ".*chest.*", ".*head.*"],
            ),
            "threshold": 1.0,
        },
    )


##
# Full environment config
##


@configclass
class IsaacWdEnvCfg(ManagerBasedRLEnvCfg):
    scene: FrHumanoidSceneCfg = FrHumanoidSceneCfg(num_envs=512, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.viewer.eye = (4.0, 4.0, 3.0)
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
