"""ArticulationCfg for the fr_urdf humanoid robot.

USD asset location (after URDF conversion):
    isaaclab_assets/data/Robots/Custom/FrUrdf/fr_robot.usd

Joint summary (27 revolute joints from fr_urdf.urdf):
    Legs  (12): l/r_hip_pitch/roll/yaw, l/r_knee_pitch, l/r_ankle_pitch/roll
    Torso  (3): waist_yaw, chest_roll, chest_pitch
    Arms  (10): l/r_shoulder_pitch/roll, l/r_upper_arm_yaw, l/r_elbow_pitch, l/r_wrist_roll
    Head   (2): head_yaw, head_pitch
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

##
# Joint name constants (used by MDP to avoid hard-coded strings)
##

# Leg joints (used for locomotion actions/rewards)
FR_LEG_JOINTS = [
    "l_hip_pitch_joint", "l_hip_roll_joint", "l_hip_yaw_joint",
    "l_knee_pitch_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint",
    "r_hip_pitch_joint", "r_hip_roll_joint", "r_hip_yaw_joint",
    "r_knee_pitch_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint",
]

# Torso joints
FR_TORSO_JOINTS = ["waist_yaw_joint", "chest_roll_joint", "chest_pitch_joint"]

# Arm joints
FR_ARM_JOINTS = [
    "l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_upper_arm_yaw_joint",
    "l_elbow_pitch_joint", "l_wrist_roll_joint",
    "r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_upper_arm_yaw_joint",
    "r_elbow_pitch_joint", "r_wrist_roll_joint",
]

# Head joints
FR_HEAD_JOINTS = ["head_yaw_joint", "head_pitch_joint"]

# All 27 joints
FR_ALL_JOINTS = FR_LEG_JOINTS + FR_TORSO_JOINTS + FR_ARM_JOINTS + FR_HEAD_JOINTS

# Foot links (last links in the leg chain; used for contact sensor / termination)
FR_FOOT_LINKS = ["l_ankle_roll_link", "r_ankle_roll_link"]

# Torso/base links (contact triggers robot-fell termination)
FR_TORSO_LINKS = ["base_link", "chest_pitch_link"]

# Legacy aliases kept for backward compatibility with template MDP wiring
MY_ROBOT_CART_JOINT = "l_hip_pitch_joint"
MY_ROBOT_POLE_JOINT = "r_hip_pitch_joint"

##
# USD path resolution
##

_LOCAL_USD = os.path.join(
    ISAACLAB_ASSETS_DATA_DIR,
    "Robots", "Custom", "FrUrdf", "fr_robot.usd",
)

##
# ArticulationCfg
##

MY_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_LOCAL_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.30),
        joint_pos={
            "l_hip_pitch_joint": -0.15,
            "r_hip_pitch_joint": 0.15,
            ".*hip_roll.*": 0.0,
            ".*hip_yaw.*": 0.0,
            ".*knee_pitch.*": 0.30,
            ".*ankle_pitch.*": -0.02,
            ".*ankle_roll.*": 0.0,
            "waist_yaw_joint": 0.0,
            "chest_roll_joint": 0.0,
            "chest_pitch_joint": 0.0,
            ".*shoulder.*": 0.0,
            ".*upper_arm_yaw.*": 0.0,
            ".*elbow.*": 0.0,
            ".*wrist.*": 0.0,
            "head_yaw_joint": 0.0,
            "head_pitch_joint": 0.25,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # PD 增益 = 硬件值 × 25，参考 H1 高刚度+适中阻尼；effort_limit 大幅提升避免截断
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"],
            effort_limit_sim=150.0,
            velocity_limit_sim=6.283,
            stiffness=200.0,
            damping=20.0,
            armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=80.0,
            velocity_limit_sim=6.283,
            stiffness=120.0,
            damping=10.0,
            armature=0.01,
        ),
        "chest": ImplicitActuatorCfg(
            joint_names_expr=["chest_roll_joint", "chest_pitch_joint"],
            effort_limit_sim=80.0,
            velocity_limit_sim=6.283,
            stiffness=60.0,
            damping=10.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*shoulder.*", ".*upper_arm_yaw.*", ".*elbow.*"],
            effort_limit_sim=40.0,
            velocity_limit_sim=5.236,
            stiffness=45.0,
            damping=10.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[".*wrist.*"],
            effort_limit_sim=20.0,
            velocity_limit_sim=8.378,
            stiffness=45.0,
            damping=2.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=8.378,
            stiffness=45.0,
            damping=2.0,
        ),
    },
)
