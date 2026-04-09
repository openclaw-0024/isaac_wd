# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""站立测试脚本：发送全零动作，让机器人保持默认关节角度站立。

用法:
    source ~/miniconda3/bin/activate isaac
    cd /home/rob/isaac_workspace/isaac_wd
    python3 scripts/stand_test.py --task Template-Isaac-Wd-v0 --num_envs 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stand test: zero-action agent.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Task name.")
parser.add_argument("--duration", type=float, default=20.0, help="Run duration in seconds.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import isaac_wd.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Action space : {env.action_space}")
    print(f"[INFO]: Obs space    : {env.observation_space}")
    print("[INFO]: Sending ZERO actions -- robot should stand in default pose.")

    obs, _ = env.reset()

    # 全零动作 = 保持 init_state 里的默认关节角度
    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            obs, reward, terminated, truncated, info = env.step(zero_action)
            step += 1
            if step % 100 == 0:
                print(
                    f"[step {step:05d}]  "
                    f"mean_reward={reward.mean().item():.3f}  "
                    f"terminated={terminated.sum().item()}"
                )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
