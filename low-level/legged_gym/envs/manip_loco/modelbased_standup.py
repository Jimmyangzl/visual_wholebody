import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float

import torch
from typing import Tuple, Dict

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.terrain import Terrain, Terrain_Perlin

import sys

from scipy.linalg import solve_continuous_are

asset_root = "{LEGGED_GYM_ROOT_DIR}/resources/robots/b1z1/urdf/"
asset_file = "b2z1.urdf"

def create_env():
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.001  
    sim_params.substeps = 4  
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    asset = gym.load_asset(sim, asset_root, asset_file)
    env = gym.create_env(sim, gymapi.Vec3(0, 0, 0), 1, 1, True)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0.5)  
    actor = gym.create_actor(env, asset, pose, "b2")