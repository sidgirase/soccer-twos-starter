import os
import argparse
import random
import time

os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"
os.environ["RAY_DISABLE_METRICS_COLLECTION"] = "1"
os.environ["RAY_DISABLE_REPORTER"] = "1"
os.environ["RAY_DISABLE_DASHBOARD"] = "1"

import gym
import numpy as np
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
import soccer_twos

from role_based_reward_env import RoleBasedSoccerTwos

parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, default=38)
parser.add_argument("--iters", type=int, default=5000)
parser.add_argument("--name", type=str, default="Team_Roles_Scratch")
args = parser.parse_args()

def env_creator(env_config):
    worker_index = getattr(env_config, "worker_index", 0)
    for attempt in range(30):
        try:
            worker_id = random.randint(100, 10000) + worker_index
            env = soccer_twos.make(render=False, time_scale=50, worker_id=worker_id)
            return RoleBasedSoccerTwos(env) # Using our new Role wrapper!
        except Exception as e:
            if "in use" in str(e).lower():
                time.sleep(random.uniform(0.5, 2.0)) 
                continue
            raise e
    raise RuntimeError("Failed to find open port.")

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    register_env("RolesSoccer", env_creator)

    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
    act_space = gym.spaces.MultiDiscrete([3, 3, 3])

    config = {
        "env": "RolesSoccer",
        "env_config": {},
        "framework": "torch",
        "num_workers": args.workers,
        "num_envs_per_worker": 1,
        "num_gpus": 0, 
        
        # --- THE MAGIC HAPPENS HERE ---
        # We define TWO brains instead of one shared brain!
        "multiagent": {
            "policies": {
                "attacker_policy": (None, obs_space, act_space, {}),
                "defender_policy": (None, obs_space, act_space, {}),
            },
            "policy_mapping_fn": lambda agent_id, *args, **kwargs: 
                "attacker_policy" if agent_id in [0, 2] else "defender_policy",
        },
        
        "model": {"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"},
        "lr": 3e-4,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 10,
        "entropy_coeff": 0.01, 
        "clip_param": 0.2, 
        "vf_loss_coeff": 1.0, 
    }

    print(f"Starting Multi-Brain Role Training: {args.name} | Workers: {args.workers}")

    tune.run(
        "PPO",
        name=args.name,
        stop={"training_iteration": args.iters},
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        config=config,
    )