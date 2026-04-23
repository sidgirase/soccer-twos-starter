import os

# ------------------------------------------------------------------
# BULLETPROOF WINDOWS FIX: 
# Disable Ray's memory monitor to prevent psutil.AccessDenied errors 
# when it tries to inspect Windows system processes.
# ------------------------------------------------------------------
os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"

import gym
import numpy as np
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
import soccer_twos

from reward_shaping_env import ShapedSoccerTwos

# --- CONFIGURATION ---
# Choose your strategy here: "offensive", "defensive", or "balanced"
STRATEGY = "balanced" 
EXPERIMENT_NAME = f"PPO_Soccer_Final_{STRATEGY}"

def env_creator(env_config):
    """Creates the environment and wraps it with our custom reward shaping."""
    # Using a guaranteed safe worker_id based on Ray's worker_index.
    # Adding an offset (e.g., 50) completely bypasses any hanging zombie 
    # Unity processes on default ports from previous crashed runs.
    worker_id = getattr(env_config, "worker_index", 0) + 50
    
    env = soccer_twos.make(
        render=env_config.get("render", False),
        time_scale=env_config.get("time_scale", 50),
        worker_id=worker_id
    )
    return ShapedSoccerTwos(env, strategy=STRATEGY)

if __name__ == "__main__":
    # Initialize Ray (Local or Cluster)
    ray.init(ignore_reinit_error=True)

    # Register the custom environment
    register_env("ShapedSoccer", env_creator)

    # ------------------------------------------------------------------
    # BULLETPROOF FIX: Hardcode the gym spaces.
    # This completely avoids creating a "dummy" Unity environment, 
    # which is the #1 cause of port collisions and UnityWorkerInUseExceptions.
    # ------------------------------------------------------------------
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
    act_space = gym.spaces.MultiDiscrete([3, 3, 3])

    # Configure PPO (Self-Play configuration mapped to a single policy for stability)
    config = {
        "env": "ShapedSoccer",
        "env_config": {
            "render": False,
            "time_scale": 50, # Speed up training simulation
        },
        "framework": "torch",
        "num_workers": 7, # Adjust based on your PACE node (e.g., 4 or 8)
        "num_envs_per_worker": 1,
        "multiagent": {
            # All 4 agents (2 teams of 2) map to the exact same policy.
            # Provide the hardcoded spaces instead of None to fix the ValueError
            "policies": {"shared_policy": (None, obs_space, act_space, {})},
            "policy_mapping_fn": lambda agent_id, *args, **kwargs: "shared_policy",
        },
        "model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
        "lr": 3e-4,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 10,
    }

    print(f"Starting Training with {STRATEGY.upper()} strategy...")

    # Run training using Ray Tune
    tune.run(
        "PPO",
        name=EXPERIMENT_NAME,
        stop={"training_iteration": 5000}, # Change this to 1000+ on PACE
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        config=config,
    )