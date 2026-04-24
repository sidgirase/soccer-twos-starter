import os
import argparse

# ------------------------------------------------------------------
# Disable Ray's memory monitor to prevent psutil.AccessDenied errors 
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

# --- COMMAND LINE ARGUMENTS ---
parser = argparse.ArgumentParser(description="Highly Configurable Soccer Twos Trainer")
parser.add_argument("--strategy", type=str, default="balanced", choices=["offensive", "defensive", "balanced"], help="Reward shaping strategy")
parser.add_argument("--workers", type=int, default=10, help="Number of CPU workers")
parser.add_argument("--iters", type=int, default=3750, help="Total training iterations to reach")
parser.add_argument("--restore", type=str, default=None, help="Absolute or relative path to a checkpoint to resume training")
parser.add_argument("--name", type=str, default="", help="Custom name for the experiment folder")

# Hyperparameters
parser.add_argument("--entropy", type=float, default=0.01, help="Entropy coefficient")
parser.add_argument("--clip", type=float, default=0.2, help="PPO clip parameter")
parser.add_argument("--vf-loss", type=float, default=1.0, help="Value function loss")
parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

args = parser.parse_args()

def env_creator(env_config):
    """Creates the environment with a unique worker_id to avoid port collisions."""
    # Use a high base (1000) + the Ray worker index + a random shift from the PID
    base_offset = 1000 + (os.getpid() % 100)
    worker_id = getattr(env_config, "worker_index", 0) + base_offset
    
    env = soccer_twos.make(
        render=env_config.get("render", False),
        time_scale=env_config.get("time_scale", 50),
        worker_id=worker_id
    )
    return ShapedSoccerTwos(env, strategy=env_config.get("strategy", args.strategy))

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    register_env("ShapedSoccer", env_creator)

    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
    act_space = gym.spaces.MultiDiscrete([3, 3, 3])

    config = {
        "env": "ShapedSoccer",
        "env_config": {
            "render": False,
            "time_scale": 50,
            "strategy": args.strategy,
        },
        "framework": "torch",
        "num_workers": args.workers,
        "num_envs_per_worker": 1,
        "num_gpus": 0, 
        "multiagent": {
            "policies": {"shared_policy": (None, obs_space, act_space, {})},
            "policy_mapping_fn": lambda agent_id, *args, **kwargs: "shared_policy",
        },
        "model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
        "lr": args.lr,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 10,
        "entropy_coeff": args.entropy, 
        "clip_param": args.clip, 
        "vf_loss_coeff": args.vf_loss, 
        "lambda": args.gae_lambda,
    }

    # This dynamically applies the timestamped name we generated in the batch script!
    experiment_name = args.name if args.name else f"PPO_Soccer_Final_{args.strategy}"
    print(f"Starting Training: {experiment_name} | Strategy: {args.strategy} | Workers: {args.workers}")

    restore_path = os.path.abspath(args.restore) if args.restore else None

    tune.run(
        "PPO",
        name=experiment_name,  # Ray Tune will create a brand new folder with this exact name!
        stop={"training_iteration": args.iters},
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        config=config,
        restore=restore_path
    )