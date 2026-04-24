import os
import argparse
import random
import time

# ------------------------------------------------------------------
# Disable Ray's memory monitor and dashboard to prevent PACE errors 
# ------------------------------------------------------------------
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
    """Creates the environment with a robust retry loop to avoid PACE port collisions."""
    worker_index = getattr(env_config, "worker_index", 0)
    
    # Try up to 30 times to find a free port
    for attempt in range(30):
        try:
            # Unity uses base_port + worker_id. 
            # A random range up to 10,000 gives us a massive pool of 9,900 ports to avoid collisions
            # but is small enough that (5005 + 10000) is well below the 65535 TCP port limit!
            worker_id = random.randint(100, 10000) + worker_index
            
            env = soccer_twos.make(
                render=env_config.get("render", False),
                time_scale=env_config.get("time_scale", 50),
                worker_id=worker_id
            )
            return ShapedSoccerTwos(env, strategy=env_config.get("strategy", args.strategy))
            
        except Exception as e:
            error_msg = str(e).lower()
            # If the specific port is taken by a zombie or another user's job on this node
            if "in use" in error_msg or "address already in use" in error_msg:
                print(f"[Attempt {attempt+1}/30] Worker {worker_index} port collision (worker_id={worker_id}). Retrying...")
                time.sleep(random.uniform(0.5, 2.0)) 
                continue
            else:
                # If it's a completely different error, raise it immediately
                raise e
                
    # If we exit the loop, it means we failed 30 times in a row
    raise RuntimeError(f"Failed to find an open port for Unity environment after 30 attempts for worker {worker_index}.")

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

    experiment_name = args.name if args.name else f"PPO_Soccer_Final_{args.strategy}"
    print(f"Starting Training: {experiment_name} | Strategy: {args.strategy} | Workers: {args.workers}")

    restore_path = os.path.abspath(args.restore) if args.restore else None

    tune.run(
        "PPO",
        name=experiment_name,
        stop={"training_iteration": args.iters},
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        config=config,
        restore=restore_path
    )