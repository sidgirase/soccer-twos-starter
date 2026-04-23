"""
Multi-strategy training script for soccer agents.
Trains agents with 5 different reward shaping strategies.
Pass --strategy [1-5] to select which strategy to train.

Usage:
    python train_strategies.py --strategy 1
    python train_strategies.py --strategy 2
    ... etc
"""

import argparse
import ray
from ray import tune
from soccer_twos import EnvType
from reward_shaping_env import create_shaped_rllib_env
from reward_shaping import STRATEGY_DESCRIPTIONS


NUM_ENVS_PER_WORKER = 4


def train_strategy(strategy_id: int, num_iterations: int = 500, num_workers: int = 8):
    """
    Train an agent with the specified reward shaping strategy.
    
    Args:
        strategy_id: Which strategy to use (1-5)
        num_iterations: Number of PPO iterations to run
        num_workers: Number of parallel workers for data collection
    """
    
    ray.init(ignore_reinit_error=True)
    
    # Register the shaped environment
    tune.registry.register_env("SoccerShaped", create_shaped_rllib_env)
    
    strategy_name = f"Strategy_{strategy_id}"
    print(f"\n{'='*80}")
    print(f"Training {strategy_name}: {STRATEGY_DESCRIPTIONS[strategy_id]}")
    print(f"{'='*80}\n")
    
    config = {
        # System settings - CPU-optimized for PACE
        "num_gpus": 0,  # No GPU needed on PACE
        "num_workers": num_workers,
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "log_level": "INFO",
        "framework": "torch",
        
        # RL algorithm config
        "env": "SoccerShaped",
        "env_config": {
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "variation": EnvType.team_vs_policy,
            "multiagent": False,
            "strategy_id": strategy_id,  # Pass strategy to env
        },
        
        # Neural network architecture
        "model": {
            "vf_share_layers": True,
            "fcnet_hiddens": [512, 512],
            "fcnet_activation": "relu",
        },
        
        # PPO-specific hyperparameters
        "lr": 5e-5,
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "vf_clip_param": 0.2,
        "entropy_coeff": 0.01,
        "train_batch_size": 4096,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 20,
    }
    
    # Run training
    analysis = tune.run(
        "PPO",
        name=f"PPO_{strategy_name}",
        config=config,
        stop={
            "timesteps_total": 5000000,  # 5M timesteps per strategy
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        verbose=1,
    )
    
    # Print results
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    
    print(f"\n{'='*80}")
    print(f"Training complete for {strategy_name}")
    print(f"Best episode reward mean: {best_trial.last_result['episode_reward_mean']:.2f}")
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"{'='*80}\n")
    
    return best_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train soccer agent with different reward shaping strategies"
    )
    parser.add_argument(
        "--strategy",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=1,
        help="Which reward shaping strategy to use (1-5)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for training"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Number of PPO iterations"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all 5 strategies sequentially"
    )
    
    args = parser.parse_args()
    
    if args.all:
        print("\n" + "="*80)
        print("TRAINING ALL 5 STRATEGIES SEQUENTIALLY")
        print("="*80)
        
        best_checkpoints = {}
        for strategy_id in range(1, 6):
            checkpoint = train_strategy(
                strategy_id=strategy_id,
                num_workers=args.workers,
            )
            best_checkpoints[strategy_id] = checkpoint
        
        print("\n" + "="*80)
        print("ALL TRAINING COMPLETE")
        print("="*80)
        for strategy_id, checkpoint in best_checkpoints.items():
            print(f"Strategy {strategy_id}: {checkpoint}")
    else:
        train_strategy(
            strategy_id=args.strategy,
            num_workers=args.workers,
        )
