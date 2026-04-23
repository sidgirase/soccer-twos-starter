"""
Evaluation script for trained agents.
Tests agents against random agent and baseline agent.
Generates comparison plots and statistics.

Usage:
    python evaluate_agents.py --checkpoint path/to/checkpoint --num-matches 10
"""

import argparse
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

import soccer_twos
from example_player_agent import agent_random


def evaluate_agent(checkpoint_path: str, opponent_policy=None, 
                   num_matches: int = 10, max_steps: int = 5000):
    """
    Evaluate a trained agent against an opponent.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        opponent_policy: Policy for opponent (None = random)
        num_matches: Number of matches to play
        max_steps: Maximum steps per match
        
    Returns:
        Dictionary with statistics
    """
    
    # Create environment with trained policy vs opponent
    env = soccer_twos.make(
        variation=soccer_twos.EnvType.team_vs_policy,
        opponent_policy=opponent_policy or (lambda *_: 0),
    )
    
    # TODO: Load trained policy from checkpoint
    # For now, using random policy for testing
    # In practice, you would load the RLlib checkpoint here
    
    wins = 0
    losses = 0
    draws = 0
    total_goals_for = 0
    total_goals_against = 0
    total_reward = 0
    
    for match_num in range(num_matches):
        obs = env.reset()
        episode_reward = 0
        goals_for = 0
        goals_against = 0
        
        for step in range(max_steps):
            # Sample action from trained policy
            # TODO: Replace with actual trained policy prediction
            if isinstance(obs, dict):
                actions = {agent_id: env.action_space.sample() 
                          for agent_id in obs}
            else:
                actions = env.action_space.sample()
            
            obs, rewards, dones, info = env.step(actions)
            
            # Accumulate episode reward
            if isinstance(rewards, dict):
                episode_reward += sum(rewards.values())
            else:
                episode_reward += rewards
            
            # Check if episode done
            if isinstance(dones, dict):
                if all(dones.values()):
                    break
            elif dones:
                break
        
        total_reward += episode_reward
        
        # For now, just record results
        # In practice, extract win/loss from environment info
        
    env.close()
    
    return {
        'num_matches': num_matches,
        'avg_reward': total_reward / num_matches,
        'win_rate': wins / num_matches if num_matches > 0 else 0,
    }


def plot_comparison(results: dict):
    """
    Plot comparison of agent strategies.
    
    Args:
        results: Dictionary of {strategy_name: metrics}
    """
    strategies = list(results.keys())
    rewards = [results[s]['avg_reward'] for s in strategies]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Average rewards
    axes[0].bar(strategies, rewards, color='steelblue')
    axes[0].set_ylabel('Average Episode Reward')
    axes[0].set_xlabel('Strategy')
    axes[0].set_title('Average Reward by Strategy')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Win rates (if available)
    if 'win_rate' in results[strategies[0]]:
        win_rates = [results[s]['win_rate'] for s in strategies]
        axes[1].bar(strategies, win_rates, color='coral')
        axes[1].set_ylabel('Win Rate')
        axes[1].set_xlabel('Strategy')
        axes[1].set_title('Win Rate by Strategy')
        axes[1].set_ylim([0, 1])
        axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved plot to strategy_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained soccer agents"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--num-matches",
        type=int,
        default=10,
        help="Number of matches to evaluate"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Maximum steps per match"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("AGENT EVALUATION")
    print("="*80 + "\n")
    
    if args.checkpoint:
        # Evaluate single checkpoint
        results = evaluate_agent(
            args.checkpoint,
            num_matches=args.num_matches,
            max_steps=args.max_steps
        )
        
        print(f"\nResults for {args.checkpoint}:")
        print(json.dumps(results, indent=2))
    else:
        # Evaluate all strategies
        print("Evaluating all 5 strategies...")
        # TODO: Implement multi-strategy evaluation
