"""
Custom environment wrapper for reward shaping.
Wraps the soccer_twos environment with custom reward functions.
"""

import gym
from typing import Dict, Tuple, Any
import numpy as np
from reward_shaping import get_reward_shaper, STRATEGIES


class RewardShapingWrapper(gym.Wrapper):
    """
    Wraps soccer_twos environment with custom reward shaping.
    Applies shaped rewards based on selected strategy.
    """
    
    def __init__(self, env, strategy_id: int = 1):
        """
        Initialize the wrapper.
        
        Args:
            env: The base soccer_twos environment
            strategy_id: Which reward strategy to use (1-5)
        """
        super().__init__(env)
        self.strategy_id = strategy_id
        self.reward_shapers = {}
        self.last_observations = {}
        self.last_actions = {}
        
    def _init_shapers(self, agents):
        """Initialize reward shapers for all agents."""
        for agent_id in agents:
            if agent_id not in self.reward_shapers:
                self.reward_shapers[agent_id] = get_reward_shaper(
                    self.strategy_id, agent_id
                )
    
    def reset(self):
        """Reset environment and shapers."""
        obs = self.env.reset()
        if isinstance(obs, dict):
            self._init_shapers(obs.keys())
            for agent_id in obs:
                self.last_observations[agent_id] = obs[agent_id]
        return obs
    
    def step(self, actions: Dict[int, np.ndarray]) -> Tuple:
        """
        Step environment with reward shaping.
        
        Args:
            actions: Dictionary of {agent_id: action}
            
        Returns:
            obs, rewards, dones, infos (all with shaped rewards)
        """
        # Store actions for reward computation
        self.last_actions = actions.copy()
        
        # Get base environment step
        obs, rewards, dones, infos = self.env.step(actions)
        
        # Apply reward shaping
        # Handle both dict and scalar rewards
        if isinstance(rewards, dict):
            shaped_rewards = {}
            for agent_id in rewards:
                if agent_id not in self.reward_shapers:
                    self.reward_shapers[agent_id] = get_reward_shaper(
                        self.strategy_id, agent_id
                    )
                
                # Get current observation for this agent
                agent_obs = obs[agent_id] if isinstance(obs, dict) else obs
                agent_action = actions[agent_id] if isinstance(actions, dict) else actions
                base_reward = rewards[agent_id]
                done = dones[agent_id] if isinstance(dones, dict) else dones
                
                # Compute shaped reward
                shaped_reward = self.reward_shapers[agent_id].compute_reward(
                    agent_obs, agent_action, base_reward, done
                )
                
                shaped_rewards[agent_id] = shaped_reward
                
                # Store observation for next step
                self.last_observations[agent_id] = agent_obs
        else:
            # Scalar reward - treat as single agent with id 0
            agent_id = 0
            if agent_id not in self.reward_shapers:
                self.reward_shapers[agent_id] = get_reward_shaper(
                    self.strategy_id, agent_id
                )
            
            agent_obs = obs
            agent_action = actions[0] if isinstance(actions, dict) else actions
            base_reward = float(rewards)
            done = dones
            
            # Compute shaped reward
            shaped_reward = self.reward_shapers[agent_id].compute_reward(
                agent_obs, agent_action, base_reward, done
            )
            
            # Return as scalar
            shaped_rewards = shaped_reward
            
            # Store observation
            self.last_observations[agent_id] = agent_obs
        
        return obs, shaped_rewards, dones, infos


def create_shaped_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment with reward shaping.
    
    Args:
        env_config: Base environment configuration
                   Must include 'strategy_id' (1-5) for reward shaping
    """
    import soccer_twos
    from utils import RLLibWrapper, create_rllib_env
    
    # Extract strategy_id from config
    strategy_id = env_config.pop('strategy_id', 1)
    
    # Create base environment
    base_env = create_rllib_env(env_config)
    
    # Apply reward shaping wrapper
    shaped_env = RewardShapingWrapper(base_env, strategy_id=strategy_id)
    
    return shaped_env
