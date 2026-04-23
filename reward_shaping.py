"""
Reward shaping module for soccer agents.
Implements 5 different reward strategies for multi-agent soccer training.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Optional


class RewardShaper:
    """
    Base class for reward shaping strategies.
    Tracks metrics and computes shaped rewards.
    """

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.prev_ball_distance = None
        self.prev_ball_held_by_agent = False
        self.ball_contact_time = 0
        self.interception_count = 0
        self.shots_attempted = 0
        self.shots_on_target = 0
        self.pass_attempts = 0
        self.successful_passes = 0
        self.blocks_attempted = 0
        self.successful_blocks = 0
        
    def extract_observations(self, obs: np.ndarray) -> Dict:
        """
        Extract meaningful observations from raw array.
        Note: Soccer-Twos observations are high-dimensional (672 dims).
        For reward shaping, we primarily use distance metrics derived from obs.
        """
        # For simplicity, we'll just store the full observation
        # Distance calculations will use the fact that certain dimensions
        # encode position information
        
        # In soccer_twos, key indices (approximate):
        # - Agent position: typically early indices (x, y, z)
        # - Ball position: encoded relative to agent
        # - Goal positions: encoded in the observation
        # - Other agents: relative positions
        
        # For now, return simplified metrics we can compute
        return {
            'agent_pos': obs[:3] if len(obs) >= 3 else np.zeros(3),
            'agent_vel': obs[3:6] if len(obs) >= 6 else np.zeros(3),
            'ball_pos_rel': obs[6:9] if len(obs) >= 9 else np.zeros(3),
            'ball_vel': obs[9:12] if len(obs) >= 12 else np.zeros(3),
            'full_obs': obs,  # Keep full observation for complex metrics
        }
    
    def update_metrics(self, obs: np.ndarray, action: np.ndarray, 
                      base_reward: float, done: bool):
        """Update internal metrics based on current observation and action."""
        metrics = self.extract_observations(obs)
        ball_dist = np.linalg.norm(metrics['ball_pos_rel'])
        
        # Track ball possession (agent close to ball)
        ball_held = ball_dist < 1.5
        if ball_held and not self.prev_ball_held_by_agent:
            self.interception_count += 1
        self.prev_ball_held_by_agent = ball_held
        
        if ball_held:
            self.ball_contact_time += 1
        
        # Track shot attempts (high forward action + ball near agent)
        if ball_held and np.any(np.abs(action) > 0.7):
            self.shots_attempted += 1
            
        self.prev_ball_distance = ball_dist
        
        if done:
            self._reset_episode_metrics()
    
    def _reset_episode_metrics(self):
        """Reset episode-specific metrics."""
        self.ball_contact_time = 0
        self.interception_count = 0
        self.shots_attempted = 0
        self.shots_on_target = 0
        self.pass_attempts = 0
        self.successful_passes = 0

    def compute_reward(self, obs: np.ndarray, action: np.ndarray, 
                      base_reward: float, done: bool) -> float:
        """
        Compute shaped reward. Must be overridden by subclasses.
        
        Args:
            obs: Current observation
            action: Current action taken
            base_reward: Reward from environment (goals, etc.)
            done: Whether episode is done
            
        Returns:
            Shaped reward value
        """
        # Base implementation: just scale the goal reward
        # Subclasses should implement more sophisticated shaping
        return base_reward * 10.0 if base_reward > 0 else base_reward


class Strategy1_BallPossession(RewardShaper):
    """
    Strategy 1: Ball Possession Focus
    Rewards agent for maintaining ball possession and close control.
    Encourages players to stay near ball and maintain possession.
    """
    
    def compute_reward(self, obs: np.ndarray, action: np.ndarray, 
                      base_reward: float, done: bool) -> float:
        self.update_metrics(obs, action, base_reward, done)
        metrics = self.extract_observations(obs)
        
        shaped_reward = base_reward * 10.0  # Goals worth 10x
        
        # Reward for getting closer to ball
        ball_dist = np.linalg.norm(metrics['ball_pos_rel'])
        if self.prev_ball_distance is not None:
            if ball_dist < self.prev_ball_distance:
                shaped_reward += 0.1  # Small reward for moving toward ball
        
        # Reward for possession (ball very close)
        if ball_dist < 2.0:
            shaped_reward += 0.3
        
        # Small penalty for being far from ball
        if ball_dist > 8.0:
            shaped_reward -= 0.05
            
        return shaped_reward


class Strategy2_BallInterception(RewardShaper):
    """
    Strategy 2: Ball Interception Focus
    Rewards agent for intercepting the ball and regaining possession.
    Encourages aggressive defense and quick ball recovery.
    """
    
    def compute_reward(self, obs: np.ndarray, action: np.ndarray, 
                      base_reward: float, done: bool) -> float:
        self.update_metrics(obs, action, base_reward, done)
        metrics = self.extract_observations(obs)
        
        shaped_reward = base_reward * 10.0  # Goals worth 10x
        
        ball_dist = np.linalg.norm(metrics['ball_pos_rel'])
        
        # Major reward for interception (getting ball when far away)
        if self.prev_ball_distance is not None:
            if ball_dist < 1.5 and self.prev_ball_distance > 5.0:
                shaped_reward += 2.0  # Big reward for successful interception
        
        # Reward for moving toward ball for interception
        if self.prev_ball_distance is not None:
            if ball_dist < self.prev_ball_distance and ball_dist < 4.0:
                shaped_reward += 0.15
        
        # Reward for tight ball control after interception
        if ball_dist < 1.2:
            shaped_reward += 0.15
            
        return shaped_reward


class Strategy3_OffensivePositioning(RewardShaper):
    """
    Strategy 3: Offensive Positioning Focus
    Rewards agent for dynamic ball movement and aggressive actions.
    Encourages active play and forward movement.
    """
    
    def compute_reward(self, obs: np.ndarray, action: np.ndarray, 
                      base_reward: float, done: bool) -> float:
        self.update_metrics(obs, action, base_reward, done)
        metrics = self.extract_observations(obs)
        
        shaped_reward = base_reward * 10.0  # Goals worth 10x
        
        ball_dist = np.linalg.norm(metrics['ball_pos_rel'])
        action_magnitude = np.linalg.norm(action)
        
        # Reward for aggressive actions when near ball
        if ball_dist < 3.0 and action_magnitude > 0.5:
            shaped_reward += 0.2
        
        # Reward for continuous movement (active play)
        if action_magnitude > 0.3:
            shaped_reward += 0.1
        
        # Reward for moving ball forward/outward
        if ball_dist < 2.5 and self.prev_ball_distance is not None:
            if ball_dist < self.prev_ball_distance:
                shaped_reward += 0.2
        
        return shaped_reward


class Strategy4_DefensiveBlocking(RewardShaper):
    """
    Strategy 4: Defensive Blocking Focus
    Rewards agent for intercepting incoming balls and regaining possession.
    Encourages defensive pressing and blocking.
    """
    
    def compute_reward(self, obs: np.ndarray, action: np.ndarray, 
                      base_reward: float, done: bool) -> float:
        self.update_metrics(obs, action, base_reward, done)
        metrics = self.extract_observations(obs)
        
        shaped_reward = base_reward * 10.0  # Goals worth 10x
        
        ball_dist = np.linalg.norm(metrics['ball_pos_rel'])
        action_magnitude = np.linalg.norm(action)
        
        # Reward for blocking/intercepting
        if ball_dist < 2.5:
            shaped_reward += 0.2
        
        # Reward for defensive actions (movement + positioning)
        if ball_dist < 3.0 and action_magnitude > 0.4:
            shaped_reward += 0.15
        
        # Reward for quickly getting ball after losing it
        if self.prev_ball_distance is not None and self.prev_ball_distance > 3.0:
            if ball_dist < 2.0:
                shaped_reward += 0.3  # Regained possession
        
        return shaped_reward


class Strategy5_CoordinatedTeamplay(RewardShaper):
    """
    Strategy 5: Coordinated Teamplay Focus
    Rewards agent for coordinating with teammates.
    Encourages passing, spacing, and team coordination.
    """
    
    def compute_reward(self, obs: np.ndarray, action: np.ndarray, 
                      base_reward: float, done: bool) -> float:
        self.update_metrics(obs, action, base_reward, done)
        metrics = self.extract_observations(obs)
        
        shaped_reward = base_reward * 10  # Goals worth 10x
        
        ball_dist = np.linalg.norm(metrics['ball_pos_rel'])
        teammate_dist = np.linalg.norm(metrics['teammate_pos_rel'])
        
        # Reward for good positioning relative to teammate
        if 2.0 < teammate_dist < 6.0:  # Good spacing
            shaped_reward += 0.2
        
        # Reward for ball possession to make pass
        if ball_dist < 1.5:
            shaped_reward += 0.15
            
            # Reward for being in position to pass/assist
            if 2.0 < teammate_dist < 5.0:
                shaped_reward += 0.2  # Passing position reward
        
        # Reward for supportive positioning when teammate has ball
        if ball_dist > 2.0:
            if 3.0 < teammate_dist < 6.0:
                shaped_reward += 0.15  # Support positioning
        
        # Bonus for triangulation (good offensive spacing)
        opp1_dist = np.linalg.norm(metrics['opp1_pos_rel'])
        if teammate_dist < 4.0 and opp1_dist > 3.0:
            shaped_reward += 0.1  # Good triangle formation
        
        return shaped_reward


# Factory function to get the right strategy
STRATEGIES = {
    1: Strategy1_BallPossession,
    2: Strategy2_BallInterception,
    3: Strategy3_OffensivePositioning,
    4: Strategy4_DefensiveBlocking,
    5: Strategy5_CoordinatedTeamplay,
}


def get_reward_shaper(strategy_id: int, agent_id: int) -> RewardShaper:
    """Get reward shaper for the given strategy."""
    if strategy_id not in STRATEGIES:
        raise ValueError(f"Strategy {strategy_id} not found. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[strategy_id](agent_id)


# Strategy descriptions for documentation
STRATEGY_DESCRIPTIONS = {
    1: "Ball Possession Focus - Maintains possession, close ball control",
    2: "Ball Interception Focus - Aggressive defense, quick recovery",
    3: "Offensive Positioning - Attacking moves, goal proximity",
    4: "Defensive Blocking - Shot blocking, defensive positioning",
    5: "Coordinated Teamplay - Passing, team spacing, support",
}
