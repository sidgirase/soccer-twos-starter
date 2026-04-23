import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class ShapedSoccerTwos(MultiAgentEnv):
    """
    A Ray RLLib compatible Multi-Agent Environment Wrapper that applies
    custom reward shaping based on 3 core strategies.
    """
    def __init__(self, env, strategy="balanced"):
        super().__init__()
        self.env = env
        self.strategy = strategy
        
        # Expose spaces for Ray
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action_dict):
        obs, rewards, dones, infos = self.env.step(action_dict)
        shaped_rewards = {}

        # The state vector length is typically 336 in Soccer Twos.
        # The last 6 elements contain relative positional/velocity data:
        # [-6]: ball relative x
        # [-5]: ball relative y
        # [-4]: ball velocity x
        # [-3]: ball velocity y
        # [-2]: agent velocity x
        # [-1]: agent velocity y

        for agent_id, base_reward in rewards.items():
            r = base_reward
            
            # Apply shaping only if the game is still ongoing for this step
            if not dones.get("__all__", False) and len(obs[agent_id]) >= 336:
                agent_obs = obs[agent_id]
                ball_rel_x = agent_obs[-6]
                ball_rel_y = agent_obs[-5]
                agent_vel_x = agent_obs[-2]
                
                # Distance to the ball (heuristic)
                ball_dist = np.sqrt(ball_rel_x**2 + ball_rel_y**2)
                
                if self.strategy == "offensive":
                    # STRATEGY 1: Offensive Positioning
                    # Rewards advancing and staying close to the ball
                    r += 0.005 * (1.0 - min(ball_dist, 1.0)) # Close control bonus
                    if agent_vel_x > 0.1: 
                        r += 0.002  # Bonus for moving forward 
                        
                elif self.strategy == "defensive":
                    # STRATEGY 2: Defensive Blocking
                    # Rewards staying back and intercepting
                    if agent_vel_x < -0.1:
                        r += 0.002 # Bonus for tracking back
                    # Slight penalty for wandering too far from the ball if it's nearby
                    if ball_dist < 2.0:
                        r += 0.01 * (2.0 - ball_dist)
                        
                elif self.strategy == "balanced":
                    # STRATEGY 3: Coordinated Teamplay (RECOMMENDED)
                    # Rewards good spacing and support
                    # We give a moderate bonus for being engaged with the ball
                    r += 0.005 * (1.0 - min(ball_dist, 1.0))
                    
                    # Apply a small step penalty to encourage faster goal scoring
                    # and prevent the agents from just standing still to hoard proximity points
                    r -= 0.0005

            shaped_rewards[agent_id] = r

        return obs, shaped_rewards, dones, infos