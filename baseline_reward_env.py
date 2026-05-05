import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class BaselineSoccerTwos(MultiAgentEnv):
    """
    A highly conventional, tried-and-true reward wrapper.
    Focuses purely on distance to ball and pushing the ball forward,
    ensuring standard, predictable agent learning.
    """
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action_dict):
        obs, rewards, dones, infos = self.env.step(action_dict)
        shaped_rewards = {}

        for agent_id, base_reward in rewards.items():
            r = base_reward
            
            if not dones.get("__all__", False) and agent_id in infos:
                agent_info = infos[agent_id]
                
                if 'player_info' in agent_info and 'ball_info' in agent_info:
                    player_pos = np.array(agent_info['player_info']['position'])
                    ball_pos = np.array(agent_info['ball_info']['position'])
                    
                    ball_dist = np.linalg.norm(ball_pos - player_pos)
                    
                    is_team_0 = agent_id in [0, 1]
                    attack_dir = 1.0 if is_team_0 else -1.0
                    
                    # 1. Small time penalty to prevent standing still
                    r -= 0.0001
                    
                    # 2. Gentle proximity reward (only when close, to encourage touching)
                    if ball_dist < 2.0:
                        r += 0.0002 * (2.0 - ball_dist)
                        
                    # 3. Simple forward-progress reward for the ball
                    # The deeper the ball is in enemy territory, the higher the passive reward
                    r += (ball_pos[1] * attack_dir) * 0.00005

            shaped_rewards[agent_id] = r

        return obs, shaped_rewards, dones, infos