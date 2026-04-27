import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class RoleBasedSoccerTwos(MultiAgentEnv):
    """
    A streamlined, hack-proof MARL environment that uses continuous 
    physics gradients and possession tracking instead of brittle heuristics.
    """
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Track who touched the ball last to give the "one-time" contact reward
        self.last_touch_agent = None

    def reset(self):
        # Reset the touch tracker at the start of a new episode
        self.last_touch_agent = None
        return self.env.reset()

    def step(self, action_dict):
        obs, rewards, dones, infos = self.env.step(action_dict)
        shaped_rewards = {}

        for agent_id, base_reward in rewards.items():
            # RULE 4 & 5: Huge reward for scoring (+1.0) and penalty for conceding (-1.0)
            # The base Unity environment natively provides exactly these values!
            r = base_reward
            
            if not dones.get("__all__", False) and agent_id in infos:
                agent_info = infos[agent_id]
                
                if 'player_info' in agent_info and 'ball_info' in agent_info:
                    player_pos = np.array(agent_info['player_info']['position'])
                    ball_pos = np.array(agent_info['ball_info']['position'])
                    
                    # Exact distance from the agent to the ball
                    ball_dist = np.linalg.norm(ball_pos - player_pos)
                    
                    # Determine Attack Direction (+1 for Blue, -1 for Orange)
                    is_team_0 = agent_id in [0, 1]
                    attack_dir = 1.0 if is_team_0 else -1.0
                    
                    # =======================================================
                    # RULE 1: Ball distance to enemy goal (Curved Pull)
                    # =======================================================
                    # We define the enemy goal at X=0 (Center), Y=14.0 (End of field)
                    # This Euclidean math creates a curve that penalizes being on the sidelines!
                    enemy_goal_pos = np.array([0.0, 14.0 * attack_dir])
                    ball_to_goal_dist = np.linalg.norm(enemy_goal_pos - ball_pos)
                    
                    # Reward increases linearly the closer the ball gets to the goal center
                    # (30.0 is roughly the max diagonal distance of the field)
                    r += 0.0002 * (30.0 - ball_to_goal_dist)
                    
                    # =======================================================
                    # RULE 2: One-time positive reward for making contact
                    # =======================================================
                    # If the agent is touching the ball, and wasn't the last one to touch it
                    if ball_dist < 1.5: # 1.5 is a safe hit radius in Unity
                        if self.last_touch_agent != agent_id:
                            r += 0.1  # Big one-time burst for gaining possession!
                            self.last_touch_agent = agent_id
                            
                    # =======================================================
                    # RULE 3: Reward for being BEHIND the ball
                    # =======================================================
                    # Player is strictly behind if their forward-axis is less advanced than the ball's
                    is_behind_ball = (player_pos[1] * attack_dir) < (ball_pos[1] * attack_dir)
                    
                    if is_behind_ball:
                        # STRONG linear gradient pulling the agent toward the ball
                        r += 0.0005 * (30.0 - ball_dist)
                    else:
                        # Penalty for being in front of the ball (forces them to run back around it)
                        r -= 0.0005

            shaped_rewards[agent_id] = r

        return obs, shaped_rewards, dones, infos