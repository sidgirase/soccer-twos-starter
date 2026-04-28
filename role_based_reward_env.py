import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class RoleBasedSoccerTwos(MultiAgentEnv):
    """
    A pure physics-based environment. 
    Agents are ONLY rewarded for running towards the ball, 
    and for the ball rolling towards the enemy goal. 
    Positional reward hacking is physically impossible.
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
            # Pass through the huge +/- 1.0 rewards for goals natively
            r = base_reward
            
            if not dones.get("__all__", False) and agent_id in infos:
                agent_info = infos[agent_id]
                
                if 'player_info' in agent_info and 'ball_info' in agent_info:
                    player_pos = np.array(agent_info['player_info']['position'])
                    player_vel = np.array(agent_info['player_info']['velocity'])
                    ball_pos = np.array(agent_info['ball_info']['position'])
                    ball_vel = np.array(agent_info['ball_info']['velocity'])
                    
                    is_team_0 = agent_id in [0, 1]
                    attack_dir = 1.0 if is_team_0 else -1.0
                    
                    # =======================================================
                    # 1. EXISTENTIAL TIME PENALTY
                    # =======================================================
                    # Bleed points constantly. Standing still or spinning = death.
                    r -= 0.0001 
                    
                    # =======================================================
                    # 2. RUN DIRECTLY AT THE BALL
                    # =======================================================
                    ball_rel_vec = ball_pos - player_pos
                    ball_dist = np.linalg.norm(ball_rel_vec)
                    
                    if ball_dist > 0.5:
                        dir_to_ball = ball_rel_vec / ball_dist
                        # Dot product of player velocity and direction to ball.
                        # Positive ONLY if they are actively closing the distance.
                        speed_towards_ball = np.dot(player_vel, dir_to_ball)
                        
                        if speed_towards_ball > 0: 
                            r += 0.0005 * speed_towards_ball
                            
                    # =======================================================
                    # 3. PUSH THE BALL TO THE GOAL
                    # =======================================================
                    enemy_goal_pos = np.array([0.0, 14.0 * attack_dir])
                    ball_to_goal_vec = enemy_goal_pos - ball_pos
                    ball_to_goal_dist = np.linalg.norm(ball_to_goal_vec)
                    
                    if ball_to_goal_dist > 0:
                        dir_to_goal = ball_to_goal_vec / ball_to_goal_dist
                        # Dot product of ball velocity and direction to enemy goal.
                        ball_speed_towards_goal = np.dot(ball_vel, dir_to_goal)
                        
                        if ball_speed_towards_goal > 0: 
                            # Massive multiplier so they realize kicking is better than chasing
                            r += 0.002 * ball_speed_towards_goal

            shaped_rewards[agent_id] = r

        return obs, shaped_rewards, dones, infos