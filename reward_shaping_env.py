import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class ShapedSoccerTwos(MultiAgentEnv):
    """
    A Ray RLLib compatible Multi-Agent Environment Wrapper that applies
    custom reward shaping based on 3 core strategies, utilizing actual 
    (x,y) coordinates and a dynamic scoreboard.
    """
    def __init__(self, env, strategy="balanced"):
        super().__init__()
        self.env = env
        self.strategy = strategy
        
        # Expose spaces for Ray
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Initialize our internal scoreboard
        self.team_0_score = 0
        self.team_1_score = 0

    def reset(self):
        # Reset the scoreboard at the beginning of every 2-minute episode
        self.team_0_score = 0
        self.team_1_score = 0
        return self.env.reset()

    def step(self, action_dict):
        obs, rewards, dones, infos = self.env.step(action_dict)
        shaped_rewards = {}

        # --- SCOREBOARD TRACKING ---
        # The base env gives ~ +1.0 for scoring and -1.0 for getting scored on.
        # Team 0 = Agents 0 & 1 (Blue) | Team 1 = Agents 2 & 3 (Orange)
        # We only need to check agent 0's raw reward to know if a goal happened!
        if 0 in rewards:
            if rewards[0] > 0.5:
                self.team_0_score += 1
            elif rewards[0] < -0.5:
                self.team_1_score += 1

        for agent_id, base_reward in rewards.items():
            r = base_reward
            
            # Apply shaping only if the game is still ongoing for this step
            if not dones.get("__all__", False) and agent_id in infos:
                agent_info = infos[agent_id]
                
                # Extract the EXACT 2D coordinates provided by the soccer_twos wrapper
                if 'player_info' in agent_info and 'ball_info' in agent_info:
                    player_pos = agent_info['player_info']['position']
                    player_vel = agent_info['player_info']['velocity']
                    ball_pos = agent_info['ball_info']['position']
                    
                    # 1. Calculate relative distance to the ball
                    ball_rel_x = ball_pos[0] - player_pos[0]
                    ball_rel_y = ball_pos[1] - player_pos[1]
                    ball_dist = np.sqrt(ball_rel_x**2 + ball_rel_y**2)
                    
                    # 2. Calculate velocity strictly TOWARDS the ball using the dot product.
                    # This completely solves the "Axis/Team Orientation" problem.
                    # Positive value = moving towards ball. Negative = moving away.
                    if ball_dist > 0:
                        vel_towards_ball = (player_vel[0] * ball_rel_x + player_vel[1] * ball_rel_y) / ball_dist
                    else:
                        vel_towards_ball = 0.0
                        
                    # 3. SCOREBOARD PANIC PENALTY
                    # Determine if this agent's team is currently losing
                    is_losing = False
                    if agent_id in [0, 1] and self.team_0_score < self.team_1_score:
                        is_losing = True
                    elif agent_id in [2, 3] and self.team_1_score < self.team_0_score:
                        is_losing = True
                        
                    # Heavily penalize the agent for every second they allow themselves to be losing
                    if is_losing:
                        r -= 0.001 
                    
                    # 4. Apply SCALED reward heuristics
                    if self.strategy == "offensive":
                        # STRATEGY 1: Offensive Positioning
                        r += 0.0005 * (1.0 - min(ball_dist, 1.0)) 
                        if vel_towards_ball > 0.5: 
                            r += 0.0002  # Reward sprinting AT the ball
                            
                    elif self.strategy == "defensive":
                        # STRATEGY 2: Defensive Blocking
                        if vel_towards_ball < -0.5: 
                            r += 0.0002  # Reward retreating / falling back
                        if ball_dist < 2.0:
                            r += 0.001 * (2.0 - ball_dist)
                            
                    elif self.strategy == "balanced":
                        # STRATEGY 3: Coordinated Teamplay 
                        r += 0.0005 * (1.0 - min(ball_dist, 1.0))
                        
                        # If they are tied or winning, apply a tiny penalty to prevent stalling
                        if not is_losing:
                            r -= 0.00005

            shaped_rewards[agent_id] = r

        return obs, shaped_rewards, dones, infos