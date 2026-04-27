import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class RoleBasedSoccerTwos(MultiAgentEnv):
    """
    Splits agents into specific roles (Attacker vs Defender) with 
    highly constrained, hack-proof reward shaping.
    """
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        self.team_0_score = 0
        self.team_1_score = 0

    def reset(self):
        self.team_0_score = 0
        self.team_1_score = 0
        return self.env.reset()

    def step(self, action_dict):
        obs, rewards, dones, infos = self.env.step(action_dict)
        shaped_rewards = {}

        # --- SCOREBOARD TRACKING ---
        if 0 in rewards:
            if rewards[0] > 0.5:
                self.team_0_score += 1
            elif rewards[0] < -0.5:
                self.team_1_score += 1

        for agent_id, base_reward in rewards.items():
            r = base_reward
            
            if not dones.get("__all__", False) and agent_id in infos:
                agent_info = infos[agent_id]
                
                if 'player_info' in agent_info and 'ball_info' in agent_info:
                    player_pos = agent_info['player_info']['position']
                    player_vel = agent_info['player_info']['velocity']
                    ball_pos = agent_info['ball_info']['position']
                    ball_vel = agent_info['ball_info']['velocity']
                    
                    # Distances and Vectors
                    ball_rel_x = ball_pos[0] - player_pos[0]
                    ball_rel_y = ball_pos[1] - player_pos[1]
                    ball_dist = np.sqrt(ball_rel_x**2 + ball_rel_y**2)
                    
                    # Determine Team and Attack Direction
                    # Unity Soccer: Index 1 (Y/Z axis) is the goal-to-goal axis.
                    # Team 0 (Blue) attacks positive (+1). Team 1 (Orange) attacks negative (-1).
                    is_team_0 = agent_id in [0, 1]
                    attack_dir = 1.0 if is_team_0 else -1.0
                    
                    # --- SCALED SCOREBOARD PENALTY ---
                    # Calculate the goal differential for this specific agent's team
                    if is_team_0:
                        score_diff = self.team_0_score - self.team_1_score
                    else:
                        score_diff = self.team_1_score - self.team_0_score
                        
                    if score_diff < 0:
                        # Losing: Heavy penalty (Panic Mode)
                        r -= 0.001 
                    elif score_diff == 0:
                        # Tied: Medium penalty (Urgency to break the tie)
                        r -= 0.0005 
                    else:
                        # Winning: Tiny penalty (Keep pressure, but no panic)
                        r -= 0.00005 

                    # ==========================================
                    # ROLE 1: ATTACKER (Agents 0 and 2)
                    # ==========================================
                    if agent_id in [0, 2]:
                        # 1. Very near the ball
                        if ball_dist < 1.0:
                            r += 0.0005 * (1.0 - ball_dist)
                            
                            # 2. Positioned strictly behind the ball (pushing it forward)
                            # E.g., if attacking +Y, player's Y must be less than ball's Y
                            if player_pos[1] * attack_dir < ball_pos[1] * attack_dir:
                                r += 0.0005
                                
                        # 3. Ball moving aggressively toward enemy goal
                        # Prevents reward hacking: only rewards if ball is moving fast!
                        forward_ball_vel = ball_vel[1] * attack_dir
                        if ball_dist < 2.0 and forward_ball_vel > 2.0:
                            r += 0.001

                    # ==========================================
                    # ROLE 2: DEFENDER / GOALKEEPER (Agents 1 and 3)
                    # ==========================================
                    elif agent_id in [1, 3]:
                        in_own_half = (ball_pos[1] * attack_dir < 0)
                        
                        # 1. Contact with ball near own goal
                        if ball_dist < 1.0 and in_own_half:
                            r += 0.0005 * (1.0 - ball_dist)
                            
                        # 2. Clearance: Ball moving AWAY from own goal
                        forward_ball_vel = ball_vel[1] * attack_dir
                        if ball_dist < 2.5 and forward_ball_vel > 2.0 and in_own_half:
                            r += 0.002 # Huge reward for successful clearance
                            
                        # 3. Passing to Attacker
                        # Find teammate attacker (0 if I am 1, 2 if I am 3)
                        attacker_id = 0 if agent_id == 1 else 2
                        if attacker_id in infos and ball_dist < 2.0 and forward_ball_vel > 1.0:
                            att_pos = infos[attacker_id]['player_info']['position']
                            vec_to_att = np.array(att_pos) - np.array(player_pos)
                            dist_to_att = np.linalg.norm(vec_to_att)
                            if dist_to_att > 0.1:
                                dir_to_att = vec_to_att / dist_to_att
                                pass_accuracy = np.dot(ball_vel, dir_to_att)
                                if pass_accuracy > 3.0: # Ball moving fast & accurately toward attacker
                                    r += 0.002
                        
                        # ANTI-HACK: Penalty if the defender leaves their own half!
                        if player_pos[1] * attack_dir > 2.0:
                            r -= 0.0005

            shaped_rewards[agent_id] = r

        return obs, shaped_rewards, dones, infos