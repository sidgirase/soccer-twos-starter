from random import uniform as randfloat

import gym
from ray.rllib import MultiAgentEnv
import soccer_twos
import numpy as np


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    def __init__(self, env):
        gym.core.Wrapper.__init__(self, env)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        observations, rewards, done, info = self.env.step(action)
        new_rewards = {}

        for agent_id, reward in rewards.items():
            new_reward = reward
            
            if not dones.get("__all__", False) and len(observations[agent_id]) >= 336:
                agent_observations = observations[agent_id]
                ball_relative_position_x = agent_observations[-6]
                ball_relative_position_y = agent_observations[-5]
                agent_velocity_x = agent_observations[-2]

                # comprehensive strategy encouraging agents to get close to the ball and to score as quickly as possible
                distance_to_ball = np.sqrt(ball_relative_position_x**2 + ball_relative_position_y**2) # Euclidean distance
                new_reward -= 0.001 * min(distance_to_ball, 1.0)
                new_reward -= 0.001 # encourage agent to score as quickly as possible (minimize time)

            new_rewards[agent_id] = new_reward

        return obs, new_rewards, done, info

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return None


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
