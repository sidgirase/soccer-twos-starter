import os
import gym
import numpy as np

os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"
os.environ["RAY_DISABLE_METRICS_COLLECTION"] = "1"
os.environ["RAY_DISABLE_REPORTER"] = "1"
os.environ["RAY_DISABLE_DASHBOARD"] = "1"
os.environ["RAY_NODE_IP_ADDRESS"] = "127.0.0.1"

import ray
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import AgentInterface
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# We create a fake environment to trick RLlib. 
class DummyEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 3])

    def reset(self):
        return {0: self.observation_space.sample()}

    def step(self, action_dict):
        return {0: self.observation_space.sample()}, {0: 0}, {0: False, "__all__": False}, {}

class RewardRaidersAgent(AgentInterface):
    """
    This is the wrapper class for your trained agent.
    It loads the model checkpoint and provides actions during evaluation.
    """
    def __init__(self, env=None):
        super().__init__()
        self.name = "Reward Raiders"
        
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False, local_mode=True)

        checkpoint_path = os.path.join(
            os.path.dirname(__file__), 
            "checkpoint_001200", 
            "checkpoint-1200"
        )

        # Register the fake environment
        register_env("DummyEnv", lambda config: DummyEnv(config))
        
        # Hardcode spaces so we don't have to launch Unity to read them
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
        act_space = gym.spaces.MultiDiscrete([3, 3, 3])

        # 2. Minimal config to restore the trainer using the DummyEnv
        config = {
            "env": "DummyEnv",
            "framework": "torch",
            "num_workers": 0,
            "explore": False, # Turn off exploration during evaluation
            "multiagent": {
                "policies": {"shared_policy": (None, obs_space, act_space, {})},
                "policy_mapping_fn": lambda agent_id, *args, **kwargs: "shared_policy",
            },
        }

        # 3. Load the model
        self.trainer = PPOTrainer(config=config, env="DummyEnv")
        try:
            self.trainer.restore(checkpoint_path)
            # Extract the specific policy so we can query it directly
            self.policy = self.trainer.get_policy("shared_policy")
            print("Successfully loaded Reward Raiders agent checkpoint.")
        except Exception as e:
            print(f"Warning: Could not load checkpoint. Check the path. Error: {e}")
            self.policy = None

    def act(self, observation):
        """
        Takes a dictionary of observations (one for each player on our team)
        and returns a dictionary of actions.
        """
        actions = {}
        if self.policy is None:
            return actions
            
        # Loop through each player's observation and get their action
        for player_id in observation:
            actions[player_id], *_ = self.policy.compute_single_action(
                observation[player_id]
            )
            
        return actions