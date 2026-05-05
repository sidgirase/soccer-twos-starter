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
    This wrapper class automatically detects if your checkpoint is an older 
    1-Brain (shared_policy) run or a newer 2-Brain (MARL) run and loads it correctly.
    """
    def __init__(self, env=None):
        super().__init__()
        self.name = "Reward Raiders"
        
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False, local_mode=True)

        # Using raw string (r"") to prevent Windows path escaping issues (like \r)
        checkpoint_path = os.path.join(
            os.path.dirname(__file__),     
            # r".\checkpoints\conventional_rewards_baseline\checkpoint_001250\checkpoint-1250"
            r".\checkpoints\marl\checkpoint_001600\checkpoint-1600"
            # r".\checkpoints\vel_based\checkpoint_001300\checkpoint-1300"
        )

        # Register the fake environment
        register_env("DummyEnv", lambda config: DummyEnv(config))
        
        # Hardcode spaces so we don't have to launch Unity to read them
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
        act_space = gym.spaces.MultiDiscrete([3, 3, 3])

        # Base configuration shared between both loading attempts
        base_config = {
            "env": "DummyEnv",
            "framework": "torch",
            "num_workers": 0,
            "explore": False, # Turn off exploration during evaluation
        }

        # --- ATTEMPT 1: Load as 2-Brain MARL ---
        config_marl = base_config.copy()
        config_marl["multiagent"] = {
            "policies": {
                "attacker_policy": (None, obs_space, act_space, {}),
                "defender_policy": (None, obs_space, act_space, {}),
            },
            "policy_mapping_fn": lambda agent_id, *args, **kwargs: 
                "attacker_policy" if agent_id in [0, 2] else "defender_policy", # We can try attacker_policy for defenders too since it's just a checkpoint loading test
        }

        self.trainer = PPOTrainer(config=config_marl, env="DummyEnv")
        self.is_loaded = False
        self.is_marl = True # Assume MARL first
        
        try:
            self.trainer.restore(checkpoint_path)
            self.is_loaded = True
            print("Successfully loaded Reward Raiders as a MARL (2-Brain) agent!")
            
        except Exception as e_marl:
            print("Checkpoint is not a 2-Brain model. Falling back to Shared Policy (1-Brain)...")
            
            # --- ATTEMPT 2: Load as 1-Brain Shared Policy ---
            config_shared = base_config.copy()
            config_shared["multiagent"] = {
                "policies": {"shared_policy": (None, obs_space, act_space, {})},
                "policy_mapping_fn": lambda agent_id, *args, **kwargs: "shared_policy",
            }
            
            # Recreate trainer with the shared config
            self.trainer = PPOTrainer(config=config_shared, env="DummyEnv")
            try:
                self.trainer.restore(checkpoint_path)
                self.is_loaded = True
                self.is_marl = False # Correct flag for the act() method
                self.name = "Reward Raiders (Classic)"
                print("Successfully loaded Reward Raiders as a SHARED (1-Brain) agent!")
                
            except Exception as e_shared:
                print(f"==================================================")
                print(f"CRITICAL WARNING: Could not load checkpoint at all!")
                print(f"Check your path: {checkpoint_path}")
                print(f"Error 1 (MARL): {e_marl}")
                print(f"Error 2 (Shared): {e_shared}")
                print(f"==================================================")

    def act(self, observation):
        """
        Takes a dictionary of observations and returns actions using the correct brain architecture.
        """
        actions = {}
        
        # If the model failed to load entirely, return "Do Nothing" actions
        if not self.is_loaded:
            for player_id in observation:
                actions[player_id] = np.array([0, 0, 0])
            return actions
            
        # Loop through each player's observation and get their action
        for player_id in observation:
            if self.is_marl:
                # 2-Brain Logic: Agents 0/2 Attack, 1/3 Defend
                brain_to_use = "attacker_policy" if player_id in [0, 2] else "defender_policy" # We can use attacker_policy for defenders too since it's just a checkpoint loading test
            else:
                # 1-Brain Logic: All agents use the same policy
                brain_to_use = "shared_policy"
            
            # Get the specific Policy object from the Trainer
            policy = self.trainer.get_policy(brain_to_use)
            
            # compute_single_action on the Policy object returns a tuple, so we unpack it
            action, *_ = policy.compute_single_action(observation[player_id])
            actions[player_id] = action
            
        return actions