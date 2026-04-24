import os
import ray
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import AgentInterface
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# We create a minimal wrapper to satisfy RLlib's MultiAgentEnv requirement
class RLLibWrapper(MultiAgentEnv):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action_dict):
        return self.env.step(action_dict)

class RewardRaidersAgent(AgentInterface):
    """
    This is the wrapper class for your trained agent.
    It loads the model checkpoint and provides actions during evaluation.
    """
    def __init__(self, env=None):
        super().__init__()
        self.name = "Reward Raiders"
        
        # We need Ray to load the RLLib model, but we don't want it to spin up a full cluster
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)

        # 1. Update this path to point to your best checkpoint folder!
        # Example: "../ray_results/PPO_Soccer_Final_balanced/.../checkpoint_000500/checkpoint-500"
        checkpoint_path = os.path.join(
            os.path.dirname(__file__), 
            "..\\ray_results\\PPO_ShapedSoccer_3594d_00000_0_2026-04-23_20-11-58\\checkpoint_000250\\checkpoint-250" 
        )

        import soccer_twos
        
        # Register the environment WITH our wrapper so RLlib accepts it
        register_env("soccer_twos", lambda config: RLLibWrapper(soccer_twos.make(**config)))
        
        # Extract spaces
        dummy_env = soccer_twos.make(render=False, worker_id=98)
        obs_space = dummy_env.observation_space
        act_space = dummy_env.action_space
        dummy_env.close()

        # 2. Minimal config to restore the trainer
        config = {
            "env": "soccer_twos",
            "framework": "torch",
            "num_workers": 0,
            "explore": False, # Turn off exploration during evaluation
            "multiagent": {
                "policies": {"shared_policy": (None, obs_space, act_space, {})},
                "policy_mapping_fn": lambda agent_id, *args, **kwargs: "shared_policy",
            },
        }

        # 3. Load the model
        self.trainer = PPOTrainer(config=config, env="soccer_twos")
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