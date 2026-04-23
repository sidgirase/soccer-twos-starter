import os
import ray
from ray.rllib.agents.ppo import PPOTrainer

class RewardRaidersAgent:
    """
    This is the wrapper class for your trained agent.
    It loads the model checkpoint and provides actions during evaluation.
    """
    def __init__(self):
        # We need Ray to load the RLLib model, but we don't want it to spin up a full cluster
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)

        # 1. Update this path to point to your best checkpoint folder!
        # Example: "ray_results/PPO_Soccer_Final_balanced/.../checkpoint_000500/checkpoint-500"
        checkpoint_path = os.path.join(
            os.path.dirname(__file__), 
            "YOUR_CHECKPOINT_PATH_HERE" 
        )

        # 2. Minimal config to restore the trainer
        config = {
            "env": "soccer_twos",
            "framework": "torch",
            "num_workers": 0,
            "explore": False, # Turn off exploration during evaluation
            "multiagent": {
                "policies": {"shared_policy": (None, None, None, {})},
                "policy_mapping_fn": lambda agent_id: "shared_policy",
            },
        }

        # 3. Load the model
        self.trainer = PPOTrainer(config=config, env="soccer_twos")
        try:
            self.trainer.restore(checkpoint_path)
            print("Successfully loaded Reward Raiders agent checkpoint.")
        except Exception as e:
            print(f"Warning: Could not load checkpoint. Check the path. Error: {e}")

    def act(self, observation):
        """
        Takes an observation and returns an action.
        """
        # compute_single_action returns the action array for the given policy
        action = self.trainer.compute_single_action(
            observation, 
            policy_id="shared_policy"
        )
        return action