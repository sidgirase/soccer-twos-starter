import ray
from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env

NUM_ENVS_PER_WORKER = 4

if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "PPO",
        name="PPO_hands_on",
        config={
            "num_gpus": 1,
            "num_workers": 4,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "env": "Soccer",
            "env_config": {
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "rollout_fragment_length": 2000,
            "batch_mode": "complete_episodes",
        },
        stop={"timesteps_total": 5000000},
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
