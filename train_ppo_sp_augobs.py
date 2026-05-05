import ray
from ray import tune
from soccer_twos import EnvType

from soccer_augmented_obs import create_augobs_env


NUM_ENVS_PER_WORKER = 3


if __name__ == "__main__":
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        _temp_dir="/tmp/ray_soccertwos",
        logging_level="ERROR",
    )

    tune.registry.register_env("SoccerAugObsSP", create_augobs_env)

    analysis = tune.run(
        "PPO",
        name="PPO_SP_AUGOBS",
        config={
            "num_gpus": 0,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "env": "SoccerAugObsSP",
            "env_config": {
                "base_port": 52000,
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "single_player": True,
                "flatten_branched": True,
                "opponent_policy": lambda *_: 0,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "gamma": 0.99,
            "lambda": 0.95,
            "lr": 3e-4,
            "clip_param": 0.2,
            "entropy_coeff": 0.001,
            "rollout_fragment_length": 500,
            "train_batch_size": 12000,
            "sgd_minibatch_size": 1024,
            "num_sgd_iter": 10,
            "batch_mode": "truncate_episodes",
        },
        stop={
            "timesteps_total": 12_000_000,
        },
        checkpoint_freq=20,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial,
        metric="episode_reward_mean",
        mode="max",
    )
    print(best_checkpoint)
    print("Done training")