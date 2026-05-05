import glob
import os

import numpy as np
import ray
from gym_unity.envs import ActionFlattener
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from soccer_twos import AgentInterface, EnvType

from .soccer_augmented_obs import create_augobs_env


class TeamAgent(AgentInterface):
    def __init__(self, env):
        super().__init__()
        self.name = "YOURTEAM_AUGOBS"
        self.package_dir = os.path.dirname(os.path.abspath(__file__))
        self.flattener = ActionFlattener(env.action_space.nvec)
        self.prev_obs = {}

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                log_to_driver=False,
                local_mode=True,
            )

        register_env("SoccerEvalAugObs", create_augobs_env)

        config = {
            "num_workers": 0,
            "num_gpus": 0,
            "framework": "torch",
            "env": "SoccerEvalAugObs",
            "env_config": {
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
        }

        self.trainer = PPOTrainer(config=config)
        self.trainer.restore(self._find_checkpoint())

    def _find_checkpoint(self):
        candidates = sorted(
            glob.glob(os.path.join(self.package_dir, "checkpoint_*", "checkpoint-*"))
        )
        if not candidates:
            raise FileNotFoundError("No checkpoint file found inside agent package.")
        return candidates[-1]

    def _augment(self, player_id, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        prev = self.prev_obs.get(player_id)
        if prev is None:
            prev = np.zeros_like(obs, dtype=np.float32)
        delta = obs - prev
        delta_norm = np.array([np.linalg.norm(delta)], dtype=np.float32)
        aug = np.concatenate([obs, prev, delta, delta_norm], axis=0).astype(np.float32)
        self.prev_obs[player_id] = obs.copy()
        return aug

    def act(self, observation):
        actions = {}
        for player_id, obs in observation.items():
            aug_obs = self._augment(player_id, obs)
            action_idx = self.trainer.compute_action(aug_obs)
            actions[player_id] = self.flattener.lookup_action(int(action_idx))
        return actions