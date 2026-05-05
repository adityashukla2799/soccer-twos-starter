import glob
import os

import numpy as np
import ray
from gym_unity.envs import ActionFlattener
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

from soccer_twos import AgentInterface, EnvType
from utils import create_rllib_env


class TeamAgent(AgentInterface):
    def __init__(self, env):
        super().__init__()
        self.name = "SocLearnBaselinePPO"
        self.flattener = ActionFlattener(env.action_space.nvec)

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                log_to_driver=False,
                local_mode=True,
            )

        register_env("SoccerSubmissionEnv", create_rllib_env)

        config = {
            "num_workers": 0,
            "num_gpus": 0,
            "framework": "torch",
            "env": "SoccerSubmissionEnv",
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
        package_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = sorted(
            glob.glob(os.path.join(package_dir, "checkpoint_*", "checkpoint-*"))
        )
        candidates = [c for c in candidates if not c.endswith(".tune_metadata")]
        if not candidates:
            raise FileNotFoundError(
                "No checkpoint file found inside submission package."
            )
        return candidates[-1]

    def act(self, observation):
        actions = {}
        for player_id, obs in observation.items():
            obs = np.asarray(obs, dtype=np.float32)
            action_idx = self.trainer.compute_action(obs)
            actions[player_id] = self.flattener.lookup_action(int(action_idx))
        return actions