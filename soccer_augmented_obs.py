import numpy as np
import gym
import soccer_twos


class TemporalObservationWrapper(gym.Wrapper):
    """
    Augments the 336-dim observation with:
      - previous observation
      - one-step delta
      - delta norm
    final dim = 336 + 336 + 336 + 1 = 1009

    This is an explicit observation-space modification for the assignment.
    """

    def __init__(self, env):
        super().__init__(env)

        if len(env.observation_space.shape) != 1:
            raise ValueError(
                f"Expected flat vector observation, got {env.observation_space.shape}"
            )

        self.base_dim = env.observation_space.shape[0]
        self.aug_dim = self.base_dim * 3 + 1

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.aug_dim,),
            dtype=np.float32,
        )

        self.prev_obs = np.zeros(self.base_dim, dtype=np.float32)

    def _augment(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        delta = obs - self.prev_obs
        delta_norm = np.array([np.linalg.norm(delta)], dtype=np.float32)
        aug = np.concatenate([obs, self.prev_obs, delta, delta_norm], axis=0)
        self.prev_obs = obs.copy()
        return aug.astype(np.float32)

    def reset(self):
        obs = self.env.reset()
        self.prev_obs = np.zeros(self.base_dim, dtype=np.float32)
        return self._augment(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._augment(obs), reward, done, info


def create_augobs_env(env_config=None):
    if env_config is None:
        env_config = {}

    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )

    env = soccer_twos.make(**env_config)
    env = TemporalObservationWrapper(env)
    return env