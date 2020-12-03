import abc

import dm_env
import numpy as np
from dm_env import specs


class Physics:
    def __init__(self):
        self._p = None

    def close(self):
        del self._p

    def step(self):
        self._p.stepSimulation()


class Task(metaclass=abc.ABCMeta):
    def __init__(self, random=None) -> None:
        if not isinstance(random, np.random.RandomState):
            random = np.random.RandomState(random)
        self._random = random

    @property
    def random(self):
        return self._random

    @abc.abstractmethod
    def initialize_episode(self, physics):
        pass

    @abc.abstractmethod
    def before_step(self, action, physics: Physics) -> None:
        pass

    def after_step(self, physics: Physics) -> None:
        pass

    @abc.abstractmethod
    def observation_spec(self, physics: Physics):
        pass

    @abc.abstractmethod
    def reward_spec(self, physics: Physics):
        pass

    @abc.abstractmethod
    def action_spec(self, physics: Physics):
        pass

    @abc.abstractmethod
    def get_observation(self, physics: Physics):
        pass

    @abc.abstractmethod
    def get_reward(self, physics: Physics):
        pass

    def get_termination(self, physics: Physics):
        pass


class Environment:
    def __init__(self, physics: Physics, task: Task):
        self.physics = physics
        self.task = task
        self._reset_next_step = True

    def reset(self) -> dm_env.TimeStep:
        # self.physics._p.resetSimulation()
        self._reset_next_step = False
        self.physics._p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        self.task.initialize_episode(self.physics)
        observation = self.task.get_observation(self.physics)
        return dm_env.TimeStep(
            dm_env.StepType.FIRST, reward=None, discount=None, observation=observation
        )

    def step(self, action) -> dm_env.TimeStep:
        if self._reset_next_step:
            return self.reset()
        self.task.before_step(action, self.physics)
        self.physics.step()
        self.task.after_step(self.physics)
        reward = self.task.get_reward(self.physics)
        observation = self.task.get_observation(self.physics)
        # TODO handle terminal step
        return dm_env.TimeStep(dm_env.StepType.MID, reward, 1.0, observation)

    def observation_spec(self):
        return self.task.observation_spec(self.physics)

    def action_spec(self):
        return self.task.action_spec(self.physics)

    def reward_spec(self):
        return self.task.reward_spec(self.physics)

    def discount_spec(self):
        return specs.Array((), np.float64)

    def close(self):
        self.physics.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Allows the environment to be used in a with-statement context."""
        del exc_type, exc_value, traceback  # Unused.
        self.close()
