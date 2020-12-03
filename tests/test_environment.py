import dm_env
import numpy as np
from absl.testing import absltest, parameterized
from bullet_control.tasks import pendulum
from dm_env import test_utils


def uniform_random_policy(action_spec, random=None):
    random_state = np.random.RandomState(random)

    def policy(time_step):
        del time_step  # Unused.
        return random_state.uniform(action_spec.minimum, action_spec.maximum)

    return policy


def step_environment(env, policy, num_episodes=1, max_steps_per_episode=10):
    for _ in range(num_episodes):
        step_count = 0
        time_step = env.reset()
        print(time_step.observation)
        while not time_step.last():
            action = policy(time_step)
            time_step = env.step(action)
            print(time_step.observation)
            step_count += 1
            if step_count >= max_steps_per_episode:
                break


class TaskTest(parameterized.TestCase):
    def test_initial_state_is_randomized(self):
        # env = suite.load(domain, task, task_kwargs={"random": 42})
        env = pendulum.swingup(random=42)
        obs1 = env.reset().observation
        obs2 = env.reset().observation
        self.assertFalse(
            np.allclose(obs1, obs2),
            "Two consecutive initial states have identical observations.\n"
            "First: {}\nSecond: {}".format(obs1, obs2),
        )

    def test_deterministic_state_with_same_seed(self):
        env1 = pendulum.swingup(random=42)
        env2 = pendulum.swingup(random=42)
        obs1 = env1.reset().observation
        obs2 = env2.reset().observation
        self.assertTrue(
            np.allclose(obs1, obs2),
            "First reset with identical seeds should have identical observations.\n"
            "First: {}\nSecond: {}".format(obs1, obs2),
        )


class PendulumEnvTest(test_utils.EnvironmentTestMixin, absltest.TestCase):
    def make_object_under_test(self):
        return pendulum.swingup(random=42)

    def test_first_is_first(self):
        env = self.make_object_under_test()
        timestep = env.reset()
        self.assertEqual(dm_env.StepType.FIRST, timestep.step_type)
        timestep = env.reset()
        self.assertEqual(dm_env.StepType.FIRST, timestep.step_type)


if __name__ == "__main__":
    absltest.main()
