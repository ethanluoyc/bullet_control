from absl.testing import absltest

from bullet_control import core
from bullet_control.tasks import load


class LoadTest(absltest.TestCase):
    def test_load_without_kwargs(self):
        env = load.load("pendulum", "swingup")
        self.assertIsInstance(env, core.Environment)

    def test_load_with_kwargs(self):
        env = load.load("pendulum", "swingup", task_kwargs={"random": 99})
        self.assertIsInstance(env, core.Environment)


class LoaderConstantsTest(absltest.TestCase):
    def testSuiteConstants(self):
        self.assertNotEmpty(load.BENCHMARKING)
        # self.assertNotEmpty(suite.EASY)
        # self.assertNotEmpty(suite.HARD)
        # self.assertNotEmpty(suite.EXTRA)


if __name__ == "__main__":
    absltest.main()
