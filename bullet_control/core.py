import functools
import inspect
import pybullet


class BulletClient(object):
    """A wrapper for pybullet to manage different clients."""

    def __init__(self, connection_mode=pybullet.DIRECT, options=""):
        """Create a simulation and connect to it."""
        self._client = pybullet.connect(pybullet.SHARED_MEMORY)
        if (self._client < 0):
            print("options=", options)
            self._client = pybullet.connect(connection_mode, options=options)

    def __del__(self):
        """Clean up connection if not already done."""
        try:
            pybullet.disconnect(physicsClientId=self._client)
        except pybullet.error:
            pass

    def __getattr__(self, name):
        """Inject the client id into Bullet functions."""
        # TODO investigate if there are methods which do not tack physicsClientId
        attribute = getattr(pybullet, name)
        if inspect.isbuiltin(attribute):
            attribute = functools.partial(
                attribute, physicsClientId=self._client)
        return attribute


class Environment(object):
    def __init__(self, physics, task):
        self.physics = physics
        self.task = task

    def step(self, action):
        return self.task.step(action, self.physics)

    def reset(self):
        return self.task.on_reset(self.physics)


class Task(object):
    """A task is a problem to be solved in a domain"""

    def on_reset(self, physics):
        pass

    def get_observation(self, physics):
        pass

    def step(self, action, physics):
        # TODO before step and after step?
        pass

    def get_reward(self, physics):
        pass

    def get_termination(self, physics):
        pass
