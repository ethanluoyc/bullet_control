import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pybullet
import pybullet_data
from bullet_control import core
from bullet_control import physics as p
from dm_env import specs
from pybullet_utils.bullet_client import BulletClient


class Physics(p.Physics):
    def _bind(self, bullet_client, bodies):
        super()._bind(bullet_client, bodies)
        self.pole = self.parts["pole"]
        self.slider = self.jdict["slider"]
        self.j1 = self.jdict["hinge"]


class PendulumTask(core.Task):
    def __init__(self, swingup: bool, random=None):
        super().__init__(random=random)
        self.swingup = swingup
        self._observation_shape = (5,)
        self._action_shape = ()

    def initialize_episode(self, physics):
        u = self.random.uniform(low=-0.1, high=0.1)
        physics.j1.reset_current_position(u if not self.swingup else 3.1415 + u, 0)
        physics.j1.set_motor_torque(0)
        return self.get_observation(physics)

    def before_step(self, action, physics):
        assert np.isfinite(action)
        physics.slider.set_motor_torque(100 * np.clip(action, -1, +1))

    def observation_spec(self, physics):
        return specs.Array(self._observation_shape, np.float64)

    def action_spec(self, physics):
        return specs.BoundedArray(self._action_shape, np.float64, -1.0, +1.0)

    def reward_spec(self, physics):
        return specs.Array((), np.float64)

    def get_observation(self, physics):
        theta, theta_dot = physics.j1.current_position()
        x, vx = physics.slider.current_position()
        assert np.isfinite(x)

        if not np.isfinite(x):
            print("x is inf")
            x = 0

        if not np.isfinite(vx):
            print("vx is inf")
            vx = 0

        if not np.isfinite(theta):
            print("theta is inf")
            theta = 0

        if not np.isfinite(theta_dot):
            print("theta_dot is inf")
            theta_dot = 0

        return np.array([x, vx, np.cos(theta), np.sin(theta), theta_dot])

    def get_reward(self, physics):
        if self.swingup:
            theta, _ = physics.j1.current_position()
            return np.cos(theta)
        return 1.0

    def get_termination(self, physics):
        if self.swingup:
            return None
        else:
            theta, _ = physics.j1.current_position()
            return np.abs(theta) > 0.2


def swingup(random=None):
    """Create a Pendulum swingup task."""
    physics = Physics()
    # Create task.
    task = PendulumTask(swingup=True, random=random)
    path = os.path.join(pybullet_data.getDataPath(), "mjcf", "inverted_pendulum.xml")
    bc = BulletClient(connection_mode=pybullet.DIRECT)
    physics.load_MJCF(path, "cart", bc)
    # Wrap into an environment.
    env = core.Environment(physics, task)
    return env


def change_length(length=0.6, swingup=True):
    """Create a Pendulum with different length for the pole."""

    original = os.path.join(
        pybullet_data.getDataPath(), "mjcf", "inverted_pendulum.xml"
    )
    tree = ET.parse(original)

    for geom in tree.iter("geom"):
        if geom.attrib.get("name") and geom.attrib["name"] == "cpole":
            geom.set("fromto", "0 0 0 0.001 0 {}".format(length))

    # Create Physics.
    physics = Physics()
    bc = BulletClient(connection_mode=pybullet.DIRECT)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "output.xml")
        tree.write(path)
        physics.load_MJCF(path, "cart", bc)

    # Create task.
    task = PendulumTask(swingup=swingup)
    # Wrap into an environment.
    env = core.Environment(physics, task)
    return env
