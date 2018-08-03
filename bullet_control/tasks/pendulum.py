import numpy as np
from .core import Task
from .. import physics


class Physics(physics.Physics):
    def _bind(self, bullet_client, bodies):
        super()._bind(bullet_client, bodies)
        self.pole = self.parts["pole"]
        self.slider = self.jdict["slider"]
        self.j1 = self.jdict["hinge"]


class PendulumTask(Task):
    def __init__(self, swingup):
        self.swingup = swingup

    def on_reset(self, physics):
        u = np.random.uniform(low=-.1, high=.1)
        physics.j1.reset_current_position(u if not self.swingup else 3.1415 + u, 0)
        physics.j1.set_motor_torque(0)
        return self.get_observation(physics)

    def step(self, action, physics):
        action = np.asscalar(action)
        assert np.isfinite(action)
        physics.slider.set_motor_torque(100 * np.clip(action, -1, +1))

    def observation_spec(self):
        return 5

    def action_spec(self):
        return 1

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

        return np.array(
            [x, vx, np.cos(theta),
             np.sin(theta), theta_dot])

    def get_reward(self, physics):
        pass

    def get_termination(self, physics):
        pass
