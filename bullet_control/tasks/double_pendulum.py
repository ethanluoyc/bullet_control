import numpy as np
from .. import core
from .. import physics


class Physics(physics.Physics):
    def _bind(self, bullet_client, bodies):
        super()._bind(bullet_client, bodies)
        self.pole2 = self.parts["pole2"]
        self.slider = self.jdict["slider"]
        self.j1 = self.jdict["hinge"]
        self.j2 = self.jdict["hinge2"]


class DoublePendulum(core.Task):
    def step(self, action, physics):
        assert (np.isfinite(action).all())
        physics.slider.set_motor_torque(
            200 * float(np.clip(action[0], -1, +1)))

    def on_reset(self, physics):
        u = np.random.uniform(low=-.1, high=.1, size=[2])
        physics.j1.reset_current_position(float(u[0]), 0)
        physics.j2.reset_current_position(float(u[1]), 0)
        physics.j1.set_motor_torque(0)
        physics.j2.set_motor_torque(0)

    def action_spec(self):
        return 1

    def observation_spec(self):
        return 9

    def get_observation(self, physics):
        theta, theta_dot = physics.j1.current_position()
        gamma, gamma_dot = physics.j2.current_position()
        x, vx = physics.slider.current_position()
        pos_x, _, _ = physics.pole2.pose().xyz()
        assert (np.isfinite(x))
        return np.array([
            x,
            vx,
            pos_x,
            np.cos(theta),
            np.sin(theta),
            theta_dot,
            np.cos(gamma),
            np.sin(gamma),
            gamma_dot,
        ])

    def get_reward(self, physics):
        pass

    def get_termination(self, physics):
        pass
