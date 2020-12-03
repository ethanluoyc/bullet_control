from bullet_control.core import Environment
from bullet_control.physics import Physics
from .walkerbase import WalkerBase


def run():
    ant = Walker2D()
    physics = Physics(ant.foot_list)
    physics.load_MJCF("walker2d.xml")
    return Environment(physics, ant)


class Walker2D(WalkerBase):
    foot_list = ["foot", "foot_left"]

    def __init__(self):
        WalkerBase.__init__(
            self, "walker2d.xml", "torso", action_dim=6, obs_dim=22, power=0.40
        )

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.8 and abs(pitch) < 1.0 else -1

    def on_reset(self, physics):
        WalkerBase.on_reset(self, physics)
        for n in ["foot_joint", "foot_left_joint"]:
            physics.jdict[n].power_coef = 30.0
