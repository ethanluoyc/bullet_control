from bullet_control.core import Environment
from .walkerbase import WalkerBase, Physics
from bullet_control import physics
import numpy as np


def run():
    ant = HalfCheetah()
    physics = Physics(ant.foot_list)
    physics.load_MJCF('half_cheetah.xml')
    return Environment(physics, ant)


class HalfCheetah(WalkerBase):
    foot_list = ["ffoot", "fshin", "fthigh", "bfoot", "bshin",
                 "bthigh"]  # track these contacts with ground

    def __init__(self):
        WalkerBase.__init__(self, "half_cheetah.xml", "torso", action_dim=6, obs_dim=26, power=0.90)

    def alive_bonus(self, z, pitch):
        # Use contact other than feet to terminate episode: due to a lot of strange walks using knees
        return +1 if np.abs(pitch) < 1.0 and not self.feet_contact[1] and not self.feet_contact[
            2] and not self.feet_contact[4] and not self.feet_contact[5] else -1

    def on_reset(self, physics):
        WalkerBase.on_reset(self, physics)
        physics.jdict["bthigh"].power_coef = 120.0
        physics.jdict["bshin"].power_coef = 90.0
        physics.jdict["bfoot"].power_coef = 60.0
        physics.jdict["fthigh"].power_coef = 140.0
        physics.jdict["fshin"].power_coef = 60.0
        physics.jdict["ffoot"].power_coef = 30.0