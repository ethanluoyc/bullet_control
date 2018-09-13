from bullet_control.tasks.locomotors.walkerbase import WalkerBase, Physics
from bullet_control.core import Environment
import numpy as np

def run():
    ant = Ant()
    physics = Physics(ant.foot_list)
    physics.load_MJCF('ant.xml')
    return Environment(physics, ant)


class Ant(WalkerBase):
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

    def __init__(self):
        WalkerBase.__init__(self, "ant.xml", "torso", action_dim=8, obs_dim=28, power=2.5)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground


if __name__ == '__main__':
    env = run()
    env.reset()
    while True:
        env.step(np.random.randn(env.task.action_spec())*0.01)