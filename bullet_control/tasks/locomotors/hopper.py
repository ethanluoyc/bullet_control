import numpy as np
import pybullet
import os
import pybullet_data
from bullet_control import physics, core
from bullet_control.core import Environment
from bullet_control.tasks.locomotors.walkerbase import WalkerBase, Physics


def run():
    ant = Hopper()
    physics = Physics(ant.foot_list)
    physics.load_MJCF("hopper.xml")
    return Environment(physics, ant)


class Hopper(WalkerBase):
    foot_list = ["foot"]

    def __init__(self):
        WalkerBase.__init__(
            self, "hopper.xml", "torso", action_dim=3, obs_dim=15, power=0.75
        )

    def step(self, a, physics):
        assert np.isfinite(a).all()
        for n, j in enumerate(physics.ordered_joints):
            j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))

    def get_observation(self, physics):
        j = np.array(
            [j.current_relative_position() for j in physics.ordered_joints],
            dtype=np.float32,
        ).flatten()
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        self.joint_speeds = j[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

        body_pose = physics.robot_body.pose()
        parts_xyz = np.array([p.pose().xyz() for p in physics.parts.values()]).flatten()
        self.body_xyz = (
            parts_xyz[0::3].mean(),
            parts_xyz[1::3].mean(),
            body_pose.xyz()[2],
        )  # torso z is more informative than mean z
        self.body_rpy = body_pose.rpy()
        z = self.body_xyz[2]
        if self.initial_z == None:
            self.initial_z = z
        r, p, yaw = self.body_rpy
        self.walk_target_theta = np.arctan2(
            self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]
        )
        self.walk_target_dist = np.linalg.norm(
            [
                self.walk_target_y - self.body_xyz[1],
                self.walk_target_x - self.body_xyz[0],
            ]
        )
        angle_to_target = self.walk_target_theta - yaw

        rot_speed = np.array(
            [
                [np.cos(-yaw), -np.sin(-yaw), 0],
                [np.sin(-yaw), np.cos(-yaw), 0],
                [0, 0, 1],
            ]
        )
        vx, vy, vz = np.dot(
            rot_speed, physics.robot_body.speed()
        )  # rotate speed back to body point of view

        more = np.array(
            [
                z - self.initial_z,
                np.sin(angle_to_target),
                np.cos(angle_to_target),
                0.3 * vx,
                0.3 * vy,
                0.3 * vz,
                # 0.3 is just scaling typical speed into -1..+1, no physical sense here
                r,
                p,
            ],
            dtype=np.float32,
        )
        return np.clip(np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

    def calc_potential(self):
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0
        debugmode = 0
        if debugmode:
            print("calc_potential: self.walk_target_dist")
            print(self.walk_target_dist)
            print("self.scene.dt")
            print(self.scene.dt)
            print("self.scene.frame_skip")
            print(self.scene.frame_skip)
            print("self.scene.timestep")
            print(self.scene.timestep)
        return -self.walk_target_dist / self.scene.dt

    def action_spec(self):
        return 3

    def observation_spec(self):
        return 15

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.8 and abs(pitch) < 1.0 else -1


if __name__ == "__main__":
    env = run()
    while True:
        env.step(np.random.randn(env.task.action_spec()) * 0.001)
