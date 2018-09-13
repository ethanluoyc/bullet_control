import numpy as np
from bullet_control.core import Task
from bullet_control import physics


class Physics(physics.Physics):
    def _bind(self, bullet_client, bodies):
        self.feet = [self.parts[f] for f in self.foot_list]


class WalkerBase(Task):
    def __init__(self, fn, robot_name, action_dim, obs_dim, power):
        # MJCFBasedRobot.__init__(self, fn, robot_name, action_dim, obs_dim)
        self.fn = fn

        self.robot_name = robot_name
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        self.power = power
        self.camera_x = 0
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        self.body_xyz = [0, 0, 0]

    def on_reset(self, physics):
        for j in physics.ordered_joints:
            j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0)

        self.feet = [physics.parts[f] for f in physics.foot_list]
        self.feet_contact = np.array([0.0 for f in physics.foot_list], dtype=np.float32)
        self.scene.actor_introduce(self)
        self.initial_z = None

    def action_spec(self):
        return self.action_dim

    def observation_spec(self):
        return self.obs_dim

    def step(self, action, physics):
        assert (np.isfinite(action).all())
        for n, j in enumerate(action.ordered_joints):
            j.set_motor_torque(self.power * j.power_coef * float(np.clip(action[n], -1, +1)))

    def get_observation(self, physics):
        j = np.array([j.current_relative_position() for j in physics.ordered_joints],
                     dtype=np.float32).flatten()
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        self.joint_speeds = j[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

        body_pose = physics.robot_body.pose()
        parts_xyz = np.array([p.pose().xyz() for p in physics.parts.values()]).flatten()
        self.body_xyz = (
            parts_xyz[0::3].mean(), parts_xyz[1::3].mean(),
            body_pose.xyz()[2])  # torso z is more informative than mean z
        self.body_rpy = body_pose.rpy()
        z = self.body_xyz[2]
        if self.initial_z == None:
            self.initial_z = z
        r, p, yaw = self.body_rpy
        self.walk_target_theta = np.arctan2(self.walk_target_y - self.body_xyz[1],
                                            self.walk_target_x - self.body_xyz[0])
        self.walk_target_dist = np.linalg.norm(
            [self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]])
        angle_to_target = self.walk_target_theta - yaw

        rot_speed = np.array(
            [[np.cos(-yaw), -np.sin(-yaw), 0],
             [np.sin(-yaw), np.cos(-yaw), 0],
             [0, 0, 1]]
        )
        vx, vy, vz = np.dot(rot_speed,
                            physics.robot_body.speed())  # rotate speed back to body point of view

        more = np.array([z - self.initial_z,
                         np.sin(angle_to_target), np.cos(angle_to_target),
                         0.3 * vx, 0.3 * vy, 0.3 * vz,
                         # 0.3 is just scaling typical speed into -1..+1, no physical sense here
                         r, p], dtype=np.float32)
        return np.clip(np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

    def calc_potential(self):
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0
        debugmode = 0
        if (debugmode):
            print("calc_potential: self.walk_target_dist")
            print(self.walk_target_dist)
            print("self.scene.dt")
            print(self.scene.dt)
            print("self.scene.frame_skip")
            print(self.scene.frame_skip)
            print("self.scene.timestep")
            print(self.scene.timestep)
        return - self.walk_target_dist / self.scene.dt
