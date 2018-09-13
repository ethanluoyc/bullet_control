from .walkerbase import WalkerBase
import numpy as np
import pybullet_data
import os


class Humanoid(WalkerBase):
    self_collision = True
    foot_list = ["right_foot", "left_foot"]  # "left_hand", "right_hand"

    def __init__(self):
        WalkerBase.__init__(self, 'humanoid_symmetric.xml', 'torso', action_dim=17, obs_dim=44,
                            power=0.41)

    # 17 joints, 4 of them important for walking (hip, knee), others may as well be turned off, 17/4 = 4.25

    def on_reset(self, bullet_client):
        WalkerBase.on_reset(self, bullet_client)
        self.motor_names = ["abdomen_z", "abdomen_y", "abdomen_x"]
        self.motor_power = [100, 100, 100]
        self.motor_names += ["right_hip_x", "right_hip_z", "right_hip_y", "right_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["left_hip_x", "left_hip_z", "left_hip_y", "left_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["right_shoulder1", "right_shoulder2", "right_elbow"]
        self.motor_power += [75, 75, 75]
        self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
        self.motor_power += [75, 75, 75]
        self.motors = [self.jdict[n] for n in self.motor_names]
        if self.random_yaw:
            position = [0, 0, 0]
            orientation = [0, 0, 0]
            yaw = self.np_random.uniform(low=-3.14, high=3.14)
            if self.random_lean and self.np_random.randint(2) == 0:
                cpose.set_xyz(0, 0, 1.4)
                if self.np_random.randint(2) == 0:
                    pitch = np.pi / 2
                    position = [0, 0, 0.45]
                else:
                    pitch = np.pi * 3 / 2
                    position = [0, 0, 0.25]
                roll = 0
                orientation = [roll, pitch, yaw]
            else:
                position = [0, 0, 1.4]
                orientation = [0, 0, yaw]  # just face random direction, but stay straight otherwise
            self.robot_body.reset_position(position)
            self.robot_body.reset_orientation(orientation)
        self.initial_z = 0.8

    random_yaw = False
    random_lean = False

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        force_gain = 1
        for i, m, power in zip(range(17), self.motors, self.motor_power):
            m.set_motor_torque(float(force_gain * power * self.power * np.clip(a[i], -1, +1)))

    def alive_bonus(self, z, pitch):
        return +2 if z > 0.78 else -1  # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying


def get_cube(_p, x, y, z):
    body = _p.loadURDF(os.path.join(pybullet_data.getDataPath(), "cube_small.urdf"), [x, y, z])
    _p.changeDynamics(body, -1, mass=1.2)  # match Roboschool
    part_name, _ = _p.getBodyInfo(body)
    part_name = part_name.decode("utf8")
    bodies = [body]
    return BodyPart(_p, part_name, bodies, 0, -1)


def get_sphere(_p, x, y, z):
    body = _p.loadURDF(os.path.join(pybullet_data.getDataPath(), "sphere2red_nocol.urdf"),
                       [x, y, z])
    part_name, _ = _p.getBodyInfo(body)
    part_name = part_name.decode("utf8")
    bodies = [body]
    return BodyPart(_p, part_name, bodies, 0, -1)


class HumanoidFlagrun(Humanoid):
    def __init__(self):
        Humanoid.__init__(self)
        self.flag = None

    def on_reset(self, physics):
        Humanoid.on_reset(self, physics)
        self.flag_reposition()

    def flag_reposition(self):
        self.walk_target_x = self.np_random.uniform(low=-self.scene.stadium_halflen,
                                                    high=+self.scene.stadium_halflen)
        self.walk_target_y = self.np_random.uniform(low=-self.scene.stadium_halfwidth,
                                                    high=+self.scene.stadium_halfwidth)
        more_compact = 0.5  # set to 1.0 whole football field
        self.walk_target_x *= more_compact
        self.walk_target_y *= more_compact

        if (self.flag):
            # for b in self.flag.bodies:
            #	print("remove body uid",b)
            #	p.removeBody(b)
            self._p.resetBasePositionAndOrientation(self.flag.bodies[0],
                                                    [self.walk_target_x, self.walk_target_y, 0.7],
                                                    [0, 0, 0, 1])
        else:
            self.flag = get_sphere(self._p, self.walk_target_x, self.walk_target_y, 0.7)
        self.flag_timeout = 600 / self.scene.frame_skip  # match Roboschool

    def calc_state(self):
        self.flag_timeout -= 1
        state = Humanoid.calc_state(self)
        if self.walk_target_dist < 1 or self.flag_timeout <= 0:
            self.flag_reposition()
            state = Humanoid.calc_state(self)  # caclulate state again, against new flag pos
            self.potential = self.calc_potential()  # avoid reward jump
        return state


class HumanoidFlagrunHarder(HumanoidFlagrun):
    def __init__(self):
        HumanoidFlagrun.__init__(self)
        self.flag = None
        self.aggressive_cube = None
        self.frame = 0

    def on_reset(self, physics):
        HumanoidFlagrun.on_reset(self, physics)

        self.frame = 0
        if (self.aggressive_cube):
            self._p.resetBasePositionAndOrientation(self.aggressive_cube.bodies[0], [-1.5, 0, 0.05],
                                                    [0, 0, 0, 1])
        else:
            self.aggressive_cube = get_cube(self._p, -1.5, 0, 0.05)
        self.on_ground_frame_counter = 0
        self.crawl_start_potential = None
        self.crawl_ignored_potential = 0.0
        self.initial_z = 0.8

    def alive_bonus(self, z, pitch):
        if self.frame % 30 == 0 and self.frame > 100 and self.on_ground_frame_counter == 0:
            target_xyz = np.array(self.body_xyz)
            robot_speed = np.array(self.robot_body.speed())
            angle = self.np_random.uniform(low=-3.14, high=3.14)
            from_dist = 4.0
            attack_speed = self.np_random.uniform(low=20.0,
                                                  high=30.0)  # speed 20..30 (* mass in cube.urdf = impulse)
            time_to_travel = from_dist / attack_speed
            target_xyz += robot_speed * time_to_travel  # predict future position at the moment the cube hits the robot
            position = [target_xyz[0] + from_dist * np.cos(angle),
                        target_xyz[1] + from_dist * np.sin(angle),
                        target_xyz[2] + 1.0]
            attack_speed_vector = target_xyz - np.array(position)
            attack_speed_vector *= attack_speed / np.linalg.norm(attack_speed_vector)
            attack_speed_vector += self.np_random.uniform(low=-1.0, high=+1.0, size=(3,))
            self.aggressive_cube.reset_position(position)
            self.aggressive_cube.reset_velocity(linearVelocity=attack_speed_vector)
        if z < 0.8:
            self.on_ground_frame_counter += 1
        elif self.on_ground_frame_counter > 0:
            self.on_ground_frame_counter -= 1
        # End episode if the robot can't get up in 170 frames, to save computation and decorrelate observations.
        self.frame += 1
        return self.potential_leak() if self.on_ground_frame_counter < 170 else -1

    def potential_leak(self):
        z = self.body_xyz[2]  # 0.00 .. 0.8 .. 1.05 normal walk, 1.2 when jumping
        z = np.clip(z, 0, 0.8)
        return z / 0.8 + 1.0  # 1.00 .. 2.0

    def calc_potential(self):
        # We see alive bonus here as a leak from potential field. Value V(s) of a given state equals
        # potential, if it is topped up with gamma*potential every frame. Gamma is assumed 0.99.
        #
        # 2.0 alive bonus if z>0.8, potential is 200, leak gamma=0.99, (1-0.99)*200==2.0
        # 1.0 alive bonus on the ground z==0, potential is 100, leak (1-0.99)*100==1.0
        #
        # Why robot whould stand up: to receive 100 points in potential field difference.
        flag_running_progress = Humanoid.calc_potential(self)

        # This disables crawl.
        if self.body_xyz[2] < 0.8:
            if self.crawl_start_potential is None:
                self.crawl_start_potential = flag_running_progress - self.crawl_ignored_potential
            # print("CRAWL START %+0.1f %+0.1f" % (self.crawl_start_potential, flag_running_progress))
            self.crawl_ignored_potential = flag_running_progress - self.crawl_start_potential
            flag_running_progress = self.crawl_start_potential
        else:
            # print("CRAWL STOP %+0.1f %+0.1f" % (self.crawl_ignored_potential, flag_running_progress))
            flag_running_progress -= self.crawl_ignored_potential
            self.crawl_start_potential = None

        return flag_running_progress + self.potential_leak() * 100