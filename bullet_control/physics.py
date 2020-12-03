import logging
import os

import numpy as np
import pybullet
import pybullet_data
from pybullet_envs.robot_bases import BodyPart, Joint

from bullet_control import core

_LOG = logging.getLogger(__name__)


class Camera:
    def __init__(self, physics, width=64, height=64) -> None:

        self.physics = physics
        self.distance = 2
        self.target_position = [0, 0, 0]
        self.yaw = 0
        self.pitch = 0
        self.width = width
        self.height = height
        self.up_axis_index = 2
        self.roll = 0

    def render(self):
        bullet_client = self.physics._p
        view_matrix = bullet_client.computeViewMatrixFromYawPitchRoll(
            self.target_position,
            self.distance,
            self.yaw,
            self.pitch,
            self.roll,
            self.up_axis_index,
        )

        proj_matrix = bullet_client.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.width) / self.height,
            nearVal=0.1,
            farVal=100.0,
        )

        img_arr = bullet_client.getCameraImage(
            self.width,
            self.height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
        )
        return img_arr[2]


class Physics(core.Physics):
    def __init__(self):
        super().__init__()
        self.parts = {}
        self.jdict = {}
        self.ordered_joints = []
        self.objects = None

        self.robot_name = None
        self.robot_body = None

        self._p = None
        self.camera = Camera(self)

    def load_plane(self, bullet_client):
        filename = os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf")
        self._ground_plane_mjcf = bullet_client.loadSDF(filename)
        for i in self._ground_plane_mjcf:
            bullet_client.changeDynamics(i, -1, lateralFriction=0.8, restitution=0.5)
            bullet_client.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 0.8])
            bullet_client.configureDebugVisualizer(
                pybullet.COV_ENABLE_PLANAR_REFLECTION, 1
            )

    def load_MJCF(self, xml_file, robot_name, bullet_client):
        self.robot_name = robot_name
        self.robot_body = None

        self.load_plane(bullet_client)
        flags = (
            pybullet.URDF_USE_SELF_COLLISION
            | pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
            | pybullet.URDF_GOOGLEY_UNDEFINED_COLORS
        )
        self.objects = bullet_client.loadMJCF(xml_file, flags)
        self._bind(bullet_client, self.objects)

    def _bind(self, bullet_client, bodies):
        """Bind objects and bullet physics client to physics instance."""
        assert self._p is None
        self._p = bullet_client

        parts = self.parts

        joints = self.jdict
        ordered_joints = self.ordered_joints
        # streamline the case where bodies is actually just one body
        if np.isscalar(bodies):
            bodies = [bodies]

        for i in range(len(bodies)):
            if self._p.getNumJoints(bodies[i]) == 0:
                part_name, robot_name = self._p.getBodyInfo(bodies[i])
                self.robot_name = robot_name.decode("utf8")
                part_name = part_name.decode("utf8")
                parts[part_name] = BodyPart(self._p, part_name, bodies, i, -1)
            for j in range(self._p.getNumJoints(bodies[i])):
                self._p.setJointMotorControl2(
                    bodies[i],
                    j,
                    pybullet.POSITION_CONTROL,
                    positionGain=0.1,
                    velocityGain=0.1,
                    force=0,
                )
                jointInfo = self._p.getJointInfo(bodies[i], j)
                joint_name = jointInfo[1]
                part_name = jointInfo[12]

                joint_name = joint_name.decode("utf8")
                part_name = part_name.decode("utf8")

                _LOG.debug("ROBOT PART '%s'", part_name)
                _LOG.debug("ROBOT JOINT '%s'", joint_name)

                parts[part_name] = BodyPart(self._p, part_name, bodies, i, j)

                if part_name == self.robot_name:
                    self.robot_body = parts[part_name]

                if (
                    i == 0 and j == 0 and self.robot_body is None
                ):  # if nothing else works, we take this as robot_body
                    parts[self.robot_name] = BodyPart(
                        self._p, self.robot_name, bodies, 0, -1
                    )
                    self.robot_body = parts[self.robot_name]

                if joint_name[:6] == "ignore":
                    Joint(self._p, joint_name, bodies, i, j).disable_motor()
                    continue

                if joint_name[:8] != "jointfix":
                    joints[joint_name] = Joint(self._p, joint_name, bodies, i, j)
                    ordered_joints.append(joints[joint_name])

                    joints[joint_name].power_coef = 100.0

                # TODO: Maybe we need this
                # joints[joint_name].power_coef,
                # joints[joint_name].max_velocity
                # = joints[joint_name].limits()[2:4]
                # self.ordered_joints.append(joints[joint_name])
                # self.jdict[joint_name] = joints[joint_name]

        return parts, joints, ordered_joints, self.robot_body

    # def reset_pose(self, position, orientation):
    #     self.parts[self.robot_name].reset_pose(position, orientation)

    def render(self):
        self.camera.render()
