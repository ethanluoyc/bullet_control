import logging
import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pybullet
import pybullet_data
from dm_control.utils import containers
from dm_env import specs
from pybullet_envs.robot_bases import BodyPart, Joint
from pybullet_utils.bullet_client import BulletClient

from bullet_control import camera, core

_LOG = logging.getLogger(__name__)

SUITE = containers.TaggedTasks()

PLANE_FILE = os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf")
MODEL_FILE = os.path.join(pybullet_data.getDataPath(), "mjcf", "inverted_pendulum.xml")


def _create_bullet_client():
    return BulletClient(connection_mode=pybullet.DIRECT)


@SUITE.add("benchmarking")
def swingup(random=None):
    """Create a Pendulum swingup task."""
    bc = _create_bullet_client()
    with open(MODEL_FILE, "rt") as infile:
        model_xml_string = infile.read()
    physics = Physics(bc, model_xml_string)
    # Create task.
    task = SwingUp(random=random)
    # Wrap into an environment.
    env = core.Environment(physics, task)
    return env


def change_length(length=0.6, random=None):
    """Create a Pendulum with different length for the pole."""

    # TODO(yl): use the MJCF editor from dm_env to perform modifications.
    original = MODEL_FILE
    tree = ET.parse(original)

    for geom in tree.iter("geom"):
        if geom.attrib.get("name") and geom.attrib["name"] == "cpole":
            geom.set("fromto", "0 0 0 0.001 0 {}".format(length))
    modified_xml_string = ET.tostring(tree.getroot(), encoding="unicode", method="xml")

    # Create Physics.
    bc = _create_bullet_client()
    physics = Physics(bc, modified_xml_string)
    # Create task.
    task = SwingUp(random=random)
    # Wrap into an environment.
    env = core.Environment(physics, task)
    return env


class Physics(core.Physics):
    """Wrapper around PyBullet physics engine.
    TODO(yl): move shared methods to the base class so that other tasks can reuse.
    """

    def __init__(self, bullet_client: BulletClient, model_xml_string: str) -> None:
        self.robot_name = "cart"
        self.parts = {}
        self.jdict = {}
        self.ordered_joints = []
        self.objects = None
        self.robot_body = None

        self._p = bullet_client
        self._model_xml_string = model_xml_string
        self._ground_plane_mjcf = None
        self._camera = camera.Camera(self)

    def step(self):
        self._p.stepSimulation()

    def close(self):
        del self._p

    def reset(self):
        self._p.resetSimulation()
        self._p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        self._load_plane()
        self._load_MJCF()

    def render(self):
        return self._camera.render()

    def _load_plane(self):
        bullet_client = self._p
        self._ground_plane_mjcf = bullet_client.loadSDF(PLANE_FILE)
        for i in self._ground_plane_mjcf:
            bullet_client.changeDynamics(i, -1, lateralFriction=0.8, restitution=0.5)
            bullet_client.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 0.8])
            bullet_client.configureDebugVisualizer(
                pybullet.COV_ENABLE_PLANAR_REFLECTION, 1
            )

    def _load_MJCF(self):
        self.robot_body = None

        flags = (
            pybullet.URDF_USE_SELF_COLLISION
            | pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
            | pybullet.URDF_GOOGLEY_UNDEFINED_COLORS
        )
        with tempfile.NamedTemporaryFile("wt") as tmpf:
            tmpf.write(self._model_xml_string)
            tmpf.flush()
            self.objects = self._p.loadMJCF(tmpf.name, flags)
        self._bind(self.objects)

    def _bind(self, bodies):
        """Bind objects and bullet physics client to physics instance."""
        self.parts = {}
        self.jdict = {}
        self.ordered_joints = []
        self.objects = None
        self.robot_body = None

        ordered_joints = self.ordered_joints
        # streamline the case where bodies is actually just one body
        if np.isscalar(bodies):
            bodies = [bodies]

        for i in range(len(bodies)):
            if self._p.getNumJoints(bodies[i]) == 0:
                part_name, robot_name = self._p.getBodyInfo(bodies[i])
                self.robot_name = robot_name.decode("utf8")
                part_name = part_name.decode("utf8")
                self.parts[part_name] = BodyPart(self._p, part_name, bodies, i, -1)
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

                self.parts[part_name] = BodyPart(self._p, part_name, bodies, i, j)

                if part_name == self.robot_name:
                    self.robot_body = self.parts[part_name]

                if (
                    i == 0 and j == 0 and self.robot_body is None
                ):  # if nothing else works, we take this as robot_body
                    self.parts[self.robot_name] = BodyPart(
                        self._p, self.robot_name, bodies, 0, -1
                    )
                    self.robot_body = self.parts[self.robot_name]

                if joint_name[:6] == "ignore":
                    Joint(self._p, joint_name, bodies, i, j).disable_motor()
                    continue

                if joint_name[:8] != "jointfix":
                    self.jdict[joint_name] = Joint(self._p, joint_name, bodies, i, j)
                    ordered_joints.append(self.jdict[joint_name])

                    self.jdict[joint_name].power_coef = 100.0

                # TODO: Maybe we need this
                # joints[joint_name].power_coef,
                # joints[joint_name].max_velocity
                # = joints[joint_name].limits()[2:4]
                # self.ordered_joints.append(joints[joint_name])
                # self.jdict[joint_name] = joints[joint_name]

        # Bind the relevant parts of the physics object
        self.pole = self.parts["pole"]
        self.slider = self.jdict["slider"]
        self.j1 = self.jdict["hinge"]

    # def reset_pose(self, position, orientation):
    #     self.parts[self.robot_name].reset_pose(position, orientation)


class SwingUp(core.Task):
    """Pendulum swing-up task."""

    def __init__(self, random=None):
        super().__init__(random=random)
        self.swingup = True
        self._observation_shape = (5,)
        self._action_shape = (1,)

    def initialize_episode(self, physics):
        u = self.random.uniform(low=-0.1, high=0.1)
        physics.j1.reset_current_position(u if not self.swingup else 3.1415 + u, 0)
        physics.j1.set_motor_torque(0)
        return self.get_observation(physics)

    def before_step(self, action, physics):
        action = np.asscalar(action)
        assert np.isfinite(action)
        physics.slider.set_motor_torque(100 * np.clip(action, -1, +1))

    def observation_spec(self, physics):
        return specs.Array(self._observation_shape, np.float32, name="observation")

    def action_spec(self, physics):
        return specs.BoundedArray(
            self._action_shape, np.float32, -1.0, +1.0, name="action"
        )

    def reward_spec(self, physics):
        return specs.Array((), np.float64, name="reward")

    def get_observation(self, physics):
        theta, theta_dot = physics.j1.current_position()
        x, vx = physics.slider.current_position()
        assert np.isfinite(x)

        if not np.isfinite(x):
            _LOG.warn("x is inf")
            x = 0

        if not np.isfinite(vx):
            _LOG.warn("vx is inf")
            vx = 0

        if not np.isfinite(theta):
            _LOG.warn("theta is inf")
            theta = 0

        if not np.isfinite(theta_dot):
            _LOG.warn("theta_dot is inf")
            theta_dot = 0

        return np.array(
            [x, vx, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32
        )

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
