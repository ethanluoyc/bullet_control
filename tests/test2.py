import pybullet as p
import pybullet_data
import pybullet_envs
import tempfile
import os
from bullet_control.core import Environment
from bullet_control import pendulum

def change_pendulum_length_and_load(l=0.6):
    import xml.etree.ElementTree as ET
    original = os.path.join(pybullet_data.getDataPath(), 'mjcf', 'inverted_pendulum.xml')
    print(original)
    tree = ET.parse(original)

    for geom in tree.iter('geom'):
        if geom.attrib.get('name') and geom.attrib['name'] == 'cpole':
            geom.set('fromto', '0 0 0 0.001 0 {}'.format(l))

    physics = pendulum.Physics()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'output.xml')
        tree.write(path)
        physics.load_MJCF(path)

    task = pendulum.PendulumTask(swingup=True)
    env = Environment(physics, task)
    return env

env = change_pendulum_length_and_load()
obs = env.reset()
print(obs)
