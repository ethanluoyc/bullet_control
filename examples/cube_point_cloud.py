import os
import sys

import matplotlib
import numpy as np
import open3d as o3d
import pybullet
import pybullet as p
import pybullet_data

from bullet_control.camera import Camera
from bullet_control.camera import get_point_cloud


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = p.loadURDF("plane100.urdf", [0, 0, -1])
    cube = p.loadURDF("cube_no_rotation.urdf", [0, 0, 1])
    camera = Camera()

    camera.nearPlane = 4
    camera.farPlane = 10
    camera.camTarget = [0, 0, 0.0]
    rgbs = []
    depths = []
    pcls = []
    for angle in np.linspace(0, 2 * np.pi, 10):
        camera.camTargetPos = [0.0, 0.0, 0.0]
        camera.camPos = [3 * np.cos(angle), 3 * np.sin(angle), 5.0]
        rgb, dep, depth_buffer = camera.snap(pybullet)
        rgbs.append(rgb)
        depths.append(dep)
        pcls.append(get_point_cloud(rgb, depth_buffer, camera._model_view_matrix, camera._projection_matrix))

    o3d.visualization.draw_geometries(pcls)

if __name__ == "__main__":
    main()
