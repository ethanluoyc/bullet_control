"""Get point cloud data from Kuka grasping environment."""
from gym import spaces
import numpy as np
import open3d as o3d
import pybullet as pb
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv


def unproject(win, modelView, modelProj, viewport):
    """Batched version of gluUnproject.
    Args:
        win: ndarray of shape (N, 3) points in pixel coordinates
        modelView: model-view matrix of the camera
        modelProj: projection matrix of the camera
        viewport: viewport

    Returns:
        pts: points in world coordinates

    Notes:
        This function is a direct port of

        https://www.khronos.org/opengl/wiki/GluProject_and_gluUnProject_code

        The arguments are the same as those expected by gluUnProject, except
        that win is a batch of points.

    """
    # Compute the inverse transform
    m = np.linalg.inv(modelProj @ modelView)  # 4 x 4
    winx = win[:, 0]
    winy = win[:, 1]
    winz = win[:, 2]
    # [B, 4]
    input_ = np.zeros((win.shape[0], 4), dtype=win.dtype)
    input_[:, 0] = (winx - viewport[0]) / viewport[2] * 2.0 - 1.0
    input_[:, 1] = (winy - viewport[1]) / viewport[3] * 2.0 - 1.0
    input_[:, 2] = winz * 2.0 - 1.0
    input_[:, 3] = 1.0
    out = (m @ input_.T).T
    # Check if out[3] == 0 ?
    out[:, 3] = 1 / out[:, 3]
    out[:, 0] = out[:, 0] * out[:, 3]
    out[:, 1] = out[:, 1] * out[:, 3]
    out[:, 2] = out[:, 2] * out[:, 3]
    return out[:, :3]


def get_point_cloud(rgb_buffer, z_buffer, view_matrix, projection_matrix):
    """Convert RGB-D data into point cloud

    Args:
        rgb_buffer: ndarray of shape (H, W, 4) representing the RGBA data.
            Expects data to be have range 0-255.
        depth_buffer: ndarray of shape (H, W) representing the z buffer.
            Note that this is not the true depth of the scene, but rather
            the z-buffer.
        view_matrix: model-view matrix of the camera. Can be obtained
            from PyBullet. Expects it to be in column-major format which
            is the format returned by PyBullet.
        projection_matrix: projection matrix of the camera. Obtained from Pybullet;
            needs to be in column-major format.
    Returns:
        pcl: Open3D PointCloud.
    """
    h, w = rgb_buffer.shape[:2]
    px, py = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    px, py = px.reshape(-1), py.reshape(-1)
    pz = z_buffer.reshape(w * h)
    wins = np.stack([px, (h - py), pz], axis=-1)
    colors = rgb_buffer[py, px, :3]
    # Compute pixels in world space
    points = unproject(
        wins,
        np.asarray(view_matrix).reshape((4, 4)).T,
        np.asarray(projection_matrix).reshape((4, 4)).T,
        (0, 0, w, h),
    )
    # If you are not interested in visualizing the point cloud simply return
    # points which is the 3D coordiates of the pixels in world space.
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.open3d.cpu.pybind.utility.Vector3dVector(np.array(points))
    # open3d expects colors to be normalized between 0 and 1
    pcl.colors = o3d.open3d.cpu.pybind.utility.Vector3dVector(np.array(colors) / 255.0)
    return pcl


def make_env(
    renders=True,
    is_discrete: bool = False,
    use_height_hack: bool = True,
    block_random: float = 0,
    camera_random: float = 0,
    test: bool = False,
    num_objects: int = 5,
    width: int = 64,
    height: int = 64,
    max_steps: int = 8,
):
    """Make a Kuka grasp environment."""
    return KukaDiverseObjectEnv(
        renders=renders,
        isDiscrete=is_discrete,
        removeHeightHack=not use_height_hack,
        blockRandom=block_random,
        cameraRandom=camera_random,
        numObjects=num_objects,
        isTest=test,
        width=width,
        height=height,
        maxSteps=max_steps,
    )


def capture(env):
    """Capture image from scene and convert to point cloud."""
    viewMatrix = env._view_matrix
    projectionMatrix = env._proj_matrix

    near = 0.01
    far = 10
    width = env._width
    height = env._height

    img_arr = env._p.getCameraImage(
        width,
        height,
        viewMatrix,
        projectionMatrix,
        renderer=env._p.ER_BULLET_HARDWARE_OPENGL,
    )
    assert len(img_arr) == 5

    w = img_arr[0]  # width of the image, in pixels
    h = img_arr[1]  # height of the image, in pixels
    rgbBuffer = img_arr[2]  # color data RGB
    depthBuffer = img_arr[3]  # depth data
    segmentationBuffer = img_arr[4]  # depth data

    return get_point_cloud(rgbBuffer, depthBuffer, viewMatrix, projectionMatrix)


def main():
    env = make_env()
    env.reset()
    print("Camera configuration:")
    print(env._cam_dist, env._cam_yaw, env._cam_pitch, env._width, env._height)
    pcl = capture(env)

    for t in range(50):
        pb.stepSimulation()
    o3d.visualization.draw_geometries([pcl])


if __name__ == "__main__":
    main()
