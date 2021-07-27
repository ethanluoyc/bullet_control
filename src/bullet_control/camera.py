import numpy as np
import pybullet as p
import pybullet
import glm
import open3d as o3d


class CameraIntrinsics:
    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


def _PrimeSense():
    """
    Intel PrimeSense intrinsics parameters, adapted from open3d intrinsics.
    See

    https://github.com/intel-isl/Open3D/blob/1728fc12934561321623c8798ef3b83059137321/cpp/open3d/camera/PinholeCameraIntrinsic.cpp#L49

    """
    return CameraIntrinsics(640, 480, 525.0, 525.0, 319.5, 239.5)


class Camera:
    """Camera for capturing images in Pybullet simulation.

    Notes:
      See http://ksimek.github.io/2013/08/13/intrinsic/ for a good reference on camera matrix
      And http://www.songho.ca/opengl/gl_projectionmatrix.html

      For modifications specific to OpenGL quirks, see
        http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    """

    def __init__(self):
        intrinsics = _PrimeSense()
        self.width = intrinsics.width
        self.height = intrinsics.height
        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.cx = intrinsics.cx
        self.cy = intrinsics.cy

        self.near = 0.01
        self.far = 100
        self.camPos = [0, 5, 5]
        self.camTargetPos = [0, 0, 0]
        self.upAxis = [0, 0, 1]

    @property
    def aspect(self):
        return self.width / self.height

    @property
    def fov(self):
        return np.arctan((self.width / 2) / self.fx) * 2

    @property
    def intrinsic_matrix(self):
        # Note skew is assumed to be zero, so [0, 1] == 0
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0.0, 1.0]])

    @property
    def matrix(self):
        # Adapted from dm_control
        viewMatrix = np.asarray(self._model_view_matrix).reshape((4, 4)).T
        # projectionMatrix = np.asarray(self._projection_matrix).reshape((4,4)).T
        # return self.intrinsic_matrix @ viewMatrix
        # Translation matrix (4x4).
        # pos = np.array(self.camPos)
        fovy = np.arctan((self.height / 2) / self.fy) * 2
        # translation = np.eye(4)
        # translation[0:3, 3] = -pos
        # # Rotation matrix (4x4).
        # rotation = np.eye(4)
        # rotation[0:3, 0:3] = rot
        # Focal transformation matrix (3x4).
        focal_scaling = (1.0 / np.tan(fovy / 2)) * self.height / 2.0
        focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]
        # Image matrix (3x3).
        image = np.eye(3)
        image[0, 2] = (self.width - 1) / 2.0
        image[1, 2] = (self.height - 1) / 2.0
        return image @ focal @ viewMatrix  # rotation @ translation

    @property
    def _model_view_matrix(self):
        viewMatrix = pybullet.computeViewMatrix(
            self.camPos, self.camTargetPos, self.upAxis
        )
        return viewMatrix

    @property
    def _projection_matrix(self):
        width = self.width
        height = self.height
        near = self.near
        far = self.far
        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy
        # TODO(yl): verify this
        left = cx * near / -fx
        top = cy * near / fy
        right = -(width - cx) * near / -fx
        bottom = -(height - cy) * near / fy
        # Note pybullet returns the projection matrix in column major
        # To convert it into the convention used in Python, do
        # np.asarray(projectionMatrix).reshape((4,4)).T
        projectionMatrix = pybullet.computeProjectionMatrix(
            left, right, bottom, top, near, far
        )
        return projectionMatrix

    def snap(self, bullet_client):
        # eye, target, up
        viewMatrix = self._model_view_matrix
        projectionMatrix = self._projection_matrix

        near = self.near
        far = self.far
        width = self.width
        height = self.height

        img_arr = bullet_client.getCameraImage(
            width,
            height,
            viewMatrix,
            projectionMatrix,
            renderer=bullet_client.ER_BULLET_HARDWARE_OPENGL,
        )
        assert len(img_arr) == 5

        w = img_arr[0]  # width of the image, in pixels
        h = img_arr[1]  # height of the image, in pixels
        rgbBuffer = img_arr[2]  # color data RGB
        depthBuffer = img_arr[3]  # depth data
        segmentationBuffer = img_arr[4]  # segmentation data

        colors = []
        rgb = np.reshape(rgbBuffer, (h, w, 4))
        rgb = rgb
        # Convert z-buffer depth to real depth
        # depth = self.near / (1. - (1. - self.near / self.far) * depthBuffer)
        trueDepth = (2.0 * near * far) / (
            far + near - (2.0 * depthBuffer - 1.0) * (far - near)
        )
        return rgbBuffer, trueDepth, depthBuffer


def unproject(win, modelView, modelProj, viewport):
    """Batched version of glu::unproject."""
    # https://www.khronos.org/opengl/wiki/GluProject_and_gluUnProject_code
    # 4 x 4
    m = np.linalg.inv(modelProj @ modelView)
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
    # Check if out[3] == 0?
    out[:, 3] = 1 / out[:, 3]
    out[:, 0] = out[:, 0] * out[:, 3]
    out[:, 1] = out[:, 1] * out[:, 3]
    out[:, 2] = out[:, 2] * out[:, 3]
    return out[:, :3]


def get_point_cloud(rgb_buffer, depth_buffer, view_matrix, projection_matrix):
    h, w = rgb_buffer.shape[:2]
    px, py = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    px, py = px.reshape(-1), py.reshape(-1)
    pz = depth_buffer.reshape(w * h)
    wins = np.stack([px, (h - py), pz], axis=-1)
    colors = rgb_buffer[py, px, :3]
    # compute pixels in world space
    points = unproject(
        wins,
        np.asarray(view_matrix).reshape((4, 4)).T,
        np.asarray(projection_matrix).reshape((4, 4)).T,
        (0, 0, w, h),
    )
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.open3d.cpu.pybind.utility.Vector3dVector(np.array(points))
    # open3d expects colors to be normalized between 0 and 1
    pcl.colors = o3d.open3d.cpu.pybind.utility.Vector3dVector(np.array(colors) / 255.0)
    return pcl
