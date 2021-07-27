import unittest
import numpy as np
from bullet_control.camera import Camera, unproject
import numpy.testing as np_test
import glm


class CameraTest(unittest.TestCase):
    def test_unproject(self):
        camera = Camera()
        camera.near = 0.1
        camera.far = 100
        camera.camTargetPos = [0, 0, 0.0]
        camera.upAxis = [0, 0, 1]
        camera.camTarget = [0, 0, 0.0]
        camera.camPos = [5.0, 0.0, 5]

        modelView = np.array(camera._model_view_matrix).reshape((4, 4)).T
        modelProj = np.array(camera._projection_matrix).reshape((4, 4)).T
        viewport = (0, 0, 64, 64)

        unprojected_np = unproject(
            np.array([[32, 24, 1.0]]), modelView, modelProj, viewport
        )[0]
        viewMatrix = camera._model_view_matrix
        projectionMatrix = camera._projection_matrix

        modelView = glm.mat4(
            viewMatrix[0],
            viewMatrix[1],
            viewMatrix[2],
            viewMatrix[3],
            viewMatrix[4],
            viewMatrix[5],
            viewMatrix[6],
            viewMatrix[7],
            viewMatrix[8],
            viewMatrix[9],
            viewMatrix[10],
            viewMatrix[11],
            viewMatrix[12],
            viewMatrix[13],
            viewMatrix[14],
            viewMatrix[15],
        )

        modelProj = glm.mat4(
            projectionMatrix[0],
            projectionMatrix[1],
            projectionMatrix[2],
            projectionMatrix[3],
            projectionMatrix[4],
            projectionMatrix[5],
            projectionMatrix[6],
            projectionMatrix[7],
            projectionMatrix[8],
            projectionMatrix[9],
            projectionMatrix[10],
            projectionMatrix[11],
            projectionMatrix[12],
            projectionMatrix[13],
            projectionMatrix[14],
            projectionMatrix[15],
        )

        unprojected_glm = glm.unProject([32, 24, 1.0], modelView, modelProj, viewport)
        np_test.assert_allclose(np.array(unprojected_glm), unprojected_np, rtol=1e-4)

    def test_compute(self):
        # Compute OpenGL compatible perspective matrix
        camera = Camera()
        camera.near = 0.1
        camera.far = 100
        camera.camTargetPos = [0, 0, 0.0]
        camera.upAxis = [0, 0, 1]
        camera.camTarget = [0, 0, 0.0]
        camera.camPos = [5.0, 0.0, 5]
        near = camera.near
        far = camera.far
        width = camera.width
        height = camera.height
        near = camera.near
        far = camera.far
        fx = camera.fx
        fy = camera.fy
        cx = camera.cx
        cy = camera.cy
        left = cx * near / -fx
        top = cy * near / fy
        right = -(width - cx) * near / -fx
        bottom = -(height - cy) * near / fy
        alpha = near
        s = 0
        beta = s
        x0 = 0
        y0 = 0
        A = near + far
        B = near * far
        K = np.array(
            [
                [alpha, s, -x0, 0],
                [0, beta, -y0, 0],
                [0, 0, A, B],
                [0, 0, -1, 0],
            ]
        )
        import glm

        NDC = np.array(
            [
                [2 / (right - left), 0, 0, -(right + left) / (right - left)],
                [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
                [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                [0, 0, 0, 1],
            ]
        )

        # np.array(glm.ortho(0, camera.width, camera.height, 0, near, far))
        # print(NDC @ K)
        # print(np.array(camera._projection_matrix).reshape((4,4), order="F"))

    def test_compute_view_matrix(self):
        # https://github.com/bulletphysics/bullet3/blob/39d981bce606edaebccd5f89b9a7b24b1097aaf4/examples/SharedMemory/PhysicsClientC_API.cpp
        # computeProjectionMatrix
        # https://github.com/bulletphysics/bullet3/blob/39d981bce606edaebccd5f89b9a7b24b1097aaf4/examples/SharedMemory/PhysicsClientC_API.cpp#L4620
        camera = Camera()
        camera.near = 0.1
        camera.far = 100
        camera.camTargetPos = [0, 0, 0.0]
        camera.upAxis = [0, 0, 1]
        camera.camTarget = [0, 0, 0.0]
        camera.camPos = [5.0, 0.0, 5]

        modelView = np.array(camera._model_view_matrix).reshape((4, 4), order="F")
        modelProj = np.array(camera._projection_matrix).reshape((4, 4), order="F")
        # print(modelProj)
        mat = glm.ortho(0, camera.width, camera.height, 0, camera.near, camera.far)
        narr = np.array(mat)
        camera_matrix = np.eye(4)
        # print()
        camera_matrix[:3, :4] = camera.matrix

        width = camera.width
        height = camera.height
        near = camera.near
        far = camera.far
        fx = camera.fx
        fy = camera.fy
        cx = camera.cx
        cy = camera.cy
        # TODO(yl): verify this
        left = cx * near / -fx
        top = cy * near / fy
        right = -(width - cx) * near / -fx
        bottom = -(height - cy) * near / fy
        # Note pybullet returns the projection matrix in column major
        # To convert it into the convention used in Python, do
        # np.asarray(projectionMatrix).reshape((4,4)).T
        # Test the result in https://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
        # glFrustum
        projectionMatrix = np.array(
            [
                [2 * near / (right - left), 0, 0, 0],
                [0, 2 * near / (top - bottom), 0, 0],
                [
                    (right + left) / (right - left),
                    (top + bottom) / (top - bottom),
                    -(far + near) / (far - near),
                    -1,
                ],
                [0, 0, -(float(2) * far * near) / (far - near), 0],
            ]
        ).T
        np_test.assert_allclose(
            projectionMatrix,
            np.asarray(modelProj).reshape((4, 4), order="F"),
            rtol=1e-4,
        )
        return projectionMatrix


if __name__ == "__main__":
    unittest.main()
