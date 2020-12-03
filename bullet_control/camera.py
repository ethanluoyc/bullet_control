from bullet_control import core


class Camera:
    def __init__(self, physics: core.Physics, width=64, height=64) -> None:

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
