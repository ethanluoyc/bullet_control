from bullet_control.tasks import pendulum
import numpy as np

env = pendulum.change_length()
obs = env.reset()

while True:
    import time
    # time.sleep(1)
    env.step(np.zeros(env.task.action_spec()))