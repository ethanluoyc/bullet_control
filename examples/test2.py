import numpy as np

from bullet_control.tasks import pendulum

env = pendulum.change_length()
obs = env.reset()

while True:
    # time.sleep(1)
    print(env.step(np.random.randn(env.task.action_spec())))
    print(env.task.get_observation(env.physics))
