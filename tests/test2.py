from bullet_control.tasks import pendulum

env = pendulum.change_length()
obs = env.reset()

while True:
    pass