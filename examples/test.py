import time

from bullet_control import physics

physics = physics.Physics()
# physics.load_MJCF('inverted_pendulum.xml')
physics.load_URDF("duck_vhacd.urdf")
# physics.load_URDF('duck_vhacd.urdf')
# investigate loading twice
# physics.load_MJCF('inverted_pendulum.xml')

print("joints")
print(physics.jdict)
print("parts")
print(physics.parts)

while True:
    time.sleep(0.1)
