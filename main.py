import time
import pybullet as p
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.resetSimulation()
p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0 / 240.0)

plane = p.loadURDF("plane.urdf")
robot = p.loadURDF("robot.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=False)

# Dampen contacts — prevents the "launching off screen" behaviour
p.setPhysicsEngineParameter(
    enableConeFriction=1,
    contactBreakingThreshold=0.001,
    contactSlop=0.001,
)

# Kill bounciness on every link (index -1 = base)
num_joints = p.getNumJoints(robot)
for link_idx in range(-1, num_joints):
    p.changeDynamics(
        robot, link_idx,
        restitution=0.0,          # no bounce
        lateralFriction=0.8,      # grip the ground
        spinningFriction=0.05,
        rollingFriction=0.01,
        linearDamping=0.04,
        angularDamping=0.04,
        contactStiffness=30000,   # softer spring → less impulse spike
        contactDamping=1000,
    )

# Also set the plane to be non-bouncy
p.changeDynamics(plane, -1, restitution=0.0, lateralFriction=0.8)

while True:
    p.stepSimulation()
    time.sleep(1.0 / 240.0)