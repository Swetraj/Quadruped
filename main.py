import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

robot = p.loadURDF("robot.urdf", [0, 0, 0.0192], useFixedBase=False)

# All 8 controllable joints parsed from your URDF
# joint index : (name, lower, upper, default)
JOINTS = {
    1: ("hip_FL",   -0.67, 0.67,    0.0),
    2: ("knee_FL",  -0.70, 1.5708,  0.5),
    5: ("hip_FR",   -0.67, 0.67,    0.0),
    6: ("knee_FR",  -0.70, 1.5708,  0.5),
    9: ("hip_RR",   -0.67, 0.67,    0.0),
    10:("knee_RR",  -0.70, 1.5708,  0.5),
    13:("hip_RL",   -0.67, 0.67,    0.0),
    14:("knee_RL",  -0.70, 1.5708,  0.5),
}

# Create one slider per joint
sliders = {}
for idx, (name, low, high, default) in JOINTS.items():
    sliders[idx] = p.addUserDebugParameter(name, low, high, default)

# Set initial pose so robot doesn't collapse
for idx, (name, low, high, default) in JOINTS.items():
    p.setJointMotorControl2(
        robot, idx,
        p.POSITION_CONTROL,
        targetPosition=default,
        force=10
    )

print("Sliders ready. Drag them in the PyBullet GUI.")

while True:
    for idx, slider_id in sliders.items():
        pos = p.readUserDebugParameter(slider_id)
        p.setJointMotorControl2(
            robot, idx,
            p.POSITION_CONTROL,
            targetPosition=pos,
            force=10,
            maxVelocity=2.0
        )
    p.stepSimulation()
    time.sleep(1.0 / 240.0)