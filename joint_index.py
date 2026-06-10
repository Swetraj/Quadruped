# Run this once to find real joint indices
import pybullet as p
import pybullet_data

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot = p.loadURDF("robot.urdf")

for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    print(f"[{i}] {info[1].decode()} type={info[2]}")

p.disconnect()