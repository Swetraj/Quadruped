import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


class QuadrupedEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    JOINTS = {
        1:  ("hip_FL",  -0.67, 0.67,   0.0),
        2:  ("knee_FL", -0.70, 1.5708, 0.5),
        5:  ("hip_FR",  -0.67, 0.67,   0.0),
        6:  ("knee_FR", -0.70, 1.5708, 0.5),
        9:  ("hip_RR",  -0.67, 0.67,   0.0),
        10: ("knee_RR", -0.70, 1.5708, 0.5),
        13: ("hip_RL",  -0.67, 0.67,   0.0),
        14: ("knee_RL", -0.70, 1.5708, 0.5),
    }
    JOINT_INDICES = list(JOINTS.keys())
    N_JOINTS = len(JOINT_INDICES)
    FOOT_LINKS = [3, 7, 11, 15]

    def __init__(self, render_mode=None, urdf_path="robot.urdf"):
        super().__init__()
        self.render_mode = render_mode
        self.urdf_path = urdf_path
        self._client = None   # physics client ID
        self._robot = None
        self._step_counter = 0
        self._max_steps = 2000
        self._prev_x = 0.0

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.N_JOINTS,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(53,), dtype=np.float32)

    # ------------------------------------------------------------------
    def _p(self):
        """Always use self._client so each env instance is isolated."""
        return self._client

    def _connect(self):
        if self._client is not None:
            try:
                p.disconnect(self._client)
            except Exception:
                pass
        if self.render_mode == "human":
            self._client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._client)
        else:
            self._client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self._client)

    # ------------------------------------------------------------------
    def _load_terrain(self):
        num_cols, num_rows = 128, 128
        scale = 0.05
        heights = np.random.uniform(0, 0.015, num_cols * num_rows).tolist()
        shape = p.createCollisionShape(
            p.GEOM_HEIGHTFIELD,
            meshScale=[scale, scale, 1.5],
            heightfieldTextureScaling=64,
            heightfieldData=heights,
            numHeightfieldRows=num_rows,
            numHeightfieldColumns=num_cols,
            physicsClientId=self._client,
        )
        terrain = p.createMultiBody(0, shape, physicsClientId=self._client)
        p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1], physicsClientId=self._client)
        p.changeDynamics(terrain, -1, lateralFriction=1.0, physicsClientId=self._client)

        for _ in range(8):
            ox = np.random.uniform(0.3, 3.0)
            oy = np.random.uniform(-1.0, 1.0)
            oz = np.random.uniform(0.01, 0.04)
            half = [np.random.uniform(0.02, 0.06)] * 3
            box_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self._client)
            box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                             rgbaColor=[0.6, 0.4, 0.2, 1], physicsClientId=self._client)
            p.createMultiBody(0, box_shape, box_visual, [ox, oy, oz], physicsClientId=self._client)

    # ------------------------------------------------------------------
    def _spawn_robot(self):
        spawn_z = 0.0192 + 0.05
        self._robot = p.loadURDF(
            self.urdf_path,
            [0, 0, spawn_z],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False,
            physicsClientId=self._client,
        )
        p.changeDynamics(self._robot, -1, linearDamping=0.1, angularDamping=0.1,
                         physicsClientId=self._client)
        for idx in self.JOINT_INDICES:
            _, _, _, default = self.JOINTS[idx]
            p.resetJointState(self._robot, idx, default, physicsClientId=self._client)
            p.setJointMotorControl2(
                self._robot, idx, p.POSITION_CONTROL,
                targetPosition=default, force=10,
                physicsClientId=self._client,
            )
        for _ in range(50):
            p.stepSimulation(physicsClientId=self._client)

    # ------------------------------------------------------------------
    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self._robot, physicsClientId=self._client)
        lin_vel, ang_vel = p.getBaseVelocity(self._robot, physicsClientId=self._client)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)

        joint_pos, joint_vel = [], []
        for idx in self.JOINT_INDICES:
            js = p.getJointState(self._robot, idx, physicsClientId=self._client)
            joint_pos.append(js[0])
            joint_vel.append(js[1])

        contacts = []
        for link in self.FOOT_LINKS:
            pts = p.getContactPoints(self._robot, linkIndexA=link, physicsClientId=self._client)
            contacts.append(1.0 if pts else 0.0)

        scan = []
        for dx in np.linspace(-0.15, 0.15, 5):
            for dy in np.linspace(-0.15, 0.15, 5):
                ray_start = [pos[0] + dx, pos[1] + dy, pos[2]]
                ray_end   = [pos[0] + dx, pos[1] + dy, pos[2] - 0.5]
                hit = p.rayTest(ray_start, ray_end, physicsClientId=self._client)[0]
                height = hit[3][2] if hit[0] >= 0 else 0.0
                scan.append(pos[2] - height)

        obs = np.array(
            list(lin_vel) + list(ang_vel) +
            [roll, pitch] +
            joint_pos + joint_vel +
            contacts + scan,
            dtype=np.float32,
        )
        return obs, pos

    # ------------------------------------------------------------------
    def _apply_action(self, action):
        for i, idx in enumerate(self.JOINT_INDICES):
            _, low, high, _ = self.JOINTS[idx]
            target = low + (action[i] + 1.0) / 2.0 * (high - low)
            p.setJointMotorControl2(
                self._robot, idx,
                p.POSITION_CONTROL,
                targetPosition=float(target),
                force=10,
                maxVelocity=3.0,
                physicsClientId=self._client,
            )

    # ------------------------------------------------------------------
    def _compute_reward(self, pos, prev_x):
        forward_reward = (pos[0] - prev_x) * 300.0  # was 100, push harder

        lateral_penalty = abs(pos[1]) * 1.0          # was 0.5, penalise drifting more

        _, orn = p.getBasePositionAndOrientation(self._robot, physicsClientId=self._client)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        tilt_penalty = (abs(roll) + abs(pitch)) * 0.5

        torque_penalty = 0.0
        for idx in self.JOINT_INDICES:
            js = p.getJointState(self._robot, idx, physicsClientId=self._client)
            torque_penalty += abs(js[3]) * 0.001

        # Remove flat survival bonus — robot was exploiting it by standing still
        return forward_reward - lateral_penalty - tilt_penalty - torque_penalty

    # ------------------------------------------------------------------
    def _is_terminated(self, pos):
        if pos[2] < 0.01:
            return True
        _, orn = p.getBasePositionAndOrientation(self._robot, physicsClientId=self._client)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        if abs(roll) > 1.2 or abs(pitch) > 1.2:
            return True
        return False

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._connect()
        self._load_terrain()
        self._spawn_robot()
        self._step_counter = 0
        obs, pos = self._get_obs()
        self._prev_x = pos[0]
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action):
        self._apply_action(action)
        p.stepSimulation(physicsClientId=self._client)
        self._step_counter += 1
        obs, pos = self._get_obs()
        reward = self._compute_reward(pos, self._prev_x)
        self._prev_x = pos[0]
        terminated = self._is_terminated(pos)
        truncated = self._step_counter >= self._max_steps
        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    def close(self):
        if self._client is not None:
            try:
                p.disconnect(self._client)
            except Exception:
                pass
            self._client = None