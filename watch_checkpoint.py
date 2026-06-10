# watch_checkpoint.py
import time
import os
from stable_baselines3 import PPO
from env import QuadrupedEnv

CHECKPOINT_DIR = "./checkpoints"

def get_latest_checkpoint():
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".zip")]
    if not files:
        return None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(CHECKPOINT_DIR, f)))
    return os.path.join(CHECKPOINT_DIR, files[-1])

while True:
    path = get_latest_checkpoint()
    if path is None:
        print("No checkpoint yet, waiting...")
        time.sleep(10)
        continue

    print(f"Loading: {path}")
    model = PPO.load(path)
    env = QuadrupedEnv(render_mode="human", urdf_path="robot.urdf")
    obs, _ = env.reset()

    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        time.sleep(1 / 240)
        if terminated or truncated:
            break

    env.close()
    print("Episode done. Reloading latest checkpoint...\n")
    time.sleep(3)