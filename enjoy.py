from stable_baselines3 import PPO
from env import QuadrupedEnv

model = PPO.load("best_model/best_model")   # or "quadruped_final"
env   = QuadrupedEnv(render_mode="human", urdf_path="robot.urdf")

obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()