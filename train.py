from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from env import QuadrupedEnv

# Vectorise: run N envs in parallel for faster data collection
N_ENVS = 8

vec_env = make_vec_env(
    lambda: QuadrupedEnv(urdf_path="robot.urdf"),
    n_envs=N_ENVS,
)

eval_env = make_vec_env(
    lambda: QuadrupedEnv(urdf_path="robot.urdf"),
    n_envs=1,
)

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,           # encourages exploration
    learning_rate=3e-4,
    tensorboard_log="./logs/",
    policy_kwargs=dict(net_arch=[256, 256]),
)

callbacks = [
    CheckpointCallback(save_freq=50_000, save_path="./checkpoints/", name_prefix="quadruped"),
    EvalCallback(eval_env, eval_freq=25_000, best_model_save_path="./best_model/"),
]

model.learn(total_timesteps=5_000_000, callback=callbacks)
model.save("quadruped_final")
print("Training done.")