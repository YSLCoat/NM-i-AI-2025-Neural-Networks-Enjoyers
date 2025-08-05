# train_sb3.py  (PPO + parallel training)

import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from src.game.racecar_env import RaceCarEnv

# --- Configuration ---
MODEL_SAVE_DIR = "models_sb3"
MODEL_NAME = "race_car_ppo_parallel_gpu"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
VECNORM_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_vecnormalize.pkl")
LOG_DIR = "logs_sb3"

TOTAL_TIMESTEPS = 2_000_000
NUM_CPU = 16
SEED = 42

def make_envs(n_envs: int, monitor_dir: str | None):
    """
    Build vectorized envs. SubprocVecEnv -> true parallelism across processes.
    If your env relies on pygame or any global state, ensure it is instance-safe.
    """
    return make_vec_env(
        env_id=RaceCarEnv,
        n_envs=n_envs,
        seed=SEED,
        vec_env_cls=SubprocVecEnv,   # ensure real parallelization
        monitor_dir=monitor_dir,     # enables episode stats logging
    )

def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Optional: sanity-check a single instance of the env API
    try:
        check_env(RaceCarEnv(), warn=True)
    except Exception as e:
        print(f"[check_env] Warning: {e}")

    # --- Training envs ---
    train_env = make_envs(NUM_CPU, monitor_dir=LOG_DIR)

    # Optional but recommended: normalize observations & rewards
    # Set gamma here too so reward normalization matches your training discount.
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)

    policy_kwargs = dict(
        net_arch=[256, 256],
        activation_fn=torch.nn.ReLU,   # keep consistent with your DQN MLP style
        ortho_init=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=1024,            # per env -> total rollout = n_steps * NUM_CPU
        batch_size=4096,         # must be <= n_steps * NUM_CPU (here 1024*16 = 16384)
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=LOG_DIR,
        seed=SEED,
        device="auto",
    )

    # --- Eval env (single process) sharing normalization stats with train env ---
    eval_env = make_envs(1, monitor_dir=None)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.995)
    # Share running obs statistics so eval sees inputs on the same scale:
    eval_env.obs_rms = train_env.obs_rms
    eval_env.training = False

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_SAVE_DIR,
        log_path=MODEL_SAVE_DIR,
        eval_freq=50_000,        # measured in env steps (aggregated across all envs)
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    print(f"--- Starting PPO Training on {NUM_CPU} envs ---")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True, callback=eval_callback)
    print("--- Training Complete ---")

    # Save model + VecNormalize stats (needed for proper scaling at inference)
    model.save(MODEL_SAVE_PATH)
    train_env.save(VECNORM_PATH)

    train_env.close()
    eval_env.close()
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"VecNormalize stats saved to {VECNORM_PATH}")

if __name__ == "__main__":
    # If your env uses pygame and you want headless training, uncomment the next line:
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    train()
