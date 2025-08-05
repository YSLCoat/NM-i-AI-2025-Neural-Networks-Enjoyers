# train_sb3.py

import os
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env # --- IMPORT VEC ENV HELPER ---
from src.game.racecar_env import RaceCarEnv

# --- Configuration ---
MODEL_SAVE_DIR = "models_sb3"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "race_car_dqn_parallel")
LOG_DIR = "logs_sb3"
TOTAL_TIMESTEPS = 2000000
NUM_CPU = 16 # --- ADDED: Number of environments to run in parallel ---

def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    policy_kwargs = dict(
        net_arch=[256, 256]
    )

    # --- MODIFIED: Create a vectorized environment to run multiple instances in parallel ---
    # This will create NUM_CPU instances of RaceCarEnv and run them on different CPU cores.
    env = make_vec_env(RaceCarEnv, n_envs=NUM_CPU)

    # 2. Instantiate the DQN model
    # Note: Many parameters are now scaled by NUM_CPU because the model "sees" NUM_CPU steps
    # for every one step of real time.
    model = DQN(
        "MlpPolicy",
        env, # Use the vectorized environment
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        buffer_size=100000,          # Increased buffer size for more diverse experiences
        learning_starts=100000,      # Increased, as we gather data NUM_CPU times faster
        batch_size=256,              # Can often be increased with parallel envs
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        target_update_interval=10000, # Also scaled to keep the update frequency similar in "real" steps
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=LOG_DIR
    )

    # 3. Train the model
    print(f"--- Starting Training on {NUM_CPU} CPUs ---")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    print("--- Training Complete ---")

    # 4. Save the trained model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    env.close()

if __name__ == '__main__':
    train()