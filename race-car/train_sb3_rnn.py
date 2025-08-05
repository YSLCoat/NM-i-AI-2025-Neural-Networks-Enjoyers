# train_sb3_rnn.py

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
# --- 1. IMPORT THE MlpLstmPolicy CLASS ---
from stable_baselines3.common.policies import MlpLstmPolicy
from src.game.racecar_env import RaceCarEnv

# --- Configuration ---
MODEL_SAVE_DIR = "models_sb3"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "race_car_ppo_lstm")
LOG_DIR = "logs_sb3"
TOTAL_TIMESTEPS = 2000000
NUM_CPU = 8

def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Policy kwargs for LSTM remain the same
    policy_kwargs = dict(
        net_arch=[],
        lstm_hidden_size=256,
        n_lstm_layers=1
    )

    env = make_vec_env(RaceCarEnv, n_envs=NUM_CPU)

    # --- 2. PASS THE CLASS 'MlpLstmPolicy' DIRECTLY ---
    model = PPO(
        MlpLstmPolicy, # Use the imported class, not the string "MlpLstmPolicy"
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        tensorboard_log=LOG_DIR
    )

    print(f"--- Starting PPO-LSTM Training on {NUM_CPU} CPUs ---")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    
    print("--- Training Complete ---")
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    env.close()

if __name__ == '__main__':
    train()