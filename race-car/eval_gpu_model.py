# eval_sb3.py — Evaluate PPO model with rendering (loads VecNormalize)
import os
import math
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from src.game.racecar_env import RaceCarEnv

# ----------------------------- Config -----------------------------
MODEL_SAVE_DIR = "models_sb3"
MODEL_BASENAME = "race_car_ppo_cuda_parallel"  # must match your training script's MODEL_NAME
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_BASENAME)
VECNORM_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_BASENAME}_vecnormalize.pkl")
NUM_EPISODES = 10
DETERMINISTIC = True
SEED = 10  # set to None for random

def resolve_model_path(path: str) -> str:
    """Return an existing model path, trying with and without .zip suffix."""
    if os.path.isfile(path):
        return path
    zipped = path + ".zip"
    if os.path.isfile(zipped):
        return zipped
    raise FileNotFoundError(f"Model not found at '{path}' or '{zipped}'")

def evaluate():
    try:
        model_path = resolve_model_path(MODEL_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run your training script to create the model first.")
        return

    # Build a single-env DummyVecEnv with rendering enabled
    env_kwargs = dict(render_mode="human")
    if SEED is not None:
        # make_vec_env handles seeding internally for DummyVecEnv
        venv = make_vec_env(RaceCarEnv, n_envs=1, env_kwargs=env_kwargs, seed=SEED)
    else:
        venv = make_vec_env(RaceCarEnv, n_envs=1, env_kwargs=env_kwargs)

    # Restore VecNormalize stats if present (recommended)
    if os.path.exists(VECNORM_PATH):
        venv = VecNormalize.load(VECNORM_PATH, venv)
        venv.training = False
        venv.norm_reward = False
        print(f"[Info] Loaded VecNormalize stats from: {VECNORM_PATH}")
    else:
        print(f"[WARN] VecNormalize stats not found at {VECNORM_PATH}. "
              f"Proceeding without observation normalization (performance may be poor).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Loading model on device: {device}")
    model = PPO.load(model_path, device=device)

    print(f"--- Evaluating model: {model_path} for {NUM_EPISODES} episode(s) ---")

    ep_returns = []
    ep_distances = []  # will be NaN if env doesn't report

    for ep in range(NUM_EPISODES):
        obs = venv.reset()
        done = False
        total_reward = 0.0
        last_info = {}
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=DETERMINISTIC)
            obs, reward, dones, infos = venv.step(action)

            # VecEnv returns arrays/lists of length n_envs (=1 here)
            total_reward += float(reward[0])
            done = bool(dones[0])
            last_info = infos[0]
            steps += 1

        # Try to read final distance from info if the env provides it
        final_distance = None
        for key in ("distance", "final_distance", "score"):
            if isinstance(last_info, dict) and key in last_info:
                try:
                    final_distance = float(last_info[key])
                except Exception:
                    pass
                break

        ep_returns.append(total_reward)
        ep_distances.append(final_distance if final_distance is not None else float("nan"))

        if final_distance is not None:
            print(f"Episode {ep + 1}: Steps={steps}, Final Distance={final_distance:.2f}, Total Reward={total_reward:.2f}")
        else:
            print(f"Episode {ep + 1}: Steps={steps}, Total Reward={total_reward:.2f}")

    # Summary
    mean_ret = sum(ep_returns) / max(len(ep_returns), 1)
    # Check if we have any valid distances
    valid_dists = [d for d in ep_distances if not (isinstance(d, float) and math.isnan(d))]
    print("\n----- Summary -----")
    print(f"Average Total Reward: {mean_ret:.2f} over {NUM_EPISODES} episode(s)")
    if valid_dists:
        mean_dist = sum(valid_dists) / len(valid_dists)
        print(f"Average Final Distance: {mean_dist:.2f}")
    else:
        print("Average Final Distance: n/a (env did not report 'distance' in info)")

    venv.close()

if __name__ == '__main__':
    evaluate()
