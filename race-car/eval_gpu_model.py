# eval_sb3.py — Evaluate PPO model with rendering, auto-wraps VecNormalize + VecFrameStack,
# and logs failure points (crash diagnostics) to CSV.

import os
import csv
import math
import time
import numpy as np
import torch
from collections import deque
from typing import Any, Dict, Optional, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, VecEnvWrapper

from src.game.racecar_env import RaceCarEnv

# ----------------------------- Config -----------------------------
MODEL_SAVE_DIR = "models_sb3"
MODEL_BASENAME = "race_car_ppo_cuda_parallel"  # must match training MODEL_NAME
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_BASENAME)
VECNORM_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_BASENAME}_vecnormalize.pkl")

NUM_EPISODES = 10
DETERMINISTIC = True
SEED = 10  # set to None for random

# Logging
CRASH_WINDOW_STEPS = 30  # how many steps before crash to log
EVAL_LOG_DIR = "eval_logs"

# ----------------------------- Helpers -----------------------------
def resolve_model_path(path: str) -> str:
    if os.path.isfile(path):
        return path
    zipped = path + ".zip"
    if os.path.isfile(zipped):
        return zipped
    raise FileNotFoundError(f"Model not found at '{path}' or '{zipped}'")

def infer_stack_n(model_obs_shape, base_obs_shape) -> int:
    model_dim = int(np.prod(model_obs_shape))
    base_dim = int(np.prod(base_obs_shape))
    if model_dim == base_dim:
        return 1
    if base_dim > 0 and model_dim % base_dim == 0:
        return model_dim // base_dim
    raise ValueError(
        f"Model expects obs dim {model_dim}, but base env provides {base_dim}. "
        f"Check that your eval wrappers (VecNormalize/VecFrameStack) match training."
    )

def unwrap_to_base_env(venv) -> Any:
    """Return the underlying gym.Env (first env) inside SB3 Vec wrappers."""
    current = venv
    # VecFrameStack/VecNormalize/etc. are VecEnvWrapper subclasses
    while isinstance(current, VecEnvWrapper):
        current = current.venv
    # current should be DummyVecEnv/SubprocVecEnv at this point
    try:
        return current.envs[0]
    except Exception:
        return None

def get_action_name(a: int, base_env) -> str:
    """Try to map action int -> name using env or module-level constants."""
    # Try attribute on the base env first
    if base_env is not None:
        if hasattr(base_env, "ACTIONS") and isinstance(base_env.ACTIONS, (list, tuple)):
            if 0 <= a < len(base_env.ACTIONS):
                return str(base_env.ACTIONS[a])
        if hasattr(base_env, "ACTION_MAP") and isinstance(base_env.ACTION_MAP, dict):
            return str(base_env.ACTION_MAP.get(a, a))
    # Fallback: try importing from module (if exported)
    try:
        from src.game.racecar_env import ACTIONS, ACTION_MAP
        if 0 <= a < len(ACTIONS):
            return str(ACTIONS[a])
        return str(ACTION_MAP.get(a, a))
    except Exception:
        pass
    # Last resort: just the int
    return str(a)

def classify_failure(info: Dict[str, Any]) -> str:
    """
    Heuristic failure type based on near-crash sensors.
    Returns one of: front, front_left, front_right, rear, side_left, side_right, unknown
    """
    def g(k, default=np.nan):
        v = info.get(k, default)
        try:
            return float(v)
        except Exception:
            return default

    fwd = g("fwd_min", np.nan)
    left = g("left_min", np.nan)
    right = g("right_min", np.nan)
    back = g("back_min", np.nan)

    # Lower value = closer obstacle
    thresh = 0.35
    # Prioritize front collisions
    if not np.isnan(fwd) and fwd < thresh:
        # If one front side is notably closer, call it that side
        if not np.isnan(left) and left < fwd - 0.05:
            return "front_left"
        if not np.isnan(right) and right < fwd - 0.05:
            return "front_right"
        return "front"
    # Rear collisions
    if not np.isnan(back) and back < thresh:
        return "rear"
    # Side brushes
    if not np.isnan(left) and left < thresh:
        return "side_left"
    if not np.isnan(right) and right < thresh:
        return "side_right"
    return "unknown"

def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default

# ----------------------------- Main -----------------------------
def evaluate():
    # Resolve model and load
    try:
        model_path = resolve_model_path(MODEL_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run training to create the model first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Loading model on device: {device}")
    model = PPO.load(model_path, device=device)

    # Build env with rendering
    env_kwargs = dict(render_mode="human")
    venv = make_vec_env(RaceCarEnv, n_envs=1, env_kwargs=env_kwargs, seed=SEED)

    # Base obs shape (before wrappers)
    base_obs_shape = venv.observation_space.shape

    # VecNormalize if stats exist
    if os.path.exists(VECNORM_PATH):
        venv = VecNormalize.load(VECNORM_PATH, venv)
        venv.training = False
        venv.norm_reward = False
        print(f"[Info] Loaded VecNormalize stats from: {VECNORM_PATH}")
    else:
        print(f"[WARN] VecNormalize stats not found at {VECNORM_PATH}. "
              f"Proceeding without observation normalization (may perform worse).")

    # Mirror FrameStack used in training
    try:
        n_stack = infer_stack_n(model.observation_space.shape, base_obs_shape)
    except Exception as e:
        print(f"[WARN] Could not infer n_stack automatically: {e} → defaulting to n_stack=3")
        n_stack = 3
    if n_stack > 1:
        venv = VecFrameStack(venv, n_stack=n_stack)
        print(f"[Info] Applied VecFrameStack(n_stack={n_stack}) to eval env.")
    else:
        print("[Info] No frame stacking required (n_stack=1).")

    # Prepare logging dirs/files
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(EVAL_LOG_DIR, f"{MODEL_BASENAME}_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "episodes_summary.csv")
    crashwin_path = os.path.join(out_dir, "crash_windows.csv")

    # Write headers
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "episode", "steps", "total_reward", "final_distance",
            "terminated", "truncated", "crashed",
            "failure_type", "crash_step",
            "action_before_crash_int", "action_before_crash_name",
            # crash metrics (from info)
            "fwd_min_at_crash", "left_min_at_crash", "right_min_at_crash",
            "back_min_at_crash",
            "ttc_fwd_at_crash", "ttc_left_at_crash", "ttc_right_at_crash",
            "headway_sec_at_crash", "rear_danger_at_crash",
            "vx_at_crash"
        ])

    with open(crashwin_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "episode", "step_offset", "step_index",
            "action_int", "action_name", "reward",
            "fwd_min", "left_min", "right_min", "back_min",
            "ttc_fwd", "ttc_left", "ttc_right", "headway_sec",
            "rear_danger", "vx"
        ])

    base_env = unwrap_to_base_env(venv)

    print(f"--- Evaluating model: {model_path} for {NUM_EPISODES} episode(s) ---")

    ep_returns = []
    ep_distances = []
    for ep in range(NUM_EPISODES):
        obs = venv.reset()
        done = False
        total_reward = 0.0
        last_info: Dict[str, Any] = {}
        steps = 0

        # Keep a rolling window for crash diagnostics
        window = deque(maxlen=CRASH_WINDOW_STEPS + 1)  # +1 to include crash step

        # Track if episode ended by crash vs time-limit
        crashed_flag = False
        truncated_flag = False
        terminated_flag = False

        action_int_prev = None
        action_name_prev = None

        while not done:
            action, _ = model.predict(obs, deterministic=DETERMINISTIC)
            # Record action in case we crash at this step
            a_int = int(np.asarray(action).squeeze())
            a_name = get_action_name(a_int, base_env)

            obs, reward, dones, infos = venv.step(action)

            r = float(reward[0])
            info = infos[0]
            steps += 1
            total_reward += r
            done = bool(dones[0])
            last_info = info

            # Extract safety features from info (may be absent depending on env version)
            rec = {
                "step_index": steps,
                "action_int": a_int,
                "action_name": a_name,
                "reward": r,
                "fwd_min": safe_float(info.get("fwd_min", np.nan)),
                "left_min": safe_float(info.get("left_min", np.nan)),
                "right_min": safe_float(info.get("right_min", np.nan)),
                "back_min": safe_float(info.get("back_min", np.nan)),
                "ttc_fwd": safe_float(info.get("ttc_fwd", np.nan)),
                "ttc_left": safe_float(info.get("ttc_left", np.nan)),
                "ttc_right": safe_float(info.get("ttc_right", np.nan)),
                "headway_sec": safe_float(info.get("headway_sec", np.nan)),
                "rear_danger": safe_float(info.get("rear_danger", np.nan)),
                "vx": safe_float(info.get("vx", np.nan)),
            }
            window.append(rec)

            action_int_prev = a_int
            action_name_prev = a_name

            # Determine termination types (best-effort)
            if done:
                crashed_flag = bool(info.get("crashed", False))
                # SB3/Gym often populate TimeLimit.truncated; we also check 'truncated'
                truncated_flag = bool(info.get("TimeLimit.truncated", False) or info.get("truncated", False))
                terminated_flag = crashed_flag or not truncated_flag

        # Final distance if provided
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

        # Summarize failure if crashed
        failure_type = "n/a"
        crash_step = ""
        fwd_c = left_c = right_c = back_c = np.nan
        ttc_f_c = ttc_l_c = ttc_r_c = head_c = rear_c = vx_c = np.nan
        if crashed_flag and len(window) > 0:
            failure_type = classify_failure(window[-1])
            crash_step = window[-1]["step_index"]
            fwd_c = window[-1]["fwd_min"]
            left_c = window[-1]["left_min"]
            right_c = window[-1]["right_min"]
            back_c = window[-1]["back_min"]
            ttc_f_c = window[-1]["ttc_fwd"]
            ttc_l_c = window[-1]["ttc_left"]
            ttc_r_c = window[-1]["ttc_right"]
            head_c = window[-1]["headway_sec"]
            rear_c = window[-1]["rear_danger"]
            vx_c = window[-1]["vx"]

            # Dump the crash window (last K steps, including crash)
            with open(crashwin_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                # step_offset: 0 at crash, negative before
                for i, rec in enumerate(list(window)[-CRASH_WINDOW_STEPS - 1:]):
                    step_offset = rec["step_index"] - crash_step
                    w.writerow([
                        ep + 1, step_offset, rec["step_index"],
                        rec["action_int"], rec["action_name"], rec["reward"],
                        rec["fwd_min"], rec["left_min"], rec["right_min"], rec["back_min"],
                        rec["ttc_fwd"], rec["ttc_left"], rec["ttc_right"], rec["headway_sec"],
                        rec["rear_danger"], rec["vx"]
                    ])

        # Episode summary row
        with open(summary_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                ep + 1, steps, f"{total_reward:.6f}",
                f"{final_distance:.6f}" if final_distance is not None else "",
                int(terminated_flag), int(truncated_flag), int(crashed_flag),
                failure_type, crash_step,
                action_int_prev if action_int_prev is not None else "",
                action_name_prev if action_name_prev is not None else "",
                fwd_c, left_c, right_c, back_c,
                ttc_f_c, ttc_l_c, ttc_r_c, head_c, rear_c, vx_c
            ])

        # Console report
        if final_distance is not None:
            print(f"Episode {ep + 1}: Steps={steps}, Final Distance={final_distance:.2f}, "
                  f"Total Reward={total_reward:.2f}, Crashed={crashed_flag}")
        else:
            print(f"Episode {ep + 1}: Steps={steps}, Total Reward={total_reward:.2f}, Crashed={crashed_flag}")

    # Summary to console
    mean_ret = sum(ep_returns) / max(len(ep_returns), 1)
    valid_dists = [d for d in ep_distances if not (isinstance(d, float) and math.isnan(d))]
    print("\n----- Summary -----")
    print(f"Average Total Reward: {mean_ret:.2f} over {NUM_EPISODES} episode(s)")
    if valid_dists:
        mean_dist = sum(valid_dists) / len(valid_dists)
        print(f"Average Final Distance: {mean_dist:.2f}")
    else:
        print("Average Final Distance: n/a (env did not report 'distance' in info)")
    print(f"[Logs] Episode summaries → {summary_path}")
    print(f"[Logs] Crash windows     → {crashwin_path}")

    venv.close()

if __name__ == '__main__':
    evaluate()
