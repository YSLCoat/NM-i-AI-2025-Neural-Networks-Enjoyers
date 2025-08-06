# model_service.py

import os
import csv
import time
import numpy as np
import torch
from collections import deque
from typing import Dict, Any, List
import uuid

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from src.game.racecar_env import RaceCarEnv, ACTION_MAP

# --- Configuration ---
MODEL_SAVE_DIR = "models_sb3"
MODEL_BASENAME = "race_car_ppo_cuda_parallel"
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_BASENAME}.zip")
VECNORM_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_BASENAME}_vecnormalize.pkl")
LOG_DIR = "eval_logs_api"
CRASH_WINDOW_SIZE = 30 # Number of steps to log before a crash

# --- Model & Environment Parameters ---
N_STACK = 3
DETERMINISTIC_PREDICTION = True

FPS = 60.0
MAX_SENSOR_PX = 1000.0
PREDICTION_HORIZON = 4   # see item #3
MIN_SAFE_STEERING_MARGIN = 120.0  # px
FWD_TTC_CUTOFF = 1.8  # seconds

sensor_options = [
    (90, "front"), (135, "right_front"), (180, "right_side"), (225, "right_back"),
    (270, "back"), (315, "left_back"), (0, "left_side"), (45, "left_front"),
    (22.5, "left_side_front"), (67.5, "front_left_front"), (112.5, "front_right_front"),
    (157.5, "right_side_front"), (202.5, "right_side_back"), (247.5, "back_right_back"),
    (292.5, "back_left_back"), (337.5, "left_side_back"),
]
SENSOR_ORDER = [name for angle, name in sensor_options]

# --- Global State ---
model: PPO
vec_normalize: VecNormalize
frame_stack: deque
history_window: deque
episode_id: str
# Log file writers
crash_log_writer: csv.DictWriter
crash_log_file: Any
prediction_log_writer: csv.DictWriter
prediction_log_file: Any

def load_model():
    """Initializes the model and both the prediction and crash loggers."""
    global model, vec_normalize, frame_stack, history_window, episode_id
    global crash_log_writer, crash_log_file, prediction_log_writer, prediction_log_file

    print("--- Loading model and resources ---")
    # ... (model loading logic remains the same)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(MODEL_PATH, device=device)
    dummy_env = make_vec_env(RaceCarEnv, n_envs=1)
    vec_normalize = VecNormalize.load(VECNORM_PATH, dummy_env)
    vec_normalize.training = False; vec_normalize.norm_reward = False

    os.makedirs(LOG_DIR, exist_ok=True)
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')

    # 1. Setup PREDICTION log (logs every step)
    pred_log_path = os.path.join(LOG_DIR, f"prediction_log_{run_timestamp}.csv")
    prediction_log_file = open(pred_log_path, "w", newline="", encoding="utf-8")
    pred_fieldnames = [
        "timestamp","episode_id","tick","distance",
        "action_name","vx","vy"
    ] + SENSOR_ORDER + ["fwd_min","left_min","right_min","back_min","ttc_fwd"]
    prediction_log_writer = csv.DictWriter(prediction_log_file, fieldnames=pred_fieldnames)
    prediction_log_writer.writeheader()
    print(f"Continuous prediction log will be saved to: {pred_log_path}")

    # 2. Setup CRASH log (logs N steps leading to a crash)
    crash_log_path = os.path.join(LOG_DIR, f"crash_log_{run_timestamp}.csv")
    crash_log_file = open(crash_log_path, "w", newline="", encoding="utf-8")
    crash_fieldnames = ["timestamp", "episode_id", "step_offset", "tick", "action_name"] + SENSOR_ORDER
    crash_log_writer = csv.DictWriter(crash_log_file, fieldnames=crash_fieldnames)
    crash_log_writer.writeheader()
    print(f"Detailed crash logs will be saved to: {crash_log_path}")

    # 3. Initialize state containers
    frame_stack = deque(maxlen=N_STACK)
    history_window = deque(maxlen=CRASH_WINDOW_SIZE)
    reset_state()

def close_model():
    """Cleans up resources on shutdown."""
    print("--- Closing resources ---")
    if crash_log_file: crash_log_file.close()
    if prediction_log_file: prediction_log_file.close()

def reset_state():
    """Resets the state for a new run."""
    global frame_stack, history_window, episode_id
    episode_id = str(uuid.uuid4())[:8] # New ID for each run
    obs_shape = vec_normalize.observation_space.shape
    # Clear frame stack for model
    frame_stack.clear()
    for _ in range(N_STACK):
        frame_stack.append(np.zeros(obs_shape, dtype=np.float32))
    # Clear history window for crash logging
    history_window.clear()
    print(f"State reset for new episode: {episode_id}")

def _create_obs_from_request(request_data: Dict[str, Any]) -> np.ndarray:
    """Constructs a numpy observation array from the API request."""
    # ... (function is unchanged)
    sensor_readings = []
    sensor_dict = request_data.get('sensors', {})
    for sensor_name in SENSOR_ORDER:
        value = sensor_dict.get(sensor_name) or 1000.0
        sensor_readings.append(value / 1000.0)
    velocity = request_data.get('velocity', {})
    vx = velocity.get('x', 0.0) / 20.0
    vy = velocity.get('y', 0.0) / 2.0
    obs = np.array(sensor_readings + [vx, vy], dtype=np.float32)
    return np.clip(obs, -1.0, 1.0)


def predict_action(request_data: Dict[str, Any]) -> List[str]:
    """Processes a request, logs data, and returns a predicted action."""
    # 1. Predict action
    raw_obs = _create_obs_from_request(request_data)
    normalized_obs = vec_normalize.normalize_obs(np.array([raw_obs]))[0]
    frame_stack.append(normalized_obs)
    stacked_observation = np.concatenate(list(frame_stack), axis=0)
    action, _ = model.predict(stacked_observation.reshape(1, -1), deterministic=DETERMINISTIC_PREDICTION)
    action_int = int(action[0]); action_name = ACTION_MAP.get(action_int, 'NOTHING')

    if action_name == "ACCELERATE" and ttc_fwd < FWD_TTC_CUTOFF:
        action_name = "DECELERATE"

    # Don’t steer into the tighter side unless it’s clearly safer
    if action_name == "STEER_LEFT" and left_min + MIN_SAFE_STEERING_MARGIN < right_min:
        action_name = "NOTHING"
    if action_name == "STEER_RIGHT" and right_min + MIN_SAFE_STEERING_MARGIN < left_min:
        action_name = "NOTHING"

    # Discourage braking if a car is too close behind AND forward is not dangerous
    rear_danger = back_min < 350.0
    fwd_clear   = fwd_min  > 450.0
    if action_name == "DECELERATE" and rear_danger and fwd_clear:
        action_name = "NOTHING"

    sensors = request_data.get("sensors", {}) or {}
    def cone_min(names):
        return float(min(sensors.get(n, MAX_SENSOR_PX) for n in names))

    FRONT_CONE = ["left_front","front_left_front","front","front_right_front","right_front"]
    LEFT_CONE  = ["left_side_front","left_front","front_left_front"]
    RIGHT_CONE = ["front_right_front","right_front","right_side_front"]
    BACK_CONE  = ["right_back","back","left_back"]

    fwd_min   = cone_min(FRONT_CONE)
    left_min  = cone_min(LEFT_CONE)
    right_min = cone_min(RIGHT_CONE)
    back_min  = cone_min(BACK_CONE)

    # Simple forward TTC (seconds)
    vx_px_per_tick = float((request_data.get("velocity", {}) or {}).get("x", 0.0))
    vx_px_per_sec  = max(vx_px_per_tick * FPS, 1e-6)
    ttc_fwd = float(fwd_min / vx_px_per_sec)  # seconds

    # 3) write **full** prediction row including sensors + diagnostics
    row = {
        "timestamp": time.strftime('%H:%M:%S'),
        "episode_id": episode_id,
        "tick": request_data.get('elapsed_ticks'),
        "distance": request_data.get('distance'),
        "action_name": action_name,
        "vx": vx_px_per_tick,
        "vy": (request_data.get("velocity", {}) or {}).get("y", 0.0),
        "fwd_min": fwd_min, "left_min": left_min, "right_min": right_min,
        "back_min": back_min, "ttc_fwd": ttc_fwd
    }
    # append all sensors (raw px)
    for name in SENSOR_ORDER:
        row[name] = sensors.get(name)
    prediction_log_writer.writerow(row)
    prediction_log_file.flush()

    # 3. Store current state in the history window for potential crash logging
    history_step = {"tick": request_data.get('elapsed_ticks'), "action_name": action_name, "sensors": request_data.get('sensors', {})}
    history_window.append(history_step)

    # 4. If a crash is detected, dump the history window to the crash log
    if request_data.get('did_crash', False):
        print(f"Crash detected at tick {request_data.get('elapsed_ticks')}. Dumping crash window to log...")
        crash_tick = request_data.get('elapsed_ticks')
        for i, step_data in enumerate(list(history_window)):
            log_entry = {
                "timestamp": time.strftime('%H:%M:%S'), "episode_id": episode_id,
                "step_offset": i - (len(history_window) - 1), # 0 is crash, -1 is step before, etc.
                "tick": step_data['tick'], "action_name": step_data['action_name'],
            }
            # Add all sensor values from that step
            for sensor_name in SENSOR_ORDER:
                log_entry[sensor_name] = step_data['sensors'].get(sensor_name)
            crash_log_writer.writerow(log_entry)
        crash_log_file.flush()
        reset_state()

    return [action_name]