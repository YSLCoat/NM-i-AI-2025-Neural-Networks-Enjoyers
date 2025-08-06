# model_service.py

import os
import csv
import time
import numpy as np
import torch
from collections import deque
from typing import Dict, Any, List

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# ACTION_MAP is defined in racecar_env.py and maps action integers to names
from src.game.racecar_env import RaceCarEnv, ACTION_MAP
# SENSOR_ORDER is derived from the sensor_options in core.py
from core import sensor_options

# --- Configuration ---
MODEL_SAVE_DIR = "models_sb3"
MODEL_BASENAME = "race_car_ppo_cuda_parallel"
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_BASENAME}.zip")
VECNORM_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_BASENAME}_vecnormalize.pkl")
LOG_DIR = "eval_logs_api"

# --- Model & Environment Parameters (must match training) ---
N_STACK = 3
DETERMINISTIC_PREDICTION = True

# The exact order of sensors used to build the observation vector, taken from core.py
SENSOR_ORDER = [name for angle, name in sensor_options]

# --- Global State (managed by lifespan events) ---
model: PPO
vec_normalize: VecNormalize
frame_stack: deque
csv_writer: csv.DictWriter
csv_file: Any

def _classify_crash_sector(sensors: Dict[str, Any]) -> str:
    """
    Analyzes sensor data to classify the primary crash location.
    Returns one of: front, front_left, front_right, side_left, side_right, rear, unknown.
    """
    # Normalize sensor readings (None = max distance) for classification
    s = {name: (sensors.get(name) or 1000.0) / 1000.0 for name in SENSOR_ORDER}
    
    # A low value indicates a close obstacle. Threshold of 0.1 (100px) for definite impact.
    impact_threshold = 0.1

    # Check frontal sensors first
    if s['front'] < impact_threshold or s['front_left_front'] < impact_threshold or s['front_right_front'] < impact_threshold:
        # Distinguish between pure front and corner impacts
        if s['left_front'] < impact_threshold or s['left_side_front'] < impact_threshold:
            return "front_left"
        if s['right_front'] < impact_threshold or s['right_side_front'] < impact_threshold:
            return "front_right"
        return "front"

    # Check side sensors
    if s['left_side'] < impact_threshold:
        return "side_left"
    if s['right_side'] < impact_threshold:
        return "side_right"

    # Check rear sensors
    if s['back'] < impact_threshold or s['back_left_back'] < impact_threshold or s['back_right_back'] < impact_threshold:
        return "rear"
        
    return "unknown"


def load_model():
    """Initializes the model, environment wrappers, and the detailed crash logger."""
    global model, vec_normalize, frame_stack, csv_writer, csv_file

    print("--- Loading model and resources ---")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECNORM_PATH):
        raise FileNotFoundError(
            f"Model or VecNormalize not found. Searched for: {MODEL_PATH} and {VECNORM_PATH}. "
            "Please ensure trained model artifacts are in the 'models_sb3' directory."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = PPO.load(MODEL_PATH, device=device)

    dummy_env = make_vec_env(RaceCarEnv, n_envs=1)
    vec_normalize = VecNormalize.load(VECNORM_PATH, dummy_env)
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    print("VecNormalize stats loaded.")

    frame_stack = deque(maxlen=N_STACK)
    reset_state()

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"crash_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    csv_file = open(log_path, "w", newline="", encoding="utf-8")
    
    # Define the new, more detailed log headers
    fieldnames = [
        "timestamp", "run_duration_ticks", "final_distance",
        "action_name", "crash_sector",
    ]
    # Add a column for each sensor
    fieldnames.extend(SENSOR_ORDER)
    
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    print(f"Detailed crash logs will be saved to: {log_path}")


def close_model():
    """Cleans up resources on shutdown."""
    print("--- Closing resources ---")
    if csv_file:
        csv_file.close()


def reset_state():
    """Resets the frame stack, typically after a crash."""
    global frame_stack
    obs_shape = vec_normalize.observation_space.shape
    for _ in range(N_STACK):
        frame_stack.append(np.zeros(obs_shape, dtype=np.float32))
    print("State has been reset (frame stack cleared).")


def _create_obs_from_request(request_data: Dict[str, Any]) -> np.ndarray:
    """Constructs a numpy observation array from the API request dictionary."""
    sensor_readings = []
    sensor_dict = request_data.get('sensors', {})
    for sensor_name in SENSOR_ORDER:
        reading = sensor_dict.get(sensor_name)
        value = reading if reading is not None else 1000.0
        sensor_readings.append(value / 1000.0)

    velocity = request_data.get('velocity', {})
    vx = velocity.get('x', 0.0) / 20.0
    vy = velocity.get('y', 0.0) / 2.0

    obs = np.array(sensor_readings + [vx, vy], dtype=np.float32)
    return np.clip(obs, -1.0, 1.0)


def _log_crash_data(request_data: Dict[str, Any], action_name: str):
    """Writes detailed diagnostic information for a crash event to the CSV log."""
    sensors = request_data.get('sensors', {})
    
    # Prepare the log entry with all the requested details
    log_entry = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "run_duration_ticks": request_data.get('elapsed_ticks'),
        "final_distance": request_data.get('distance'),
        "action_name": action_name,
        "crash_sector": _classify_crash_sector(sensors),
    }

    # Add all sensor values to the log entry
    for sensor_name in SENSOR_ORDER:
        log_entry[sensor_name] = sensors.get(sensor_name)

    csv_writer.writerow(log_entry)
    csv_file.flush() # Ensure it's written to disk immediately


def predict_action(request_data: Dict[str, Any]) -> List[str]:
    """Processes a single request to predict the next action."""
    global frame_stack
    
    raw_obs = _create_obs_from_request(request_data)
    normalized_obs = vec_normalize.normalize_obs(np.array([raw_obs]))[0]
    frame_stack.append(normalized_obs)
    stacked_observation = np.concatenate(list(frame_stack), axis=0)
    
    action, _ = model.predict(
        stacked_observation.reshape(1, -1), 
        deterministic=DETERMINISTIC_PREDICTION
    )
    action_int = int(action[0])
    action_name = ACTION_MAP.get(action_int, 'NOTHING')
    
    # The logic remains: log only when the server confirms a crash.
    if request_data.get('did_crash', False):
        print(f"Crash detected at tick {request_data.get('elapsed_ticks')}. Logging detailed diagnostics...")
        _log_crash_data(request_data, action_name)
        reset_state()

    return [action_name]