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

from src.game.racecar_env import RaceCarEnv, ACTION_MAP

# --- Configuration ---
MODEL_SAVE_DIR = "models_sb3"
MODEL_BASENAME = "race_car_ppo_cuda_parallel"
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_BASENAME}.zip")
VECNORM_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_BASENAME}_vecnormalize.pkl")
LOG_DIR = "eval_logs_api"

# --- Model & Environment Parameters (must match training) ---
N_STACK = 3
DETERMINISTIC_PREDICTION = True

# The exact order of sensors used to build the observation vector.
# This is derived from `core.py`.
SENSOR_ORDER = [
    "front", "right_front", "right_side", "right_back", "back", "left_back",
    "left_side", "left_front", "left_side_front", "front_left_front",
    "front_right_front", "right_side_front", "right_side_back",
    "back_right_back", "back_left_back", "left_side_back"
]

# --- Global State (managed by lifespan events) ---
model: PPO
vec_normalize: VecNormalize
frame_stack: deque
csv_writer: csv.DictWriter
csv_file: Any

def load_model():
    """Initializes the model, environment wrappers, and logging."""
    global model, vec_normalize, frame_stack, csv_writer, csv_file

    print("--- Loading model and resources ---")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECNORM_PATH):
        raise FileNotFoundError(
            f"Model or VecNormalize not found. Searched for: {MODEL_PATH} and {VECNORM_PATH}. "
            "Please ensure trained model artifacts are in the 'models_sb3' directory."
        )

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the trained PPO model
    model = PPO.load(MODEL_PATH, device=device)

    # Load the VecNormalize statistics
    # We need to wrap a dummy env to load the stats
    dummy_env = make_vec_env(RaceCarEnv, n_envs=1)
    vec_normalize = VecNormalize.load(VECNORM_PATH, dummy_env)
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    print("VecNormalize stats loaded.")

    # Initialize the frame stack deque
    obs_shape = vec_normalize.observation_space.shape
    frame_stack = deque(maxlen=N_STACK)
    reset_state() # Pre-fill with empty frames

    # Setup crash logging
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"crash_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    csv_file = open(log_path, "w", newline="", encoding="utf-8")
    
    # Define log headers, similar to local evaluation script
    fieldnames = [
        "timestamp", "tick", "distance", "action_name",
        "fwd_min", "left_min", "right_min", "back_min",
        "ttc_fwd", "ttc_left", "ttc_right", "headway_sec",
        "rear_danger", "vx", "vy",
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_file.flush()                 # ensure header appears immediately
    import os as _os
    try:
        _os.fsync(csv_file.fileno()) # durable on disk (optional but useful)
    except Exception:
        pass
    print(f"Crash logs will be saved to: {os.path.abspath(log_path)}")


def close_model():
    """Cleans up resources on shutdown."""
    print("--- Closing resources ---")
    if csv_file:
        csv_file.close()

def reset_state():
    """Resets the frame stack, typically after a crash."""
    global frame_stack
    obs_shape = vec_normalize.observation_space.shape
    # Fill the deque with zero-arrays representing empty frames
    for _ in range(N_STACK):
        frame_stack.append(np.zeros(obs_shape, dtype=np.float32))
    print("State has been reset (frame stack cleared).")


def _create_obs_from_request(request_data: Dict[str, Any]) -> np.ndarray:
    """
    Constructs a numpy observation array from the API request dictionary.
    The format must exactly match the one from `RaceCarEnv._get_obs`.
    """
    # 1. Sensor readings (ordered and normalized)
    sensor_readings = []
    # Use the provided sensor dictionary
    sensor_dict = request_data.get('sensors', {})
    for sensor_name in SENSOR_ORDER:
        reading = sensor_dict.get(sensor_name)
        # Handle None (no obstacle detected) as max distance (1000px)
        # then normalize to [0, 1] range.
        value = reading if reading is not None else 1000.0
        sensor_readings.append(value / 1000.0)

    # 2. Velocity components (normalized)
    velocity = request_data.get('velocity', {})
    vx = velocity.get('x', 0.0) / 20.0
    vy = velocity.get('y', 0.0) / 2.0

    # 3. Combine and clip
    obs = np.array(sensor_readings + [vx, vy], dtype=np.float32)
    return np.clip(obs, -1.0, 1.0)


def _log_crash_data(request_data: Dict[str, Any], action_name: str):
    sensors = request_data.get('sensors', {}) or {}
    velocity = request_data.get('velocity', {}) or {}
    vx = float(velocity.get('x', 0.0))
    vy = float(velocity.get('y', 0.0))

    def _sv(sdict, name, default=1000.0):
        v = sdict.get(name, default)
        return default if v is None else v

    # in _log_crash_data:
    fwd_sensors   = [_sv(sensors, s) for s in ["front_left_front", "front", "front_right_front"]]
    left_sensors  = [_sv(sensors, s) for s in ["left_side_front", "left_front"]]
    right_sensors = [_sv(sensors, s) for s in ["right_side_front", "right_front"]]
    back_sensors  = [_sv(sensors, s) for s in ["back_left_back", "back", "back_right_back"]]
    # Normalize to [0,1] like the obs
    fwd_min   = min(fwd_sensors)   / 1000.0
    left_min  = min(left_sensors)  / 1000.0
    right_min = min(right_sensors) / 1000.0
    back_min  = min(back_sensors)  / 1000.0

    # TTC / headway (same definitions as eval where possible)
    FPS = 60.0
    MAX_SENSOR_PX = 1000.0
    vx_pos = max(vx, 1e-6)
    ttc_fwd  = (fwd_min  * MAX_SENSOR_PX) / vx_pos / FPS
    ttc_left = (left_min * MAX_SENSOR_PX) / vx_pos / FPS
    ttc_right= (right_min* MAX_SENSOR_PX) / vx_pos / FPS
    headway_sec = ttc_fwd

    # Rear danger: square the ramp to match env shaping
    rear_danger = max((0.6 - back_min) / 0.6, 0.0) ** 2

    log_entry = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "tick": request_data.get('elapsed_ticks'),
        "distance": request_data.get('distance'),
        "action_name": action_name,
        "fwd_min": fwd_min,
        "left_min": left_min,
        "right_min": right_min,
        "back_min": back_min,
        "ttc_fwd": ttc_fwd,
        "ttc_left": ttc_left,
        "ttc_right": ttc_right,
        "headway_sec": headway_sec,
        "rear_danger": rear_danger,
        "vx": vx,
        "vy": vy,
    }
    csv_writer.writerow(log_entry)
    csv_file.flush()


def predict_action(request_data: Dict[str, Any]) -> List[str]:
    """
    Processes a single request to predict the next action.
    Manages state and logging internally.
    """
    global frame_stack
    
    # 1. Create the raw observation vector from request data
    raw_obs = _create_obs_from_request(request_data)

    # 2. Normalize the observation using loaded VecNormalize stats
    # Note: normalize_obs expects a batch, so we wrap and unwrap
    normalized_obs = vec_normalize.normalize_obs(np.array([raw_obs]))[0]

    # 3. Update the frame stack with the latest normalized frame
    frame_stack.append(normalized_obs)

    # 4. Concatenate frames to create the final observation for the model
    # The result is a single flat vector containing all stacked frames.
    stacked_observation = np.concatenate(list(frame_stack), axis=0)
    
    # 5. Get prediction from the model
    # Model expects a batch, so we add a dimension and then select the first result
    action, _ = model.predict(
        stacked_observation.reshape(1, -1), 
        deterministic=DETERMINISTIC_PREDICTION
    )
    action_int = int(action[0])
    action_name = ACTION_MAP.get(action_int, 'NOTHING')

    did_crash = bool(
        request_data.get("did_crash")
        or request_data.get("didCrash")
        or request_data.get("crashed")
    )
    if did_crash:
        print(f"Crash detected at tick {request_data.get('elapsed_ticks')}. Logging diagnostics...")
        try:
            _log_crash_data(request_data, action_name)
        except Exception as e:
            # don't lose the action due to logging issues
            print(f"[WARN] Crash logging failed: {e}")
        reset_state()
    
    # 6. Handle crash logic: log data and reset state for the next run
    if request_data.get('did_crash', False):
        print(f"Crash detected at tick {request_data.get('elapsed_ticks')}. Logging diagnostics...")
        _log_crash_data(request_data, action_name)
        reset_state()

    # Per the DTO, return a list of actions
    return [action_name]