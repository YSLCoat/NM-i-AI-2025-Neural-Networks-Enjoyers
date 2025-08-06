import time
import uvicorn
import datetime
import os
import collections
import logging
from logging.handlers import FileHandler

import numpy as np
import torch
from fastapi import Body, FastAPI
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# CORRECTED: The missing import is now added here
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto
from src.game.racecar_env import RaceCarEnv


# --- Configure Logging ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, "app.log")

# Setup logger to output to both console and file
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO) # Change to logging.DEBUG for detailed reward breakdowns

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

# File Handler
file_handler = FileHandler(log_path)
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)


# --- Configuration ---
HOST = "0.0.0.0"
PORT = 8000

# --- Model and Environment Configuration ---
MODEL_SAVE_DIR = "models_sb3"
MODEL_BASENAME = "race_car_ppo_cuda_parallel"
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_BASENAME}.zip")
VECNORM_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_BASENAME}_vecnormalize.pkl")
N_STACK = 3

os.environ["SDL_VIDEODRIVER"] = "dummy"

ACTIONS = ['ACCELERATE', 'DECELERATE', 'STEER_LEFT', 'STEER_RIGHT', 'NOTHING']
ACTION_MAP = {i: action for i, action in enumerate(ACTIONS)}

SENSOR_ORDER = [
    "front", "right_front", "right_side", "right_back", "back", "left_back",
    "left_side", "left_front", "left_side_front", "front_left_front",
    "front_right_front", "right_side_front", "right_side_back",
    "back_right_back", "back_left_back", "left_side_back"
]
OBS_SHAPE = (len(SENSOR_ORDER) + 2,)


# --- Global State ---
app = FastAPI()
start_time = time.time()
model = None
venv = None
startup_error_message = ""

stacked_obs_deque = collections.deque(
    [np.zeros(OBS_SHAPE, dtype=np.float32) for _ in range(N_STACK)],
    maxlen=N_STACK
)

@app.on_event("startup")
def load_model_and_env():
    global model, venv, startup_error_message
    logging.info("==================================================")
    logging.info("              STARTING API SERVER                 ")
    logging.info("==================================================")
    logging.info(f"Attempting to load model '{MODEL_BASENAME}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        if not os.path.exists(VECNORM_PATH):
            raise FileNotFoundError(f"VecNormalize stats not found at: {VECNORM_PATH}")

        model = PPO.load(MODEL_PATH, device=device)
        logging.info("PPO model loaded successfully.")

        dummy_env = make_vec_env(RaceCarEnv, n_envs=1)
        venv = VecNormalize.load(VECNORM_PATH, dummy_env)
        venv.training = False
        venv.norm_reward = False
        logging.info("VecNormalize stats loaded successfully.")
        logging.info("--- Model and environment are ready ---")

    except Exception as e:
        startup_error_message = str(e)
        logging.critical(f"Could not load model or environment. Reason: {startup_error_message}")
        logging.critical("The '/predict' endpoint will default to 'NOTHING'.")
        model = None
        venv = None

def reset_frame_stack():
    global stacked_obs_deque
    stacked_obs_deque.clear()
    for _ in range(N_STACK):
        stacked_obs_deque.append(np.zeros(OBS_SHAPE, dtype=np.float32))
    logging.info("--- EPISODE START --- Frame stack has been reset.")

@app.post('/predict', response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):
    global stacked_obs_deque
    logging.debug(f"Received request: did_crash={request.did_crash}, distance={request.distance:.2f}")

    if model is None or venv is None:
        logging.warning("Model not loaded, returning 'NOTHING'.")
        return RaceCarPredictResponseDto(actions=['NOTHING'])

    if request.did_crash:
        reset_frame_stack()

    try:
        sensor_readings = [request.sensors.get(name, 1000.0) or 1000.0 for name in SENSOR_ORDER]
        raw_obs_list = sensor_readings + [request.velocity['x'], request.velocity['y']]
        raw_obs = np.array(raw_obs_list, dtype=np.float32).reshape(1, -1)
    except (TypeError, KeyError) as e:
        logging.error(f"Error parsing request data: {e}. Returning 'NOTHING'.")
        return RaceCarPredictResponseDto(actions=['NOTHING'])

    normalized_obs = venv.normalize_obs(raw_obs)
    stacked_obs_deque.append(normalized_obs.squeeze())
    model_input = np.array(list(stacked_obs_deque)).flatten().reshape(1, -1)
    logging.debug(f"Model input shape: {model_input.shape}")

    action_int, _ = model.predict(model_input, deterministic=True)
    action_str = ACTION_MAP.get(int(action_int[0]), 'NOTHING')
    logging.info(f"Tick: {request.elapsed_ticks}, Distance: {request.distance:.2f}m, Predicted Action: {action_str}")

    return RaceCarPredictResponseDto(actions=[action_str])

# Other endpoints remain the same...
@app.get('/status')
def status():
    """Provides the operational status of the model and environment."""
    if model and venv:
        return {"status": "ok", "message": "Model and environment loaded successfully."}
    else:
        return {
            "status": "error",
            "message": "Model or environment failed to load.",
            "details": startup_error_message
        }

@app.get('/api')
def hello():
    return {
        "service": "race-car-usecase",
        "uptime": '{}'.format(datetime.timedelta(seconds=time.time() - start_time))
    }

@app.get('/')
def index():
    return "Your endpoint is running!"

if __name__ == '__main__':
    uvicorn.run('__main__:app', host=HOST, port=PORT, reload=True)