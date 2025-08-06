import time
import uvicorn
import datetime
import os
import collections

import numpy as np
import torch
from fastapi import Body, FastAPI
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto
from src.game.racecar_env import RaceCarEnv


# --- Configuration ---
HOST = "0.0.0.0"
PORT = 8000

# --- Model and Environment Configuration ---
MODEL_SAVE_DIR = "models_sb3"
MODEL_BASENAME = "race_car_ppo_cuda_parallel"
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_BASENAME}.zip")
VECNORM_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_BASENAME}_vecnormalize.pkl")
N_STACK = 3  # Must match the n_stack used in training (gpu_train.py)

# --- Prevent Pygame from trying to open a display on the server ---
os.environ["SDL_VIDEODRIVER"] = "dummy"

# --- Define action and sensor order based on training environment ---
ACTIONS = ['ACCELERATE', 'DECELERATE', 'STEER_LEFT', 'STEER_RIGHT', 'NOTHING']
ACTION_MAP = {i: action for i, action in enumerate(ACTIONS)}

# This order is derived from 'sensor_options' in 'core.py' as used by 'racecar_env.py' during training.
# It is critical that this order is maintained for the model to interpret observations correctly.
SENSOR_ORDER = [
    "front", "right_front", "right_side", "right_back", "back", "left_back",
    "left_side", "left_front", "left_side_front", "front_left_front",
    "front_right_front", "right_side_front", "right_side_back",
    "back_right_back", "back_left_back", "left_side_back"
]
OBS_SHAPE = (len(SENSOR_ORDER) + 2,) # 16 sensors + 2 velocity components (vx, vy)


# --- Global State for Model, Environment, and Frame Stacking ---
app = FastAPI()
start_time = time.time()
model = None
venv = None
startup_error_message = "" # Stores any error message that occurs during model loading

# A deque to hold the last N_STACK observations for a single, ongoing game instance.
stacked_obs_deque = collections.deque(
    [np.zeros(OBS_SHAPE, dtype=np.float32) for _ in range(N_STACK)],
    maxlen=N_STACK
)

@app.on_event("startup")
def load_model_and_env():
    """
    Load the PPO model and VecNormalize statistics when the server starts.
    This is done once to avoid loading from disk on every prediction request.
    """
    global model, venv, startup_error_message
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        if not os.path.exists(VECNORM_PATH):
            raise FileNotFoundError(f"VecNormalize stats not found at: {VECNORM_PATH}")

        # Load the trained PPO model
        model = PPO.load(MODEL_PATH, device=device)
        print("Model loaded successfully.")

        # We need a dummy env to load the VecNormalize stats into.
        # It is used for its .normalize_obs() method, not for simulation.
        dummy_env = make_vec_env(RaceCarEnv, n_envs=1)
        venv = VecNormalize.load(VECNORM_PATH, dummy_env)
        venv.training = False
        venv.norm_reward = False
        print("VecNormalize stats loaded successfully.")

    except Exception as e:
        startup_error_message = str(e)
        print(f"FATAL: Could not load model or environment. {startup_error_message}")
        print("The '/predict' endpoint will not work. Please ensure model files are present and all dependencies are installed.")
        model = None
        venv = None

def reset_frame_stack():
    """Resets the observation deque with zeros, called at the start of a new episode."""
    global stacked_obs_deque
    stacked_obs_deque.clear()
    for _ in range(N_STACK):
        stacked_obs_deque.append(np.zeros(OBS_SHAPE, dtype=np.float32))
    print("Frame stack reset for new episode.")

@app.post('/predict', response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):
    """
    Predicts the next action based on the current game state.
    """
    global stacked_obs_deque
    if model is None or venv is None:
        # Fallback action if the model failed to load
        return RaceCarPredictResponseDto(actions=['NOTHING'])

    # A crash signals the end of an episode. Reset the frame stack for the next game.
    if request.did_crash:
        reset_frame_stack()

    # 1. Construct the raw observation vector from the request data.
    # Handle missing sensors by substituting a large value (1000.0), which corresponds to "no obstacle".
    try:
        sensor_readings = [request.sensors.get(name, 1000.0) or 1000.0 for name in SENSOR_ORDER]
        raw_obs_list = sensor_readings + [request.velocity['x'], request.velocity['y']]
        raw_obs = np.array(raw_obs_list, dtype=np.float32).reshape(1, -1) # Reshape for VecNormalize
    except (TypeError, KeyError) as e:
        print(f"Error parsing request data: {e}. Returning NOTHING.")
        return RaceCarPredictResponseDto(actions=['NOTHING'])


    # 2. Normalize the observation using the loaded VecNormalize stats.
    # VecNormalize expects a batch of observations, so we pass a (1, 18) array
    # and get a (1, 18) normalized array back.
    normalized_obs = venv.normalize_obs(raw_obs)

    # 3. Update the frame stack deque with the new normalized observation.
    stacked_obs_deque.append(normalized_obs.squeeze()) # Squeeze to shape (18,)

    # 4. Prepare the final model input by flattening the deque.
    # The model expects an observation of shape (n_envs, features * n_stack), e.g., (1, 54).
    model_input = np.array(list(stacked_obs_deque)).flatten().reshape(1, -1)

    # 5. Predict the action using the model.
    action_int, _ = model.predict(model_input, deterministic=True)
    action_str = ACTION_MAP.get(int(action_int[0]), 'NOTHING')

    # 6. Return the predicted action in the response DTO.
    return RaceCarPredictResponseDto(
        actions=[action_str]
    )

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
    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT,
        reload=True # Use reload for development to see changes without restarting
    )