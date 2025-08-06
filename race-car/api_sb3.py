import os
import time
import datetime
import logging

import uvicorn
import torch
from fastapi import Body, FastAPI, HTTPException
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack

from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto
from example import return_action
from src.game.racecar_env import RaceCarEnv

# Server configuration
HOST = "0.0.0.0"
PORT = int(os.getenv("API_PORT", 9052))

# FastAPI app and uptime tracking
app = FastAPI()
start_time = time.time()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("api")

@app.post("/predict", response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):
    logger.info(
        "Predict request received: did_crash=%s, elapsed_ticks=%d, distance=%.2f",
        request.did_crash, request.elapsed_ticks, request.distance
    )
    try:
        result = return_action(request.dict())
        # Handle both list return or dict with 'actions'
        if isinstance(result, list):
            actions = result
            logger.debug("Action list returned by model: %s", actions)
        elif isinstance(result, dict) and "actions" in result:
            actions = result["actions"]
            logger.debug("Actions extracted from dict result: %s", actions)
        else:
            logger.error("Unexpected result format from return_action: %s", result)
            raise ValueError(f"Unexpected result format: {result}")
        return RaceCarPredictResponseDto(actions=actions)
    except Exception as e:
        logger.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api")
def hello():
    uptime = datetime.timedelta(seconds=time.time() - start_time)
    return {
        "service": "race-car-usecase",
        "uptime": str(uptime)
    }

@app.get("/")
def index():
    return "Your endpoint is running!"

@app.post("/evaluate")
def evaluate_model(episodes: int = 10):
    """
    Run model evaluation for a number of episodes and return summary metrics.
    Based on offline evaluation logic (see eval_gpu_model.py).
    """
    MODEL_SAVE_DIR = "models_sb3"
    MODEL_BASENAME = "race_car_ppo_cuda_parallel"
    model_path = os.path.join(MODEL_SAVE_DIR, MODEL_BASENAME)
    vecnorm_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_BASENAME}_vecnormalize.pkl")

    logger.info("Starting evaluation: episodes=%d", episodes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading model from %s on device %s", model_path, device)
    model = PPO.load(model_path, device=device)

    venv = make_vec_env(RaceCarEnv, n_envs=1, seed=0)
    if os.path.exists(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
        logger.info("Loaded VecNormalize stats from %s", vecnorm_path)
    else:
        logger.warning("VecNormalize stats not found at %s; proceeding without normalization", vecnorm_path)

    n_stack = 3
    venv = VecFrameStack(venv, n_stack=n_stack)
    logger.info("Applied VecFrameStack(n_stack=%d)", n_stack)

    returns, distances = [], []
    for ep in range(1, episodes + 1):
        obs = venv.reset()
        done = False
        total_reward = 0.0
        info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            total_reward += float(reward[0])
            done = bool(dones[0])
            info = infos[0]

        distance = info.get("distance", 0.0)
        returns.append(total_reward)
        distances.append(distance)
        logger.info(
            "Episode %d complete: total_reward=%.2f, distance=%.2f", ep, total_reward, distance
        )

    avg_return = sum(returns) / len(returns)
    avg_distance = sum(distances) / len(distances)
    result = {"episodes": episodes, "average_return": avg_return, "average_distance": avg_distance}

    logger.info("Evaluation complete: %s", result)
    return result

if __name__ == "__main__":
    uvicorn.run("api:app", host=HOST, port=PORT)
