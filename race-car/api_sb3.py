import os
import time
import math
import datetime
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import Body, FastAPI

try:
    import cloudpickle as pickle  # SB3 uses cloudpickle
except Exception:  # pragma: no cover
    import pickle  # type: ignore

import torch
from stable_baselines3 import PPO

from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto

# --------------------------------------------------------------------------------------
# Server config
# --------------------------------------------------------------------------------------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9052"))

MODEL_DIR = os.getenv("MODEL_SAVE_DIR", "models_sb3")
MODEL_NAME = os.getenv("MODEL_NAME", "race_car_ppo_cuda_parallel")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODEL_DIR, MODEL_NAME))  # we'll also try .zip
VECNORM_PATH = os.getenv("VECNORM_PATH", os.path.join(MODEL_DIR, f"{MODEL_NAME}_vecnormalize.pkl"))

# Match training/eval wrappers: VecNormalize -> VecFrameStack(n_stack=3)
N_STACK = int(os.getenv("N_STACK", "3"))
CLIP_OBS_DEFAULT = float(os.getenv("CLIP_OBS", "10.0"))  # fallback if not in vecnorm file

# How many steps to return per request (helps with network latency on the tournament server)
# You can tune this after validating queue latency. 8-16 is a common range.
BATCH_ACTIONS = int(os.getenv("BATCH_ACTIONS", "12"))

# Action names must match the game server
ACTIONS: List[str] = [
    "ACCELERATE",
    "DECELERATE",
    "STEER_LEFT",
    "STEER_RIGHT",
    "NOTHING",
]
ACTION_MAP: Dict[int, str] = {i: a for i, a in enumerate(ACTIONS)}

# Sensor order MUST match the environment used in training (see RaceCarEnv/_get_obs)
# Index map (0..15):
# 0: front, 1: right_front, 2: right_side, 3: right_back, 4: back, 5: left_back,
# 6: left_side, 7: left_front, 8: left_side_front, 9: front_left_front,
# 10: front_right_front, 11: right_side_front, 12: right_side_back,
# 13: back_right_back, 14: back_left_back, 15: left_side_back
ORDERED_SENSORS: List[str] = [
    "front",
    "right_front",
    "right_side",
    "right_back",
    "back",
    "left_back",
    "left_side",
    "left_front",
    "left_side_front",
    "front_left_front",
    "front_right_front",
    "right_side_front",
    "right_side_back",
    "back_right_back",
    "back_left_back",
    "left_side_back",
]


# --------------------------------------------------------------------------------------
# Inference engine — reproduces training-time preprocessing on the server
#   Training pipeline: VecNormalize(norm_obs=True, clip_obs=10) -> VecFrameStack(n_stack=3)
#   We mirror this by: (1) build obs (env scaling) (2) normalize using saved VecNormalize stats
#                      (3) keep a rolling stack of last N normalized frames and feed PPO
# --------------------------------------------------------------------------------------
class _VecNormStats:
    __slots__ = ("mean", "var", "eps", "clip_obs")

    def __init__(self, mean: np.ndarray, var: np.ndarray, eps: float, clip_obs: float):
        self.mean = mean.astype(np.float32)
        self.var = var.astype(np.float32)
        self.eps = float(eps)
        self.clip_obs = float(clip_obs)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        # (x - mean) / sqrt(var + eps)
        out = (x - self.mean) / np.sqrt(np.maximum(self.var, self.eps))
        # Clip to clip_obs like SB3 does
        if self.clip_obs is not None and math.isfinite(self.clip_obs):
            np.clip(out, -self.clip_obs, self.clip_obs, out=out)
        return out.astype(np.float32, copy=False)


def _resolve_model_path(path: str) -> str:
    if os.path.isfile(path):
        return path
    zipped = path + ".zip"
    if os.path.isfile(zipped):
        return zipped
    raise FileNotFoundError(f"Model not found at '{path}' or '{zipped}'")


def _enable_cuda_fastpaths() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


class InferenceEngine:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        _enable_cuda_fastpaths()

        # Load PPO
        mdl_path = _resolve_model_path(MODEL_PATH)
        self.model = PPO.load(mdl_path, device=self.device)

        # Load VecNormalize stats saved during training
        self.vec_stats = self._load_vecnorm(VECNORM_PATH)

        # Rolling stack of normalized observations
        self.n_stack = int(max(1, N_STACK))
        self.stack: deque[np.ndarray] = deque(maxlen=self.n_stack)
        self._initialized = False

        # Episode tracking to reset stack appropriately
        self._last_tick: Optional[int] = None

        # Sanity log
        obs_dim = self.vec_stats.mean.shape[0] if self.vec_stats else 18
        model_obs_dim = int(np.prod(self.model.observation_space.shape))
        expected_model_dim = obs_dim * self.n_stack
        if model_obs_dim != expected_model_dim:
            # If this prints, the model was trained with a different stack size
            print(
                f"[WARN] Model expects obs_dim={model_obs_dim}, but our pipeline builds {expected_model_dim}.\n"
                f"       Adjust N_STACK or verify the trained policy wrappers."
            )

    # ---------------------------- VecNormalize loader ----------------------------
    def _load_vecnorm(self, path: str) -> Optional[_VecNormStats]:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Could not load VecNormalize stats from '{path}': {e}\n"
                  f"       Proceeding WITHOUT observation normalization (will likely underperform).")
            # Return an identity transform
            return _VecNormStats(mean=np.zeros(18, dtype=np.float32),
                                 var=np.ones(18, dtype=np.float32),
                                 eps=1e-8,
                                 clip_obs=CLIP_OBS_DEFAULT)

        # SB3 stores a dict-like payload; 'obs_rms' holds RunningMeanStd
        obs_rms = data.get("obs_rms") or data.get("ob_rms")
        clip_obs = float(data.get("clip_obs", CLIP_OBS_DEFAULT))

        # Fallback if structure is unexpected
        if obs_rms is None:
            print("[WARN] VecNormalize payload missing 'obs_rms' — using identity stats.")
            return _VecNormStats(mean=np.zeros(18, dtype=np.float32),
                                 var=np.ones(18, dtype=np.float32),
                                 eps=float(data.get("epsilon", 1e-8)),
                                 clip_obs=clip_obs)

        # RunningMeanStd may be an object with .mean/.var/.epsilon
        mean = np.asarray(getattr(obs_rms, "mean", np.zeros(18)), dtype=np.float32)
        var = np.asarray(getattr(obs_rms, "var", np.ones(18)), dtype=np.float32)
        eps = float(getattr(obs_rms, "epsilon", data.get("epsilon", 1e-8)))

        return _VecNormStats(mean=mean, var=var, eps=eps, clip_obs=clip_obs)

    # ---------------------------- Request → obs ----------------------------
    @staticmethod
    def _build_raw_obs(req: RaceCarPredictRequestDto) -> np.ndarray:
        """Recreate RaceCarEnv._get_obs() scaling on the server.
        - sensors: 16 names → distances in px, None→1000, then divide by 1000
        - velocity: x/20, y/2
        Finally clip to [-1, 1] like the Gym env does.
        """
        sensors = req.sensors or {}
        readings: List[float] = []
        for name in ORDERED_SENSORS:
            v = sensors.get(name, None)
            if v is None:
                v = 1000.0
            try:
                val = float(v)
            except Exception:
                val = 1000.0
            readings.append(val / 1000.0)

        vx = float(req.velocity.get("x", 0.0)) / 20.0
        vy = float(req.velocity.get("y", 0.0)) / 2.0

        obs = np.asarray(readings + [vx, vy], dtype=np.float32)
        np.clip(obs, -1.0, 1.0, out=obs)
        return obs

    def _maybe_reset_stack(self, req: RaceCarPredictRequestDto, obs_norm: np.ndarray) -> None:
        """Reset frame stack at episode boundaries (tick resets or explicit crash).
        When resetting, SB3's VecFrameStack duplicates the current frame to fill the stack.
        """
        episode_reset = False
        if self._last_tick is None:
            episode_reset = True
        elif req.did_crash:
            episode_reset = True
        elif req.elapsed_ticks is not None and self._last_tick is not None and req.elapsed_ticks <= self._last_tick:
            # New game or tick reset
            episode_reset = True

        if episode_reset or not self._initialized:
            self.stack.clear()
            for _ in range(self.n_stack):
                self.stack.append(obs_norm.copy())
            self._initialized = True
        else:
            self.stack.append(obs_norm.copy())

        self._last_tick = int(req.elapsed_ticks)

    # ---------------------------- Predict ----------------------------
    def predict_actions(self, req: RaceCarPredictRequestDto) -> List[str]:
        raw_obs = self._build_raw_obs(req)  # 18-dim
        obs_norm = self.vec_stats.normalize(raw_obs) if self.vec_stats else raw_obs
        self._maybe_reset_stack(req, obs_norm)

        stacked = np.concatenate(list(self.stack), dtype=np.float32)  # 18 * n_stack

        # SB3 PPO accepts (obs_dim,) or (batch, obs_dim)
        action, _ = self.model.predict(stacked, deterministic=True)
        try:
            a_int = int(np.asarray(action).squeeze())
        except Exception:
            a_int = 4  # NOTHING (robust fallback)
        a_int = int(np.clip(a_int, 0, len(ACTIONS) - 1))
        a_name = ACTION_MAP.get(a_int, "NOTHING")

        return [a_name for _ in range(BATCH_ACTIONS)]


# --------------------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------------------
app = FastAPI()
start_time = time.time()
ENGINE = InferenceEngine()


@app.post("/predict", response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):
    """Return a *batch* of actions predicted by the PPO policy.
    This mirrors the exact preprocessing used offline (scaling → VecNormalize → FrameStack).
    """
    actions = ENGINE.predict_actions(request)
    return RaceCarPredictResponseDto(actions=actions)


@app.get("/api")
def hello():
    return {
        "service": "race-car-usecase",
        "uptime": str(datetime.timedelta(seconds=time.time() - start_time)),
        "device": ENGINE.device,
        "model_path": _resolve_model_path(MODEL_PATH),
        "vecnorm_path": VECNORM_PATH,
        "n_stack": ENGINE.n_stack,
        "batch_actions": BATCH_ACTIONS,
    }


@app.get("/")
def index():
    return "Your endpoint is running!"


@app.post("/reload")
def reload_model():  # optional helper for hot-reloading model files without restarting the server
    global ENGINE
    ENGINE = InferenceEngine()
    return {"status": "ok", "reloaded": True}


if __name__ == "__main__":
    # When running locally: `python api.py`
    uvicorn.run(
        "api:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )
