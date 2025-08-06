# api_sb3.py — Serve PPO policy for server-side verification
import os
import pickle
from collections import deque
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from stable_baselines3 import PPO

# Optional: if we can import SB3 VecNormalize to load stats for obs standardization
try:
    from stable_baselines3.common.vec_env import VecNormalize
    _HAS_VECNORM = True
except Exception:
    _HAS_VECNORM = False

from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto

# ----------------------------- Config -----------------------------
MODEL_SAVE_DIR = "models_sb3"
MODEL_BASENAME = "race_car_ppo_cuda_parallel"  # must match training
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_BASENAME)
VECNORM_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_BASENAME}_vecnormalize.pkl")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

# Match training
N_STACK = 3            # VecFrameStack(n_stack=3) in train
BASE_OBS_DIM = 18      # 16 sensors + vx + vy
STACKED_OBS_DIM = N_STACK * BASE_OBS_DIM

# How many actions to return per call
ACTION_BUFFER_LEN = int(os.environ.get("ACTION_BUFFER_LEN", "12"))

# Action name mapping: pull from your env if available, else fallback
try:
    from src.game.racecar_env import ACTIONS as ENV_ACTIONS, ACTION_MAP as ENV_ACTION_MAP
    _HAS_ENV_ACTIONS = True
except Exception:
    _HAS_ENV_ACTIONS = False
    # Fallback order: make sure this matches *your* training order if you use this path.
    # If your env exposes ACTIONS, the server will use those instead.
    ENV_ACTIONS = ["NOTHING", "ACCELERATE", "DECELERATE", "STEER_LEFT", "STEER_RIGHT"]
    ENV_ACTION_MAP = {i: a for i, a in enumerate(ENV_ACTIONS)}

# Sensor order must match the game's creation order used in training
SENSORS_ORDER = [
    "front", "right_front", "right_side", "right_back", "back", "left_back",
    "left_side", "left_front", "left_side_front", "front_left_front",
    "front_right_front", "right_side_front", "right_side_back",
    "back_right_back", "back_left_back", "left_side_back",
]

# ------------------------ Helper: Obs Normalizer -------------------
class ObsNormalizer:
    """
    Minimal loader for VecNormalize obs statistics so we can standardize observations
    without constructing a real VecEnv server-side.
    """
    def __init__(self, vecnorm_path: str, clip_obs: float = 10.0):
        self.enabled = False
        self.mean = None
        self.var = None
        self.clip_obs = clip_obs
        if _HAS_VECNORM and os.path.exists(vecnorm_path):
            try:
                # SB3's VecNormalize.load expects a VecEnv. We only need the stats.
                # We'll unpickle to grab obs_rms.{mean,var} safely.
                with open(vecnorm_path, "rb") as f:
                    vn = pickle.load(f)
                # The pickled object is a VecNormalize; pull stats
                self.mean = np.array(vn.obs_rms.mean, dtype=np.float32)
                self.var = np.array(vn.obs_rms.var, dtype=np.float32)
                self.clip_obs = float(getattr(vn, "clip_obs", clip_obs))
                self.enabled = True
                print(f"[ObsNorm] Loaded stats: clip_obs={self.clip_obs}, "
                      f"mean.shape={self.mean.shape}, var.shape={self.var.shape}")
            except Exception as e:
                print(f"[ObsNorm] Could not load VecNormalize stats from {vecnorm_path}: {e}")
                self.enabled = False

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return obs
        # Expect (1, STACKED_OBS_DIM)
        assert obs.ndim == 2, f"Expected (batch, dim), got {obs.shape}"
        # Mean/var are for base-dim; we need to tile to stacked shape
        base_mean = self.mean.reshape(1, BASE_OBS_DIM)
        base_var = self.var.reshape(1, BASE_OBS_DIM)
        mean = np.tile(base_mean, (1, N_STACK))
        var = np.tile(base_var, (1, N_STACK))
        eps = 1e-8
        norm = (obs - mean) / np.sqrt(var + eps)
        return np.clip(norm, -self.clip_obs, self.clip_obs)

# --------------------------- Predictor -----------------------------
class PPOPredictor:
    def __init__(self, model_path: str, vecnorm_path: Optional[str]):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Model] Loading PPO on device={device}")
        # Allow .zip or bare path
        path = model_path if os.path.isfile(model_path) else model_path + ".zip"
        self.model = PPO.load(path, device=device)

        # Stacked history for VecFrameStack emulation
        self._hist = deque(maxlen=N_STACK)

        # Best-effort obs normalization (optional)
        self.obs_norm = ObsNormalizer(vecnorm_path, clip_obs=10.0)

    def _build_base_obs(self, req: RaceCarPredictRequestDto) -> np.ndarray:
        """
        Build the base (unstacked) observation vector exactly like training:
        [16 sensors in training order, velocity.x, velocity.y]
        """
        s = req.sensors or {}
        # Values are normalized distances in [0, 1] already
        sensors = [float(s.get(name, 1.0) if s.get(name) is not None else 1.0) for name in SENSORS_ORDER]
        vx = float(req.velocity.get("x", 0.0))
        vy = float(req.velocity.get("y", 0.0))
        base = np.array(sensors + [vx, vy], dtype=np.float32)
        assert base.shape == (BASE_OBS_DIM,), f"Bad base obs shape: {base.shape}"
        return base

    def _stack(self, base_obs: np.ndarray) -> np.ndarray:
        """Emulate VecFrameStack(n_stack=N_STACK) over 1 env."""
        if len(self._hist) == 0:
            for _ in range(N_STACK):
                self._hist.append(base_obs.copy())
        else:
            self._hist.append(base_obs)
        stacked = np.concatenate(list(self._hist), dtype=np.float32)
        assert stacked.shape == (STACKED_OBS_DIM,), f"Bad stacked shape: {stacked.shape}"
        return stacked

    def predict_actions(self, req: RaceCarPredictRequestDto, buffer_len: int = ACTION_BUFFER_LEN) -> List[str]:
        """
        Produce a short buffer of actions. We compute the current best action and repeat it.
        Keeping it reactive per call is consistent with the platform’s batching guidance.
        """
        base = self._build_base_obs(req)
        stacked = self._stack(base)                 # (STACKED_OBS_DIM,)
        obs = stacked.reshape(1, -1)                # (1, dim)

        # Normalize if stats are available
        obs = self.obs_norm(obs)

        # Model expects a vectorized obs; predict returns action int
        act_int, _ = self.model.predict(obs, deterministic=True)
        a_int = int(np.asarray(act_int).squeeze())

        # Map to action string name
        if _HAS_ENV_ACTIONS:
            # Try ACTIONS first (list), else ACTION_MAP (dict)
            if 0 <= a_int < len(ENV_ACTIONS):
                a_name = str(ENV_ACTIONS[a_int])
            else:
                a_name = str(ENV_ACTION_MAP.get(a_int, "NOTHING"))
        else:
            # Fallback mapping
            a_name = ENV_ACTION_MAP.get(a_int, "NOTHING")

        # Return a small buffer of the same action to fill the server’s queue efficiently
        return [a_name] * buffer_len

# ----------------------------- API -------------------------------
app = FastAPI(title="Race Car PPO API", version="1.0")

# CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: Optional[PPOPredictor] = None

@app.on_event("startup")
def _load_model():
    global predictor
    predictor = PPOPredictor(MODEL_PATH, VECNORM_PATH)
    print("[Startup] Predictor ready.")

@app.get("/")
def root():
    return {"status": "ok", "message": "Race Car PPO endpoint running."}

@app.post("/predict", response_model=RaceCarPredictResponseDto)
def predict(req: RaceCarPredictRequestDto):
    if predictor is None:
        # Should not happen after startup, but guard anyway
        actions = ["NOTHING"] * ACTION_BUFFER_LEN
        return RaceCarPredictResponseDto(actions=actions)
    actions = predictor.predict_actions(req, buffer_len=ACTION_BUFFER_LEN)
    return RaceCarPredictResponseDto(actions=actions)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
