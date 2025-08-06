# api_sb3.py — Serve PPO policy for server-side verification
import os
import pickle
import logging # --- LOGGING ADDED ---
from datetime import datetime # --- LOGGING ADDED ---
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

# ----------------------------- Logging Setup -----------------------------
# --- LOGGING ADDED ---
# Create a logger
logger = logging.getLogger("api_predictor")
logger.setLevel(logging.INFO)

# Create handlers
log_file_path = "predictor_log.txt"
file_handler = logging.FileHandler(log_file_path)
console_handler = logging.StreamHandler()

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
# --- END LOGGING ADDED ---


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
ACTION_BUFFER_LEN = int(os.environ.get("ACTION_BUFFER_LEN", "4"))

# Action name mapping: pull from your env if available, else fallback
try:
    from src.game.racecar_env import ACTIONS as ENV_ACTIONS, ACTION_MAP as ENV_ACTION_MAP
    _HAS_ENV_ACTIONS = True
except Exception:
    _HAS_ENV_ACTIONS = False
    ENV_ACTIONS = ["NOTHING", "ACCELERATE", "DECELERATE", "STEER_LEFT", "STEER_RIGHT"]
    ENV_ACTION_MAP = {i: a for i, a in enumerate(ENV_ACTIONS)}

# Sensor order must match the game's creation order used in training
SENSORS_ORDER = [
    "front", "right_front", "right_side", "right_back", "back", "left_back",
    "left_side", "left_front", "left_side_front", "front_left_front",
    "front_right_front", "right_side_front", "right_side_back",
    "back_right_back", "back_left_back", "left_side_back",
]

class ObsNormalizer:
    # This class is now simplified to only normalize the base 18-dim observation
    def __init__(self, vecnorm_path: str, clip_obs: float = 10.0):
        self.enabled = False
        self.mean = None
        self.var = None
        self.clip_obs = clip_obs
        if _HAS_VECNORM and os.path.exists(vecnorm_path):
            try:
                with open(vecnorm_path, "rb") as f:
                    vn = pickle.load(f)
                # Ensure mean/var are shaped correctly for the base observation
                self.mean = np.array(vn.obs_rms.mean, dtype=np.float32).reshape(1, BASE_OBS_DIM)
                self.var = np.array(vn.obs_rms.var, dtype=np.float32).reshape(1, BASE_OBS_DIM)
                self.clip_obs = float(getattr(vn, "clip_obs", clip_obs))
                self.enabled = True
                logger.info(f"[ObsNorm] Loaded stats: clip_obs={self.clip_obs}, "
                      f"mean.shape={self.mean.shape}, var.shape={self.var.shape}")
            except Exception as e:
                logger.error(f"[ObsNorm] Could not load VecNormalize stats from {vecnorm_path}: {e}")
                self.enabled = False

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return obs
        # Expects a single, unstacked observation of shape (BASE_OBS_DIM,)
        assert obs.shape == (BASE_OBS_DIM,), f"Expected ({BASE_OBS_DIM},), got {obs.shape}"
        
        # Reshape to (1, BASE_OBS_DIM) for broadcasting
        obs = obs.reshape(1, BASE_OBS_DIM)
        
        eps = 1e-8
        norm = (obs - self.mean) / np.sqrt(self.var + eps)
        clipped = np.clip(norm, -self.clip_obs, self.clip_obs)
        
        # Return the normalized, unstacked observation, flattened back
        return clipped.flatten()

# --------------------------- Predictor -----------------------------
class PPOPredictor:
    def __init__(self, model_path: str, vecnorm_path: Optional[str]):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[Model] Loading PPO on device={device}")
        path = model_path if os.path.isfile(model_path) else model_path + ".zip"
        self.model = PPO.load(path, device=device)
        self._hist = deque(maxlen=N_STACK)
        self.obs_norm = ObsNormalizer(vecnorm_path, clip_obs=10.0)

    def _build_base_obs(self, req: RaceCarPredictRequestDto) -> np.ndarray:
        s = req.sensors or {}
        
        sensors = []
        for name in SENSORS_ORDER:
            raw_reading = s.get(name)
            reading = float(raw_reading) if raw_reading is not None else 1000.0
            sensors.append(reading / 1000.0)

        vx = float(req.velocity.get("x", 0.0))
        vy = float(req.velocity.get("y", 0.0))
        
        base = np.array(sensors + [vx, vy], dtype=np.float32)
        
        assert base.shape == (BASE_OBS_DIM,), f"Bad base obs shape: {base.shape}"
        return base

    def _stack(self, normalized_base_obs: np.ndarray) -> np.ndarray:
        # This now expects an already-normalized observation
        if len(self._hist) == 0:
            for _ in range(N_STACK):
                self._hist.append(normalized_base_obs.copy())
        else:
            self._hist.append(normalized_base_obs)
        stacked = np.concatenate(list(self._hist), dtype=np.float32)
        assert stacked.shape == (STACKED_OBS_DIM,), f"Bad stacked shape: {stacked.shape}"
        return stacked

    def predict_actions(self, req: RaceCarPredictRequestDto, buffer_len: int = ACTION_BUFFER_LEN) -> List[str]:
            try:
                logger.info("=" * 50)
                logger.info(
                    f"Received request: distance={req.distance:.2f}, "
                    f"ticks={req.elapsed_ticks}, crashed={req.did_crash}"
                )

                # -------------------- 1) Build base observation (unnormalized) --------------------
                base_obs = self._build_base_obs(req)  # shape (BASE_OBS_DIM,)
                logger.info(
                    f"Step 1: base_obs | shape={base_obs.shape}, "
                    f"min={base_obs.min():.3f}, max={base_obs.max():.3f}, mean={base_obs.mean():.3f}"
                )

                # -------------------- 2) Normalize the base observation --------------------------
                # Many ObsNormalizer impls expect (batch, dim). Normalize per-frame, then stack.
                try:
                    base_obs_2d = base_obs.reshape(1, -1)
                    norm_base_obs_2d = self.obs_norm(base_obs_2d)  # expected shape (1, BASE_OBS_DIM)
                    # Be tolerant if obs_norm returns 1D
                    if norm_base_obs_2d is None:
                        norm_base_obs = base_obs.astype(np.float32)
                    else:
                        norm_base_obs = np.asarray(norm_base_obs_2d).reshape(-1).astype(np.float32)
                except Exception as e_norm:
                    logger.warning(f"Step 2: normalization failed ({e_norm}); using unnormalized base.")
                    norm_base_obs = base_obs.astype(np.float32)

                logger.info(
                    f"Step 2: norm_base_obs | shape={norm_base_obs.shape}, "
                    f"min={norm_base_obs.min():.3f}, max={norm_base_obs.max():.3f}, mean={norm_base_obs.mean():.3f}"
                )

                # -------------------- 3) Stack normalized observations ---------------------------
                # Emulate VecFrameStack: append this normalized frame to history
                stacked_norm_obs = self._stack(norm_base_obs)  # shape (n_stack * BASE_OBS_DIM,)
                logger.info(
                    f"Step 3: stacked_norm_obs | shape={stacked_norm_obs.shape}, "
                    f"min={stacked_norm_obs.min():.3f}, max={stacked_norm_obs.max():.3f}, mean={stacked_norm_obs.mean():.3f}"
                )

                # -------------------- 4) Model prediction ----------------------------------------
                obs_for_model = stacked_norm_obs.reshape(1, -1)
                with torch.no_grad():
                    act_int, _ = self.model.predict(obs_for_model, deterministic=True)
                a_int = int(np.asarray(act_int).squeeze())
                logger.info(f"Step 4: model predicted action_int={a_int}")

                # -------------------- 5) Map to action name --------------------------------------
                if _HAS_ENV_ACTIONS:
                    if 0 <= a_int < len(ENV_ACTIONS):
                        a_name = str(ENV_ACTIONS[a_int])
                    else:
                        a_name = str(ENV_ACTION_MAP.get(a_int, "NOTHING"))
                else:
                    a_name = ENV_ACTION_MAP.get(a_int, "NOTHING")
                logger.info(f"Step 5: mapped to action_name='{a_name}'")

                # -------------------- 6) Risk-aware buffer + tiny TTC safety shim ----------------
                # Estimate forward TTC using a forward cone consistent with training
                s = req.sensors or {}
                fwd_names = ["left_front", "front_left_front", "front", "front_right_front", "right_front"]
                fwd_vals = [float(s.get(n, 1.0) if s.get(n) is not None else 1.0) for n in fwd_names]
                fwd_min = min(fwd_vals) if fwd_vals else 1.0

                # Convert normalized distance back to px for TTC: d_px = fwd_min * MAX_SENSOR_PX
                MAX_SENSOR_PX = 1000.0
                FPS = 60.0
                vx_px_per_tick = max(float(req.velocity.get("x", 0.0)), 0.0)
                if vx_px_per_tick <= 1e-6:
                    ttc_fwd = float("inf")
                else:
                    d_px = fwd_min * MAX_SENSOR_PX
                    ttc_fwd = (d_px / vx_px_per_tick) / FPS

                # Minimal override: avoid "accelerate/idle into short TTC"
                if ttc_fwd < 1.0 and a_name in ("ACCELERATE", "NOTHING"):
                    logger.info(f"TTC shim: ttc_fwd={ttc_fwd:.2f} < 1.0 → override '{a_name}' → 'DECELERATE'")
                    a_name = "DECELERATE"

                # Dynamic action buffer: be reactive under risk, relax when very safe
                if ttc_fwd < 1.0 or fwd_min < 0.5:
                    buf = 2
                elif ttc_fwd > 3.0 and fwd_min > 0.7:
                    buf = max(buffer_len + 2, 6)
                else:
                    buf = buffer_len

                logger.info(
                    f"Step 6: TTC/Buffer | fwd_min={fwd_min:.2f}, ttc_fwd={ttc_fwd:.2f}, "
                    f"final_action='{a_name}', buffer={buf}"
                )

                return [a_name] * buf

            except Exception as e:
                logger.exception(f"[predict_actions] Exception: {e}")
                # Fail-safe: short buffer of no-ops to avoid erratic behavior
                return ["NOTHING"] * 2

# ----------------------------- API -------------------------------
app = FastAPI(title="Race Car PPO API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
predictor: Optional[PPOPredictor] = None

@app.on_event("startup")
def _load_model():
    global predictor
    predictor = PPOPredictor(MODEL_PATH, VECNORM_PATH)
    logger.info("[Startup] Predictor ready.")

@app.get("/")
def root():
    return {"status": "ok", "message": "Race Car PPO endpoint running."}

@app.post("/predict", response_model=RaceCarPredictResponseDto)
def predict(req: RaceCarPredictRequestDto):
    if predictor is None:
        actions = ["NOTHING"] * ACTION_BUFFER_LEN
        logger.error("Predictor not initialized, returning 'NOTHING'")
        return RaceCarPredictResponseDto(actions=actions)
    
    actions = predictor.predict_actions(req, buffer_len=ACTION_BUFFER_LEN)
    logger.info(f"--> Responding with actions: {actions}\n") # --- LOGGING ADDED ---
    return RaceCarPredictResponseDto(actions=actions)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")