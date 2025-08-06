# api_sb3.py — Serve PPO policy for server-side verification (final)
# Matches training wrappers: VecNormalize -> VecFrameStack(3)
# Adds robust VecNormalize unwrapping, TTC+rear safety shim, short dynamic buffer.

import os
# Keep server CPU usage predictable (avoid oversubscription)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pickle
import logging
from collections import deque
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from stable_baselines3 import PPO

# Optional: SB3 VecNormalize for loading obs stats
try:
    from stable_baselines3.common.vec_env import VecNormalize
    _HAS_VECNORM = True
except Exception:
    _HAS_VECNORM = False

from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto

# ----------------------------- Logging -----------------------------
logger = logging.getLogger("api_predictor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler("predictor_log.txt")
    ch = logging.StreamHandler()
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

# ----------------------------- Config ------------------------------
MODEL_SAVE_DIR = "models_sb3"
MODEL_BASENAME = "race_car_ppo_cuda_parallel"  # must match training
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_BASENAME)
VECNORM_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_BASENAME}_vecnormalize.pkl")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))  # default 9052 for hosted evals

# Must match training wrappers
N_STACK = 3                 # VecFrameStack(n_stack=3) in train
BASE_OBS_DIM = 18           # 16 sensors + vx + vy
STACKED_OBS_DIM = N_STACK * BASE_OBS_DIM

# Keep buffers short; we’ll shorten further under risk
ACTION_BUFFER_LEN = int(os.environ.get("ACTION_BUFFER_LEN", "2"))

# Safety shim thresholds (tuneable without retraining)
TTC_ACC_BLOCK_SEC = 1.2     # block ACCELERATE if forward TTC < 1.2s
REAR_CLOSE_THRESH = 0.35    # rear beam < 0.35 (normalized) → “on our bumper”
FWD_CLEAR_FOR_NUDGE = 0.65  # if front is this clear, allow gentle accel under rear pressure

# Action mapping from the env if available
try:
    from src.game.racecar_env import ACTIONS as ENV_ACTIONS, ACTION_MAP as ENV_ACTION_MAP
    _HAS_ENV_ACTIONS = True
except Exception:
    _HAS_ENV_ACTIONS = False
    ENV_ACTIONS = ["NOTHING", "ACCELERATE", "DECELERATE", "STEER_LEFT", "STEER_RIGHT"]
    ENV_ACTION_MAP = {i: a for i, a in enumerate(ENV_ACTIONS)}

# Sensor order must match training
SENSORS_ORDER = [
    "front", "right_front", "right_side", "right_back", "back", "left_back",
    "left_side", "left_front", "left_side_front", "front_left_front",
    "front_right_front", "right_side_front", "right_side_back",
    "back_right_back", "back_left_back", "left_side_back",
]
FWD_IDXS  = (7, 9, 0, 10, 1)  # left_front, front_left_front, front, front_right_front, right_front
BACK_IDXS = (3, 4, 5)         # right_back, back, left_back

# ------------------------- Obs Normalizer --------------------------
class ObsNormalizer:
    """
    Normalize a single 18-D frame using VecNormalize stats saved at training time.
    Robustly unwraps to find obs_rms even if wrappers changed.
    """
    def __init__(self, vecnorm_path: str, clip_obs: float = 10.0):
        self.enabled = False
        self.mean = None
        self.var = None
        self.clip_obs = clip_obs

        if not _HAS_VECNORM or not os.path.exists(vecnorm_path):
            logger.warning(f"[ObsNorm] VecNormalize not available or stats file missing: {vecnorm_path}")
            return

        try:
            with open(vecnorm_path, "rb") as f:
                obj = pickle.load(f)

            def _extract_obs_rms(x):
                seen = set()
                cur = x
                last_clip = clip_obs
                while cur is not None and id(cur) not in seen:
                    seen.add(id(cur))
                    if hasattr(cur, "obs_rms") and hasattr(cur.obs_rms, "mean") and hasattr(cur.obs_rms, "var"):
                        return cur.obs_rms, float(getattr(cur, "clip_obs", last_clip))
                    last_clip = float(getattr(cur, "clip_obs", last_clip))
                    cur = getattr(cur, "venv", None)
                return None, clip_obs

            obs_rms, clip = _extract_obs_rms(obj)
            if obs_rms is None:
                logger.warning(f"[ObsNorm] Could not find obs_rms in {vecnorm_path}; running UN-normalized.")
                return

            self.mean = np.array(obs_rms.mean, dtype=np.float32).reshape(1, BASE_OBS_DIM)
            self.var  = np.array(obs_rms.var,  dtype=np.float32).reshape(1, BASE_OBS_DIM)
            self.clip_obs = clip
            self.enabled = True
            logger.info(f"[ObsNorm] Loaded stats: clip_obs={self.clip_obs}, mean.shape={self.mean.shape}, var.shape={self.var.shape}")
        except Exception as e:
            logger.exception(f"[ObsNorm] Failed to load stats from {vecnorm_path}: {e}")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return obs.astype(np.float32, copy=False)
        assert obs.shape == (BASE_OBS_DIM,), f"Expected ({BASE_OBS_DIM},), got {obs.shape}"
        obs2d = obs.reshape(1, BASE_OBS_DIM)
        eps = 1e-8
        norm = (obs2d - self.mean) / np.sqrt(self.var + eps)
        clipped = np.clip(norm, -self.clip_obs, self.clip_obs)
        return clipped.astype(np.float32, copy=False).reshape(-1)

# --------------------------- Predictor -----------------------------
class PPOPredictor:
    def __init__(self, model_path: str, vecnorm_path: Optional[str]):
        # Avoid CPU oversubscription in torch on servers
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[Model] Loading PPO on device={device}")
        path = model_path if os.path.isfile(model_path) else model_path + ".zip"
        self.model = PPO.load(path, device=device)

        # Sanity: observation size expected by model vs what we will build
        try:
            model_dim = int(np.prod(self.model.observation_space.shape))
            expected = STACKED_OBS_DIM
            if model_dim != expected:
                logger.warning(f"[Sanity] Model expects obs_dim={model_dim}, server builds {expected}. "
                               f"Check N_STACK={N_STACK} & BASE_OBS_DIM={BASE_OBS_DIM}.")
        except Exception as e:
            logger.warning(f"[Sanity] Could not verify model obs shape: {e}")

        self.obs_norm = ObsNormalizer(vecnorm_path, clip_obs=10.0)
        self._hist = deque(maxlen=N_STACK)

    def _stack(self, normalized_base_obs: np.ndarray) -> np.ndarray:
        if len(self._hist) == 0:
            for _ in range(N_STACK):
                self._hist.append(normalized_base_obs.copy())
        else:
            self._hist.append(normalized_base_obs)
        stacked = np.concatenate(list(self._hist), dtype=np.float32)
        assert stacked.shape == (STACKED_OBS_DIM,), f"Bad stacked shape: {stacked.shape}"
        return stacked

    def _build_base_obs(self, req: RaceCarPredictRequestDto) -> np.ndarray:
        # Training-identical scaling: sensors in [0,1], vx/20, vy/2
        s = req.sensors or {}
        sensors = []
        for name in SENSORS_ORDER:
            raw = s.get(name, None)
            v = 1.0 if raw is None else float(raw) / 1000.0
            sensors.append(max(0.0, min(1.0, v)))
        vx_raw = float((req.velocity or {}).get("x", 0.0))
        vy_raw = float((req.velocity or {}).get("y", 0.0))
        vx = vx_raw / 20.0
        vy = vy_raw / 2.0
        base = np.array(sensors + [vx, vy], dtype=np.float32)

        # ================================ FIX START =================================
        # This step is CRITICAL to match the training environment (`racecar_env.py`).
        # The environment clips the observation vector to [-1.0, 1.0] *before*
        # it is passed to the VecNormalize wrapper. Omitting this step causes
        # out-of-distribution inputs, especially for velocity, leading to poor
        # model performance.
        base = np.clip(base, -1.0, 1.0)
        # ================================= FIX END ==================================

        assert base.shape == (BASE_OBS_DIM,), f"Bad base obs shape: {base.shape}"
        return base

    def _safety_shim(self, req: RaceCarPredictRequestDto, a_name: str, sensors01: np.ndarray) -> tuple[str, float, float, float]:
        """
        Emergency guardrails that operate on *pre-normalized* sensor distances in [0,1].
        sensors01: np.ndarray of shape (16,), values in [0,1] (1.0 = far, 0.0 = very close).
        Returns: (final_action, ttc_fwd, fwd_min01, back_min01)
        """
        # Forward TTC (use front cone min distance in [0,1])
        fwd_min01 = float(np.min(sensors01[list(FWD_IDXS)]))   # 0..1
        d_fwd_px = fwd_min01 * 1000.0

        # Ego speed in px/tick (raw, not scaled)
        vx_raw = float((req.velocity or {}).get("x", 0.0))
        vx_pos = max(vx_raw, 1e-6)
        ttc_fwd = (d_fwd_px / vx_pos) / 60.0  # seconds

        # Rear proximity (narrow cone) in [0,1]
        back_min01 = float(np.min(sensors01[list(BACK_IDXS)]))

        # ---------- Overrides ----------
        # 1) If TTC is short, never accelerate AND don't idle into it: brake.
        if ttc_fwd < TTC_ACC_BLOCK_SEC and a_name in ("ACCELERATE", "NOTHING"):
            a_name = "DECELERATE"

        # 2) Rear pressure: if someone is on our bumper and front is fairly clear,
        #    avoid hard braking (prefer NOTHING or gentle ACCELERATE).
        if a_name == "DECELERATE" and back_min01 < REAR_CLOSE_THRESH:
            if fwd_min01 >= FWD_CLEAR_FOR_NUDGE:
                a_name = "ACCELERATE"  # open the gap a bit
            elif fwd_min01 >= 0.55:
                a_name = "NOTHING"     # coast instead of brake

        return a_name, ttc_fwd, fwd_min01, back_min01


    def predict_actions(self, req: RaceCarPredictRequestDto, buffer_len: int = ACTION_BUFFER_LEN) -> List[str]:
        try:
            # 1) Build base obs (18,) with training-identical scaling AND clipping
            base = self._build_base_obs(req)

            # 2) Normalize single frame, then stack to (54,) for the policy
            base_norm = self.obs_norm(base)                      # z-scored frame (18,)
            stacked_norm = self._stack(base_norm)                # (54,)
            obs_for_model = stacked_norm.reshape(1, -1)

            # Optional: clipping diagnostics on normalized features
            if self.obs_norm.enabled:
                clip_val = getattr(self.obs_norm, "clip_obs", 10.0)
                clip_frac = float(np.mean((base_norm <= -clip_val) | (base_norm >= clip_val)))
                if clip_frac > 0.05:
                    logger.warning(f"[ObsNorm] {clip_frac*100:.1f}% dims at clip (+/-{clip_val}). Stats may not match.")

            # 3) Policy prediction
            with torch.no_grad():
                act_int, _ = self.model.predict(obs_for_model, deterministic=True)
            a_int = int(np.asarray(act_int).squeeze())
            if _HAS_ENV_ACTIONS and 0 <= a_int < len(ENV_ACTIONS):
                a_model = str(ENV_ACTIONS[a_int])
            else:
                a_model = str(ENV_ACTION_MAP.get(a_int, "NOTHING"))

            # 4) SAFETY SHIM MUST USE PRE-NORM SENSORS IN [0,1]
            sensors01 = base[:16] # This uses the pre-normalized (but now correctly clipped) obs
            a_safe, ttc_fwd, fwd_min01, back_min01 = self._safety_shim(req, a_model, sensors01)

            # 5) Dynamic buffer: react FAST when risky or shim changed the action
            risky = (ttc_fwd < TTC_ACC_BLOCK_SEC) or (fwd_min01 < 0.45)
            dyn_buf = 1 if risky or (a_safe != a_model) else buffer_len
            # keep a sane upper bound anyway
            dyn_buf = min(dyn_buf, 3)

            logger.info(
                f"[Predict] a_model={a_model} -> a_final={a_safe} | "
                f"ttc_fwd={ttc_fwd:.2f}s, fwd_min01={fwd_min01:.2f}, back_min01={back_min01:.2f}, buf={dyn_buf}"
            )

            return [a_safe] * dyn_buf

        except Exception as e:
            logger.exception(f"[predict_actions] error: {e}")
            return ["NOTHING"] * max(1, min(buffer_len, 2))

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

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict", response_model=RaceCarPredictResponseDto)
def predict(req: RaceCarPredictRequestDto):
    if predictor is None:
        actions = ["NOTHING"] * ACTION_BUFFER_LEN
        logger.error("Predictor not initialized, returning 'NOTHING'")
        return RaceCarPredictResponseDto(actions=actions)
    actions = predictor.predict_actions(req, buffer_len=ACTION_BUFFER_LEN)
    return RaceCarPredictResponseDto(actions=actions)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")