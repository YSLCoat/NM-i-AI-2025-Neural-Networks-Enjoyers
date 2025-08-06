# api_sb3.py — Serve PPO policy for server-side verification (hysteresis + overtake + rear safety)
import os
import pickle
import logging
from datetime import datetime
from collections import deque
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from stable_baselines3 import PPO

# Optional: load VecNormalize stats for obs standardization
try:
    from stable_baselines3.common.vec_env import VecNormalize
    _HAS_VECNORM = True
except Exception:
    _HAS_VECNORM = False

from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto

# ----------------------------- Logging -----------------------------
logger = logging.getLogger("api_predictor")
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

log_file_path = "predictor_log.txt"
file_handler = logging.FileHandler(log_file_path)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# ----------------------------- Config -----------------------------
MODEL_SAVE_DIR = "models_sb3"
MODEL_BASENAME = os.environ.get("MODEL_BASENAME", "race_car_ppo_cuda_parallel")
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_BASENAME)
VECNORM_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_BASENAME}_vecnormalize.pkl")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

# Match training (VecFrameStack(n_stack=3), 16 sensors + vx + vy = 18)
N_STACK = int(os.environ.get("N_STACK", "3"))
BASE_OBS_DIM = 18
STACKED_OBS_DIM = N_STACK * BASE_OBS_DIM

# How many actions to return per call (the server consumes a small buffer)
DEFAULT_ACTION_BUFFER_LEN = int(os.environ.get("ACTION_BUFFER_LEN", "2"))

# --- Safety shim knobs (env vars let you tune without redeploying) ---
TTC_BLOCK_SEC    = float(os.environ.get("TTC_BLOCK_SEC",    "1.35"))  # engage "no-accelerate" latch below this
TTC_RELEASE_SEC  = float(os.environ.get("TTC_RELEASE_SEC",  "1.75"))  # release latch only after this (with sustain)
RELEASE_SUSTAIN  = int(os.environ.get("RELEASE_SUSTAIN",    "3"))     # consecutive safe frames required to release
COAST_MIN_SEC    = float(os.environ.get("COAST_MIN_SEC",    "1.35"))  # prefer NOTHING if TTC in [coast_min, coast_max]
COAST_MAX_SEC    = float(os.environ.get("COAST_MAX_SEC",    "1.75"))

REAR_CLOSE_TH    = float(os.environ.get("REAR_CLOSE_TH",    "0.35"))  # normalized [0..1], lower is closer
REAR_SIDE_TH     = float(os.environ.get("REAR_SIDE_TH",     "0.45"))
FWD_CLEAR_TH     = float(os.environ.get("FWD_CLEAR_TH",     "0.60"))  # forward "clear" min sensor
OVERTAKE_TTC_SEC = float(os.environ.get("OVERTAKE_TTC_SEC", "1.60"))  # start preferring lateral when under this
OVERTAKE_MARGIN  = float(os.environ.get("OVERTAKE_MARGIN",  "0.15"))  # side must be this much clearer than forward
STEER_HOLD_STEPS = int(os.environ.get("STEER_HOLD_STEPS",   "2"))     # commit to a lane change briefly
DANGER_BUF_LEN   = int(os.environ.get("DANGER_BUF_LEN",     "1"))     # when risky, shorten buffer to react faster

# ----------------------------- Actions -----------------------------
# Try to import from the env (best). If not available on server, FALL BACK to *training order*:
# ['ACCELERATE', 'DECELERATE', 'STEER_LEFT', 'STEER_RIGHT', 'NOTHING']
try:
    from src.game.racecar_env import ACTIONS as ENV_ACTIONS, ACTION_MAP as ENV_ACTION_MAP
    _HAS_ENV_ACTIONS = True
except Exception:
    _HAS_ENV_ACTIONS = False
    ENV_ACTIONS = ['ACCELERATE', 'DECELERATE', 'STEER_LEFT', 'STEER_RIGHT', 'NOTHING']  # << correct fallback
    ENV_ACTION_MAP = {i: a for i, a in enumerate(ENV_ACTIONS)}
# NOTE: The training env orders actions as above (see racecar_env.py) :contentReference[oaicite:8]{index=8}

# ----------------------------- Sensors -----------------------------
# Must match training sensor order (normalized to [0,1] by /1000 and clipped)
SENSORS_ORDER = [
    "front", "right_front", "right_side", "right_back", "back", "left_back",
    "left_side", "left_front", "left_side_front", "front_left_front",
    "front_right_front", "right_side_front", "right_side_back",
    "back_right_back", "back_left_back", "left_side_back",
]
# Cones used by the shim (indices into the 16-dim sensor head)
FWD_CONE        = (7, 9, 0, 10, 1)         # left_front, front_left_front, front, front_right_front, right_front
LEFT_CONE       = (8, 7, 9)                # left_side_front block
RIGHT_CONE      = (10, 1, 11)              # right-side-forward block
BACK_CONE_NAR   = (3, 4, 5)                # right_back, back, left_back
REAR_LEFT_CONE  = (15, 14, 5)              # left_side_back, back_left_back, left_back
REAR_RIGHT_CONE = (12, 13, 3)              # right_side_back, back_right_back, right_back

# ----------------------------- ObsNormalizer -----------------------------
class ObsNormalizer:
    """
    Normalizes a single unstacked 18-dim observation (like VecNormalize).
    We load obs_rms.{mean,var} from your VecNormalize pickle if available.
    """
    def __init__(self, vecnorm_path: str, clip_obs: float = 10.0):
        self.enabled = False
        self.mean = None
        self.var = None
        self.clip_obs = clip_obs
        if _HAS_VECNORM and os.path.exists(vecnorm_path):
            try:
                with open(vecnorm_path, "rb") as f:
                    vn = pickle.load(f)
                self.mean = np.array(vn.obs_rms.mean, dtype=np.float32).reshape(1, BASE_OBS_DIM)
                self.var  = np.array(vn.obs_rms.var,  dtype=np.float32).reshape(1, BASE_OBS_DIM)
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
        assert obs.shape == (BASE_OBS_DIM,), f"Expected ({BASE_OBS_DIM},), got {obs.shape}"
        x = obs.reshape(1, BASE_OBS_DIM)
        eps = 1e-8
        norm = (x - self.mean) / np.sqrt(self.var + eps)
        clipped = np.clip(norm, -self.clip_obs, self.clip_obs)
        return clipped.ravel()

# ----------------------------- Predictor -----------------------------
class PPOPredictor:
    def __init__(self, model_path: str, vecnorm_path: Optional[str]):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[Model] Loading PPO on device={device}")
        path = model_path if os.path.isfile(model_path) else model_path + ".zip"
        self.model = PPO.load(path, device=device)

        self.obs_norm = ObsNormalizer(vecnorm_path, clip_obs=10.0)
        self._hist = deque(maxlen=N_STACK)

        # safety shim state (hysteresis & short-term commitments)
        self._accel_block_latch = False
        self._release_count = 0
        self._hold_action_name: Optional[str] = None
        self._hold_steps = 0
        self._last_steer_dir = 0  # -1 left, +1 right, 0 none

    # ---------- helpers ----------
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
        s = req.sensors or {}
        sensors = []
        for name in SENSORS_ORDER:
            raw = s.get(name, None)
            v = 1.0 if raw is None else float(raw) / 1000.0  # match env scaling
            sensors.append(max(0.0, min(1.0, v)))            # hard clip [0,1]
        vx_raw = float((req.velocity or {}).get("x", 0.0))
        vy_raw = float((req.velocity or {}).get("y", 0.0))
        vx = vx_raw / 20.0  # matches env _get_obs
        vy = vy_raw / 2.0
        base = np.array(sensors + [vx, vy], dtype=np.float32)
        assert base.shape == (BASE_OBS_DIM,), f"Bad base obs shape: {base.shape}"
        return base

    def _cones_and_ttc(self, sensors_01: np.ndarray, vx_raw_px_tick: float) -> Dict[str, float]:
        fwd_min   = float(np.min(sensors_01[list(FWD_CONE)]))
        left_min  = float(np.min(sensors_01[list(LEFT_CONE)]))
        right_min = float(np.min(sensors_01[list(RIGHT_CONE)]))
        back_min  = float(np.min(sensors_01[list(BACK_CONE_NAR)]))
        rear_left_min  = float(np.min(sensors_01[list(REAR_LEFT_CONE)]))
        rear_right_min = float(np.min(sensors_01[list(REAR_RIGHT_CONE)]))

        d_fwd_px  = fwd_min * 1000.0
        vx_pos    = max(vx_raw_px_tick, 1e-6)
        ttc_fwd   = (d_fwd_px / vx_pos) / 60.0

        # simple, speed-agnostic rear heuristic based on normalized distance
        rear_pressure = float(np.clip((0.6 - back_min) / 0.6, 0.0, 1.0))  # 0 when far, 1 when very close

        return dict(
            fwd_min=fwd_min, left_min=left_min, right_min=right_min,
            back_min=back_min, rear_left_min=rear_left_min, rear_right_min=rear_right_min,
            ttc_fwd=ttc_fwd, rear_pressure=rear_pressure
        )

    def _hold_if_any(self, a_name: str, metrics: Dict[str, float]) -> Optional[str]:
        """Keep a committed steer for a couple of steps, unless a new emergency appears."""
        if self._hold_steps <= 0 or self._hold_action_name is None:
            return None
        # break hold if forward TTC has become critically small:
        if metrics["ttc_fwd"] < 0.9:
            self._hold_action_name = None
            self._hold_steps = 0
            return None
        self._hold_steps -= 1
        return self._hold_action_name

    def _choose_with_shim(self, policy_action: str, sensors_01: np.ndarray, vx_raw_px_tick: float
                          ) -> Tuple[str, int, Dict[str, float]]:
        """
        Apply hysteresis, rear-awareness, and overtake nudges.
        Returns (final_action_name, buffer_len, metrics).
        """
        m = self._cones_and_ttc(sensors_01, vx_raw_px_tick)

        # 1) honor any active hold (steer commitment) unless emergency
        held = self._hold_if_any(policy_action, m)
        if held is not None:
            return held, DEFAULT_ACTION_BUFFER_LEN, m  # normal buffer

        # 2) hysteresis on accelerate (stops thrashing near thresholds)
        if self._accel_block_latch:
            if m["ttc_fwd"] > TTC_RELEASE_SEC:
                self._release_count += 1
                if self._release_count >= RELEASE_SUSTAIN:
                    self._accel_block_latch = False
                    self._release_count = 0
            else:
                self._release_count = 0
        else:
            if m["ttc_fwd"] < TTC_BLOCK_SEC:
                self._accel_block_latch = True
                self._release_count = 0

        # 3) lateral nudge (overtake preference) when forward is tight
        final_action = policy_action
        buffer_len = DEFAULT_ACTION_BUFFER_LEN

        if m["ttc_fwd"] < OVERTAKE_TTC_SEC:
            left_better  = (m["left_min"]  - m["fwd_min"])  > OVERTAKE_MARGIN
            right_better = (m["right_min"] - m["fwd_min"])  > OVERTAKE_MARGIN

            # avoid merging into an occupied rear side
            left_occupied  = m["rear_left_min"]  < REAR_SIDE_TH
            right_occupied = m["rear_right_min"] < REAR_SIDE_TH

            if left_better and not left_occupied:
                final_action = "STEER_LEFT"
                self._hold_action_name = final_action
                self._hold_steps = STEER_HOLD_STEPS
                self._last_steer_dir = -1
                return final_action, 2, m  # small buffer to commit
            if right_better and not right_occupied:
                final_action = "STEER_RIGHT"
                self._hold_action_name = final_action
                self._hold_steps = STEER_HOLD_STEPS
                self._last_steer_dir = +1
                return final_action, 2, m

        # 4) coast band: stabilize headway
        if COAST_MIN_SEC <= m["ttc_fwd"] <= COAST_MAX_SEC:
            # prefer NOTHING over small accel/decel chatter
            final_action = "NOTHING"
            buffer_len = DEFAULT_ACTION_BUFFER_LEN
        else:
            # 5) “No-accelerate” latch
            if self._accel_block_latch and final_action == "ACCELERATE":
                final_action = "DECELERATE" if m["fwd_min"] < FWD_CLEAR_TH else "NOTHING"
                buffer_len = DANGER_BUF_LEN

            # 6) rear-pressure aware braking
            if final_action == "DECELERATE" and (m["back_min"] < REAR_CLOSE_TH) and (m["fwd_min"] >= FWD_CLEAR_TH):
                final_action = "NOTHING"
                buffer_len = DANGER_BUF_LEN

        # 7) shorten buffer in danger (react faster)
        if (m["ttc_fwd"] < TTC_BLOCK_SEC) or (m["fwd_min"] < 0.45):
            buffer_len = min(buffer_len, DANGER_BUF_LEN)

        return final_action, buffer_len, m

    # ---------- main ----------
    def predict_actions(self, req: RaceCarPredictRequestDto, buffer_len: int = DEFAULT_ACTION_BUFFER_LEN) -> List[str]:
        try:
            # 1) Build base obs & normalize, then stack normalized frames
            base = self._build_base_obs(req)
            base_norm = self.obs_norm(base)                 # (18,)
            stacked_norm = self._stack(base_norm)           # (54,)
            obs_for_model = stacked_norm.reshape(1, -1)     # (1, 54)

            # 2) Policy prediction
            with torch.no_grad():
                act_int, _ = self.model.predict(obs_for_model, deterministic=True)
            a_int = int(np.asarray(act_int).squeeze())
            if 0 <= a_int < len(ENV_ACTIONS):
                policy_action = str(ENV_ACTIONS[a_int])
            else:
                policy_action = str(ENV_ACTION_MAP.get(a_int, "NOTHING"))

            # 3) Safety shim + buffer logic
            sensors_01 = base[:16]
            vx_raw = float((req.velocity or {}).get("x", 0.0))  # px/tick
            final_action, dyn_buf, m = self._choose_with_shim(policy_action, sensors_01, vx_raw)

            # 4) Logging (compact but enough to diagnose)
            logger.info(
                "dist=%.1f ticks=%d pol=%s final=%s buf=%d | "
                "ttc=%.2f fwd=%.2f L=%.2f R=%.2f back=%.2f rL=%.2f rR=%.2f latch=%s rel=%d",
                float(getattr(req, "distance", 0.0)),
                int(getattr(req, "elapsed_ticks", 0)),
                policy_action, final_action, dyn_buf,
                m["ttc_fwd"], m["fwd_min"], m["left_min"], m["right_min"],
                m["back_min"], m["rear_left_min"], m["rear_right_min"],
                str(self._accel_block_latch), self._release_count
            )

            return [final_action] * dyn_buf

        except Exception as e:
            logger.exception(f"[predict_actions] error: {e}")
            # safe hard-fallback
            return ["NOTHING"] * max(2, buffer_len // 2)

# ----------------------------- API -----------------------------
app = FastAPI(title="Race Car PPO API", version="1.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
predictor: Optional[PPOPredictor] = None

@app.on_event("startup")
def _load_model():
    global predictor
    predictor = PPOPredictor(MODEL_PATH, VECNORM_PATH)
    logger.info("[Startup] Predictor ready with hysteresis & overtake shim.")

@app.get("/")
def root():
    return {"status": "ok", "message": "Race Car PPO endpoint running."}

@app.post("/predict", response_model=RaceCarPredictResponseDto)
def predict(req: RaceCarPredictRequestDto):
    if predictor is None:
        actions = ["NOTHING"] * DEFAULT_ACTION_BUFFER_LEN
        logger.error("Predictor not initialized, returning 'NOTHING'")
        return RaceCarPredictResponseDto(actions=actions)
    actions = predictor.predict_actions(req, buffer_len=DEFAULT_ACTION_BUFFER_LEN)
    return RaceCarPredictResponseDto(actions=actions)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level=os.environ.get("UVICORN_LOG_LEVEL", "info"))
