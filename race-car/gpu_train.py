# train_sb3_gpu.py  — PPO + parallel envs + CUDA (RTX 5090) + VecNormalize + Eval/Checkpoint
# Drop-in replacement for your current script.
# Requires: stable-baselines3 >= 2.0, torch >= 2.0

import os
from typing import Optional

# ---- Prevent CPU oversubscription in env worker processes ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# If you use pygame, run headless during training:
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from src.game.racecar_env import RaceCarEnv  # <-- your env

# ----------------------------- Config -----------------------------

MODEL_SAVE_DIR = "models_sb3"
MODEL_NAME = "race_car_ppo_cuda_parallel"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
VECNORM_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_vecnormalize.pkl")
LOG_DIR = "logs_sb3"

# Hardware profile (Threadripper 7960X + RTX 5090)
NUM_CPU = 24         # 32 if your env is very light AND truly instance-safe
N_STEPS = 4096       # per-env rollout length
SEED = 42
TOTAL_TIMESTEPS = 200_000_000

# PPO core hyperparams (GPU-friendly)
NET_ARCH = [256, 256, 256, 256]        # a bit wider; GPU handles this easily
LEARNING_RATE = 3e-4
N_EPOCHS = 3
GAMMA = 0.995
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.02

# VecNormalize settings
CLIP_OBS = 10.0
# Keep reward normalization during training; turn it off for eval/inference

# Evaluation / checkpoint cadence (measured in env steps across all envs)
UPDATES_PER_EVAL = 4     # eval every ~UPDATES_PER_EVAL rollout updates
UPDATES_PER_CKPT = 20    # checkpoint every ~UPDATES_PER_CKPT updates

# ----------------------------- Utilities -----------------------------

def choose_batch_size(global_batch: int) -> int:
    """
    Return a batch_size that exactly divides global_batch and is large enough
    to keep the GPU fed. We try 2, 3, then 4 minibatches per update.
    """
    for k in (2, 3, 4):
        if global_batch % k == 0:
            return global_batch // k
    # Fallback: find a large divisor (>= 4096 when possible)
    for bs in range(16384, 2047, -512):
        if global_batch % bs == 0:
            return bs
    # Last resort: just use the whole batch (1 minibatch)
    return global_batch

def make_envs(n_envs: int, monitor_dir: Optional[str]):
    """
    Build vectorized envs. SubprocVecEnv -> true parallelism across processes.
    Your RaceCarEnv must be instance-safe across processes (no module-level globals).
    """
    return make_vec_env(
        env_id=RaceCarEnv,
        n_envs=n_envs,
        seed=SEED,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=monitor_dir,
    )

def enable_cuda_fastpaths():
    # Enable fast matmul/conv paths on Ada/Blackwell-class GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")  # PyTorch 2.x
        except Exception:
            pass

# ----------------------------- Training -----------------------------

def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Optional: sanity check a single env instance
    try:
        check_env(RaceCarEnv(), warn=True)
    except Exception as e:
        print(f"[check_env] Warning: {e}")

    # CUDA fast paths
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available; falling back to CPU. (Expected a 5090 here.)")
        device = "cpu"
    else:
        device = "cuda"
        enable_cuda_fastpaths()

    # Compute global batch and a GPU-friendly batch_size that divides it exactly
    global_batch = NUM_CPU * N_STEPS           # e.g., 24 * 1024 = 24576
    batch_size = choose_batch_size(global_batch)  # e.g., 12288 (2 minibatches)
    eval_freq = global_batch * UPDATES_PER_EVAL
    ckpt_freq = global_batch * UPDATES_PER_CKPT

    print(f"[Config] n_envs={NUM_CPU}, n_steps={N_STEPS}, global_batch={global_batch}, "
          f"batch_size={batch_size}, n_epochs={N_EPOCHS}, device={device}")
    print(f"[Cadence] eval_freq={eval_freq} steps, ckpt_freq={ckpt_freq} steps")

    # -------- Training envs --------
    train_env = make_envs(NUM_CPU, monitor_dir=LOG_DIR)
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=CLIP_OBS,
        gamma=GAMMA,
    )

    policy_kwargs = dict(
        net_arch=NET_ARCH,
        activation_fn=torch.nn.ReLU,
        ortho_init=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,          # per-env
        batch_size=batch_size,    # must divide n_steps * n_envs
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        target_kl=TARGET_KL,
        verbose=1,
        tensorboard_log=LOG_DIR,
        seed=SEED,
        device=device,            # <-- GPU
    )

    # -------- Eval env (shares obs normalization stats) --------
    eval_env = make_envs(1, monitor_dir=None)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,   # do not normalize rewards at eval/inference
        clip_obs=CLIP_OBS,
        gamma=GAMMA,
    )
    # Share observation normalization with training env
    eval_env.obs_rms = train_env.obs_rms
    eval_env.training = False

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_SAVE_DIR,
        log_path=MODEL_SAVE_DIR,
        eval_freq=eval_freq,       # in env steps (aggregated across all envs)
        n_eval_episodes=8,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=ckpt_freq,       # in env steps
        save_path=MODEL_SAVE_DIR,
        name_prefix=MODEL_NAME,
        save_replay_buffer=False,
        save_vecnormalize=False,   # we save VecNormalize explicitly after learn()
        verbose=1,
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    print(f"--- Starting PPO Training on {NUM_CPU} envs (CUDA) ---")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True, callback=callbacks)
    print("--- Training Complete ---")

    # Save model + VecNormalize stats (needed for proper scaling at inference)
    model.save(MODEL_SAVE_PATH)
    train_env.save(VECNORM_PATH)

    # Cleanup
    train_env.close()
    eval_env.close()

    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"VecNormalize stats saved to {VECNORM_PATH}")

# ----------------------------- Inference Helper -----------------------------

def load_for_inference(model_path: str = MODEL_SAVE_PATH, vecnorm_path: str = VECNORM_PATH):
    """
    Example of loading the trained model and VecNormalize stats for rollout/serving.
    """
    venv = make_envs(1, monitor_dir=None)
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(model_path, device=device)
    return model, venv

# ----------------------------- Entrypoint -----------------------------

if __name__ == "__main__":
    train()
