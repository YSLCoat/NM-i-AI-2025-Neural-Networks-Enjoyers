import os
from stable_baselines3 import DQN
from src.game.racecar_env import RaceCarEnv

MODEL_SAVE_DIR = "models_sb3"
MODEL_NAME = "race_car_ppo_cuda_parallel_47185920_steps.zip"
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
NUM_EPISODES = 10 

def evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run train_sb3.py to train and save a model first.")
        return

    env = RaceCarEnv(render_mode="human")
    model = DQN.load(MODEL_PATH, env=env)
    print(f"--- Evaluating model: {MODEL_PATH} for {NUM_EPISODES} episodes ---")

    for episode in range(NUM_EPISODES):
        obs, info = env.reset(seed=10)
        
        done = False
        total_reward = 0.0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            total_reward += reward

        final_distance = info.get('distance', 0)
        print(f"Episode {episode + 1}: Final Distance = {final_distance:.2f}, Total Reward = {total_reward:.2f}")

    env.close()

if __name__ == '__main__':
    evaluate()