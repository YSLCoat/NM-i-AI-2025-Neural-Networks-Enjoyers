# evaluate_ppo.py

import os
from stable_baselines3 import PPO  # Import PPO instead of DQN
from src.game.racecar_env import RaceCarEnv

# --- Configuration ---
MODEL_SAVE_DIR = "models_sb3"
# --- IMPORTANT: Change this to the name of the PPO model you want to test ---
# It could be "race_car_ppo.zip" or "race_car_ppo_lstm.zip"
MODEL_NAME = "race_car_ppo.zip" 
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
NUM_EPISODES = 10 # How many episodes to run for evaluation

def evaluate():
    """
    Loads a trained PPO model and runs it in the RaceCarEnv to evaluate its performance.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run the PPO training script to train and save a model first.")
        return

    # 1. Create the environment with render_mode="human" to see the game
    env = RaceCarEnv(render_mode="human")
    
    # 2. Load the trained PPO model
    model = PPO.load(MODEL_PATH, env=env)

    print(f"--- Evaluating PPO model: {MODEL_PATH} for {NUM_EPISODES} episodes ---")

    for episode in range(NUM_EPISODES):
        # Reset the environment to get the initial state
        obs, info = env.reset()
        
        done = False
        total_reward = 0.0
        
        while not done:
            # 3. Use the model to predict the best action
            action, _states = model.predict(obs, deterministic=True)
            
            # 4. Take the action in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            total_reward += reward

        final_distance = info.get('distance', 0)
        print(f"Episode {episode + 1}: Final Distance = {final_distance:.2f}, Total Reward = {total_reward:.2f}")

    # 5. Clean up the environment
    env.close()

if __name__ == '__main__':
    evaluate()