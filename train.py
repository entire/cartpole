import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback
import torch

# YYYY-MM-DD-HH-MM-SS 
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/{timestamp}.log"),
        logging.StreamHandler()
    ]
)

class TensorBoardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        # Log scalar values
        self.logger.record('train/reward', self.training_env.buf_rews[0])
        return True

def train_agent():
    logging.info("Starting training process...")
    
    # Set device to MPS if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Create environment
    env = make_vec_env("CartPole-v1", n_envs=1)
    logging.info("Environment created successfully.")

    # Set up model and logging with GPU support
    model = PPO("MlpPolicy", 
                env, 
                verbose=1, 
                tensorboard_log=f"./tensorboard_logs/{timestamp}/",
                device=device)
    logging.info("Model initialized with PPO algorithm.")

    # Create custom TensorBoard callback
    tensorboard_callback = TensorBoardCallback()

    # Train the model with callback
    total_timesteps = 10000
    logging.info(f"Training started for {total_timesteps} timesteps.")
    model.learn(total_timesteps=total_timesteps, callback=tensorboard_callback)

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    model.save(f"models/ppo_cartpole_{timestamp}")
    logging.info(f"Training complete. Model saved to 'models/ppo_cartpole_{timestamp}'.")

if __name__ == "__main__":
    train_agent()
