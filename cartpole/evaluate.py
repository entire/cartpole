import logging
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("evaluate.log"),
        logging.StreamHandler()
    ]
)

def evaluate_agent():
    logging.info("Starting evaluation process...")

    # Load the trained model
    model_path = "models/ppo_cartpole"
    logging.info(f"Loading model from '{model_path}'.")
    model = PPO.load(model_path)

    # Create environment with render_mode
    env = make_vec_env("CartPole-v1", n_envs=1)
    env = env.envs[0]  # Access the single environment
    logging.info("Environment created successfully.")

    # Evaluate the policy
    n_eval_episodes = 10
    logging.info(f"Evaluating policy over {n_eval_episodes} episodes.")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, render=True)
    logging.info(f"Evaluation complete. Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Close the environment
    env.close()
    logging.info("Environment closed.")

if __name__ == "__main__":
    evaluate_agent()
