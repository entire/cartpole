from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    def _on_step(self) -> bool:
        # Example: Log custom metrics
        self.logger.record("custom/reward_sum", self.locals["rewards"].sum())
        return True
