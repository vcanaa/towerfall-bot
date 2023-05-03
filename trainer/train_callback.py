import os
import logging

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import   ts2xy
from stable_baselines3.common.monitor import load_results

class TrainCallback(BaseCallback):
  '''
  Callback for saving a models

  :param log_dir: Path to the folder where the model will be saved.
    It must contains the file created by the ``Monitor`` wrapper.
  :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
  '''
  def __init__(self, check_freq: int, n_episodes: int, log_dir: str, verbose: int = 1):
    super(TrainCallback, self).__init__(verbose)
    self.check_freq = check_freq
    self.n_episodes = n_episodes
    self.log_dir = log_dir
    self.save_path = os.path.join(log_dir, 'models')
    self.best_mean_reward: float = -np.inf

  def _init_callback(self) -> None:
    # Create folder if needed
    if self.save_path is not None:
      os.makedirs(self.save_path, exist_ok=True)

  def _on_step(self) -> bool:
    if self.n_calls % self.check_freq != 0:
      return True

    # Retrieve training reward
    x, y = ts2xy(load_results(self.log_dir), 'timesteps')
    if len(x) > 0:
      mean_reward = np.mean(y[-self.n_episodes:])
      if mean_reward > self.best_mean_reward:
        self.best_mean_reward = mean_reward # type: ignore
        assert self.model
        model_path = os.path.join(self.save_path, str(self.n_calls))
        logging.info(f'Saving model to {model_path}')
        self.model.save(model_path)

    return True