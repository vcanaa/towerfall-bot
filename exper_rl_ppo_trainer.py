import argparse
import logging
import os
import numpy as np

from envs import TowerfallBlankEnv, GridObservation, PlayerObservation, FollowTargetObjective
from common import Connection, GridView

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import   ts2xy, plot_results
from stable_baselines3.common.monitor import load_results, Monitor
from stable_baselines3.common.callbacks import BaseCallback

class NoLevelFormatter(logging.Formatter):
  def format(self, record):
    return record.getMessage()

logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(NoLevelFormatter())

_HOST = '127.0.0.1'
_PORT = 12024


class SaveOnBestTrainingRewardCallback(BaseCallback):
  """
  Callback for saving a model (the check is done every ``check_freq`` steps)
  based on the training reward (in practice, we recommend using ``EvalCallback``).

  :param check_freq:
  :param log_dir: Path to the folder where the model will be saved.
    It must contains the file created by the ``Monitor`` wrapper.
  :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
  """
  def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
    super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
    self.check_freq = check_freq
    self.log_dir = log_dir
    self.save_path = os.path.join(log_dir, "best_model")
    self.best_mean_reward = -np.inf

  def _init_callback(self) -> None:
    # Create folder if needed
    if self.save_path is not None:
      os.makedirs(self.save_path, exist_ok=True)

  def _on_step(self) -> bool:
    if self.n_calls % self.check_freq == 0:

      # Retrieve training reward
      x, y = ts2xy(load_results(self.log_dir), "timesteps")
      if len(x) > 0:
        # Mean training reward over the last 100 episodes
        mean_reward = np.mean(y[-100:])
        if self.verbose >= 1:
          print(f"Num timesteps: {self.num_timesteps}")
          print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

        # New best model, you could save the agent here
        if mean_reward > self.best_mean_reward:
          self.best_mean_reward = mean_reward
          # Example for saving best model
          if self.verbose >= 1:
            print(f"Saving new best model to {self.save_path}")
            assert self.model
            self.model.save(self.save_path)

    return True

log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

# TODO: make this more configurable
configs = {
  "ppo_params": {
    "policy": "MultiInputPolicy",
    "n_steps": 200,
    "batch_size": 200,
    "policy_kwargs": {
      "net_arch": [256, 256]
    },
  },
  "total_timesteps": 2000
}

def main(load_from=None, save_to=None):
  connection = Connection(_HOST, _PORT)

  grid_view = GridView(grid_factor=5)
  env = TowerfallBlankEnv(
    connection=connection,
    observations= [
      GridObservation(grid_view),
      PlayerObservation()
    ],
    objective=FollowTargetObjective(grid_view))
  check_env(env)

  if  load_from is not None and os.path.exists(load_from):
    logging.info(f'Loading model from {load_from}')
    model = PPO.load(load_from, env = env)
  else:
    model = PPO(
      env=env,
      verbose=1,
      **configs['ppo_params'],
      tensorboard_log="./tensorboard/ppo_test"
    )

  # best_rew_mean, rew_std = evaluate_policy(model, env=env, deterministic=False)

  logging.info('###############################################')
  logging.info(f'Starting to train for {configs["total_timesteps"]} timesteps...')

  callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

  # while True:
  model.learn(
    total_timesteps=configs['total_timesteps'],
    progress_bar=True,
    callback=callback)
  model.logger.dump()

    # rew_mean, rew_std = evaluate_policy(model, env=env, deterministic = False)

    # if rew_mean > best_rew_mean:
    #   if save_to is not None:
    #     os.makedirs(os.path.dirname(save_to), exist_ok=True)
    #     logging.info(f'Saving model to {save_to}')
    #     model.save(save_to)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--load-from', type=str, default=None)
  parser.add_argument('--save-to', type=str, default='rl_models/test.model')
  args = parser.parse_args()

  main(**vars(args))