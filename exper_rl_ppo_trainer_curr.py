import argparse
import logging
import os
import numpy as np
import time
import json

from envs import TowerfallBlankEnv, GridObservation, PlayerObservation, FollowTargetObjective, FollowCloseTargetCurriculum
from common import Connection, GridView

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import   ts2xy, plot_results
from stable_baselines3.common.monitor import load_results, Monitor
from stable_baselines3.common.callbacks import BaseCallback

from typing import Any

class NoLevelFormatter(logging.Formatter):
  def format(self, record):
    return record.getMessage()

logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(NoLevelFormatter())

_HOST = '127.0.0.1'
_PORT = 12024


class SaveOnBestTrainingRewardCallback(BaseCallback):
  '''
  Callback for saving a model (the check is done every ``check_freq`` steps)
  based on the training reward (in practice, we recommend using ``EvalCallback``).

  :param check_freq:
  :param log_dir: Path to the folder where the model will be saved.
    It must contains the file created by the ``Monitor`` wrapper.
  :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
  '''
  def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
    super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
    self.check_freq = check_freq
    self.log_dir = log_dir
    self.save_path = os.path.join(log_dir, 'models')
    self.best_mean_reward = -np.inf

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
      # Mean training reward over the last 1 episodes
      mean_reward = np.mean(y[-1:])
      if self.verbose >= 1:
        print(f'Num timesteps: {self.num_timesteps}')
        print(f'Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}')

      # # New best model, you could save the agent here
      # if mean_reward > self.best_mean_reward:
      #   self.best_mean_reward = mean_reward
      #   # Example for saving best model
      #   if self.verbose >= 1:
      #     print(f'Saving new best model to {self.save_path}')
      #     assert self.model
      #     self.model.save(self.save_path)

      assert self.model
      model_path = os.path.join(self.save_path, str(self.n_calls))
      print(f'Saving model to {model_path}')
      self.model.save(model_path)
    return True


def main(load_from=None, save_to=None):
  log_dir = f'tmp/{time.time_ns()//100000000}'
  os.makedirs(log_dir, exist_ok=True)

  configs = dict[str, Any] (
    ppo_params= dict(
      policy= 'MultiInputPolicy',
      policy_kwargs= dict(
        net_arch= [256, 256]
      ),
    ),
    grid_params=dict(
      sight=50
    ),
    objective_params = dict(
      bounty=5,
      episode_max_len=90,
      rew_dc=2
    ),
    learn_params = dict()
  )

  connection = Connection(_HOST, _PORT)

  grid_view = GridView(grid_factor=5)
  objective = FollowCloseTargetCurriculum(grid_view, **configs['objective_params'])
  env = TowerfallBlankEnv(
    connection=connection,
    observations= [
      GridObservation(grid_view, **configs['grid_params']),
      PlayerObservation()
    ],
    objective=objective)
  env = Monitor(env, os.path.join(log_dir))
  check_env(env)

  configs['ppo_params']['n_steps'] = objective.max_total_steps
  configs['ppo_params']['batch_size'] = objective.max_total_steps
  if load_from is not None and os.path.exists(load_from):
    logging.info(f'Loading model from {load_from}')
    model = PPO.load(load_from, env = env)
    model.n_steps = configs['ppo_params']['n_steps']
    model.batch_size = configs['ppo_params']['batch_size']
  else:
    model = PPO(
      env=env,
      verbose=1,
      tensorboard_log=os.path.join(log_dir, 'tensorboard'),
      **configs['ppo_params'],
    )

  best_rew_mean, rew_std = evaluate_policy(
    model,
    n_eval_episodes=len(objective.start_ends),
    env=env,
    deterministic=False)

  # configs['learn_params']['total_timesteps'] = objective.max_total_steps * 10
  # logging.info('###############################################')
  # logging.info(f"Starting to train for {configs['learn_params']['total_timesteps']} timesteps...")

  # callback = SaveOnBestTrainingRewardCallback(check_freq=configs['ppo_params']['n_steps'], log_dir=log_dir)

  # with open(os.path.join(log_dir, 'hparams.json'), 'w') as file:
  #   file.write(json.dumps(configs, indent=2))
  # # while True:
  # model.learn(
  #   progress_bar=True,
  #   callback=callback,
  #   **configs['learn_params'])
  # model.logger.dump()

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