import argparse
import logging
import os
import numpy as np
import time
import json

import optuna
from optuna.trial import Trial
from optuna.exceptions import TrialPruned

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

class TrainCallback(BaseCallback):
  '''
  Callback for saving a models

  :param log_dir: Path to the folder where the model will be saved.
    It must contains the file created by the ``Monitor`` wrapper.
  :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
  '''
  def __init__(self, trial: Trial, check_freq: int, n_episodes: int, log_dir: str, verbose: int = 1):
    super(TrainCallback, self).__init__(verbose)
    self.trial = trial
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
        print(f'Saving model to {model_path}')
        self.model.save(model_path)

      # self.trial.report(self.best_mean_reward, self.n_calls) # type: ignore
      # if self.trial.should_prune():
      #   raise optuna.exceptions.TrialPruned()

    return True


def run_experiment(trial: Trial) -> float:
  print('STARTING EXPERIMENT')
  n_steps = trial.suggest_int('ppo_params/n_steps', 200, 400, step=100)
  max_distance = 20
  configs = dict[str, Any] (
    ppo_params= dict(
      policy= 'MultiInputPolicy',
      n_steps= n_steps,
      batch_size= n_steps,
      policy_kwargs= dict(
        net_arch = [trial.suggest_int('ppo_params/layer_size', 128, 256, step=64)] * trial.suggest_int('ppo_params/depth', 1, 3)
      ),
    ),
    grid_params=dict(
      # sight= trial.suggest_int('grid_params/sight', 30, 60, step=10),
      sight = trial.suggest_int('grid_params/sight', max_distance, 2*max_distance, step=8),
    ),
    objective_params = dict(
      # bounty=trial.suggest_float('objective_params/bounty', 4, 8, step=2),
      bounty = 1,
      distance = 8,
      max_distance = max_distance,
      episode_max_len=30,
      # rew_dc=trial.suggest_int('objective_params/rew_dc', 1, 2)
      rew_dc=1
    ),
    learn_params = dict()
  )

  grid_view = GridView(grid_factor=5)
  objective = FollowCloseTargetCurriculum(grid_view, **configs['objective_params'])
  print('CREATIN ENV')
  connection = Connection(_HOST, _PORT)
  try:
    env = TowerfallBlankEnv(
      connection=connection,
      observations= [
        GridObservation(grid_view, **configs['grid_params']),
        PlayerObservation()
      ],
      objective=objective)
    print('ENV CREATED')

    log_dir = f'tmp/{time.time_ns()//100000000}'
    os.makedirs(log_dir, exist_ok=True)

    env = Monitor(env, os.path.join(log_dir))
    check_env(env)

    # configs['ppo_params']['n_steps'] = objective.max_total_steps
    # configs['ppo_params']['batch_size'] = objective.max_total_steps

    # if load_from is not None and os.path.exists(load_from):
    #   logging.info(f'Loading model from {load_from}')
    #   model = PPO.load(load_from, env = env)
    #   model.n_steps = configs['ppo_params']['n_steps']
    #   model.batch_size = configs['ppo_params']['batch_size']
    # else:
    print('CREATING MODEL')
    model = PPO(
      env=env,
      verbose=1,
      tensorboard_log=os.path.join(log_dir, 'tensorboard'),

      **configs['ppo_params'],
    )

    # best_rew_mean, rew_std = evaluate_policy(
    #   model,
    #   n_eval_episodes=len(objective.start_ends),
    #   env=env,
    #   deterministic=False)

    configs['learn_params']['total_timesteps'] = max(n_steps * 10, objective.max_total_steps)
    logging.info('###############################################')
    logging.info(f"Starting to train for {configs['learn_params']['total_timesteps']} timesteps...")

    callback = TrainCallback(
      trial,
      configs['ppo_params']['n_steps'],
      objective.n_episodes,
      log_dir=log_dir)

    with open(os.path.join(log_dir, 'hparams.json'), 'w') as file:
      file.write(json.dumps(configs, indent=2))
    # while True:
    print(f'PARAMS: {configs}')
    print('LEARNING')
    model = model.learn(
      progress_bar=True,
      callback=callback,
      **configs['learn_params'])
    model.logger.dump()
    print('FINISHED')
    return callback.best_mean_reward
  finally:
    connection.close()


def main(load_from=None, save_to=None):
  # optuna_db = f'tmp/optuna/{time.time_ns()//100000000}'
  optuna_db = f'sqlite:///optuna/experiments.db'
  # os.makedirs(log_dir, exist_ok=True)

  study = optuna.create_study(
    storage=optuna_db,
    study_name='no-name-d1bb728d-1d19-4b97-870a-430e1eaddca9',
    direction='maximize',
    # pruner=optuna.pruners.SuccessiveHalvingPruner()
    load_if_exists=True,
    )
  study.optimize(run_experiment, n_trials=100, show_progress_bar=True)
  # study.save(os.path.join(log_dir, 'study.pkl'))

  print(f'Study name: {study.study_name}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--load-from', type=str, default=None)
  parser.add_argument('--save-to', type=str, default='rl_models/test.model')
  args = parser.parse_args()

  main(**vars(args))