import argparse
import logging
import os
import numpy as np
import time
import json

import wandb
from wandb.wandb_run import Run
from wandb.integration.sb3 import WandbCallback

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

from typing import Any, Optional, Union, Callable, Dict, List, Tuple

class NoLevelFormatter(logging.Formatter):
  def format(self, record):
    return record.getMessage()

logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(NoLevelFormatter())

_HOST = '127.0.0.1'
_PORT = 12024

class GroupCallback(BaseCallback):
  def __init__(self, callbacks: list[BaseCallback], verbose: int = 0):
    super(GroupCallback, self).__init__(verbose)
    self.callbacks = callbacks

  def _init_callback(self) -> None:
    assert self.model
    for callback in self.callbacks:
      callback.init_callback(self.model)

  def _on_step(self) -> bool:
    result = True
    for callback in self.callbacks:
      r = callback.on_step()
      if not r:
        result = False
    return result

  def _on_training_end(self) -> None:
    for callback in self.callbacks:
      callback.on_training_end()


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


def init_wandb(configs, project_name: str, name: str) -> Run:
  logging.debug('Starting wandb.')
  run = wandb.init(
      project=project_name,
      name=name,
      tags=[],
      config=configs,
      sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
      # monitor_gym=True,  # auto-upload the videos of agents playing the game
      save_code=True,  # optional
  )
  assert type(run) is Run, f'Expected Run, got {type(run)}'
  # if debug:
      # wandb.watch(model, log='all', log_freq=1)
  logging.info(f'Initialized wandb run {run.name}, id:{run.id}')
  logging.debug('Finished loading wandb.')
  return run


def create_env(configs, connection: Connection) -> TowerfallBlankEnv:
  grid_view = GridView(grid_factor=5)
  objective = FollowCloseTargetCurriculum(grid_view, **configs['objective_params'])
  print('CREATIN ENV')
  env = TowerfallBlankEnv(
    connection=connection,
    observations= [
      GridObservation(grid_view, **configs['grid_params']),
      PlayerObservation()
    ],
    objective=objective)
  print('ENV CREATED')
  check_env(env)
  return env


def run_experiment(trial: Trial) -> float:
  print('STARTING EXPERIMENT')
  # n_steps = trial.suggest_int('ppo_params/n_steps', 200, 400, step=100)
  n_steps = trial.suggest_int('ppo_params/n_steps', 1024, 2048, step=1024)
  max_distance = 40
  configs = dict[str, Any] (
    ppo_params= dict(
      policy = 'MultiInputPolicy',
      n_steps = n_steps,
      batch_size = trial.suggest_int('ppo_params/batch_size', 64, 128, step=64),
      # learning_rate= trial.suggest_float('ppo_params/learning_rate', 1e-4, 1e-2, log=True),
      learning_rate= 1e-5,
      policy_kwargs= dict(
        # net_arch = [trial.suggest_int('ppo_params/layer_size', 128, 256, step=64)] * trial.suggest_int('ppo_params/depth', 1, 3)
        net_arch = [256, 256]
      ),
    ),
    grid_params=dict(
      # sight= trial.suggest_int('grid_params/sight', 30, 60, step=10),
      sight = trial.suggest_int('grid_params/sight', max_distance, 2*max_distance, step=max_distance),
    ),
    objective_params = dict(
      # bounty=trial.suggest_float('objective_params/bounty', 4, 8, step=2),
      bounty = 1,
      distance = 20,
      max_distance = max_distance,
      episode_max_len=60,
      # rew_dc=trial.suggest_int('objective_params/rew_dc', 1, 2)
      rew_dc=1
    ),
    learn_params = dict()
  )

  connection = Connection(_HOST, _PORT)
  try:
    env = create_env(configs, connection)

    log_dir = f'tmp/{trial.study.study_name}/{trial.number}'
    os.makedirs(log_dir, exist_ok=True)

    monitored_env = Monitor(env, os.path.join(log_dir))

    print('CREATING MODEL')
    model = PPO(
      env=monitored_env,
      verbose=2,
      tensorboard_log=os.path.join(log_dir, 'tensorboard'),
      **configs['ppo_params'],
    )

    # configs['learn_params']['total_timesteps'] = max(n_steps * 10, objective.max_total_steps)
    configs['learn_params']['total_timesteps'] = 400000
    logging.info('###############################################')
    logging.info(f"Starting to train for {configs['learn_params']['total_timesteps']} timesteps...")

    logging.info(f'Creating wandb run for trial {trial.study.study_name}/{trial.number}')
    run: Run = init_wandb(configs, trial.study.study_name, f'trial_{trial.number}')

    assert isinstance(env.objective, FollowCloseTargetCurriculum)
    try:
      train_callback = TrainCallback(
        trial,
        configs['ppo_params']['n_steps'],
        env.objective.n_episodes,
        log_dir=log_dir)
      wandb_callback = WandbCallback(
          # gradient_save_freq=100,
          # model_save_freq=100,
          # model_save_path=f"{log_dir}/models",
          verbose=2
        )
      callback = GroupCallback([train_callback, wandb_callback])

      with open(os.path.join(log_dir, 'hparams.json'), 'w') as file:
        file.write(json.dumps(configs, indent=2))

      print(f'PARAMS: {configs}')
      print('LEARNING')
      model = model.learn(
        progress_bar=True,
        callback=callback,
        **configs['learn_params'])
      model.logger.dump()
      print('FINISHED')
    finally:
      run.finish()
      # pass
    return train_callback.best_mean_reward
  finally:
    connection.close()


def evaluate(load_from: str):
  logging.info(f'Loading experiment from {load_from}')
  with open(os.path.join(load_from, 'hparams.json'), 'r') as file:
    configs = json.load(file)

  connection = Connection(_HOST, _PORT)
  env = Monitor(create_env(configs, connection))

  model_names = os.listdir(os.path.join(load_from, 'models'))
  last_model = None
  last_step = -1
  for name in model_names:
    if name == 'model.zip':
      last_model = 'model.zip'
      break
    step = int(name.replace('.zip', ''))
    if step > last_step:
      last_step = step
      last_model = name

  last_model = os.path.join(load_from, 'models', last_model)
  # last_model = os.path.abspath(last_model)

  logging.info(f'Loading model from {last_model}')
  model = PPO.load(last_model)

  logging.info(f'Running evaluation for {last_model}')
  logging.info('Deterministic=False')
  evaluate_policy(model, env=env, n_eval_episodes=15, render=False, deterministic=False)
  logging.info('Deterministic=True')
  evaluate_policy(model, env=env, n_eval_episodes=15, render=False, deterministic=True)
  logging.info(f'Finished evaluation for {last_model}')


def main(load_from=None):
  if load_from:
    evaluate(load_from)
    return

  assert False, 'Not implemented'
  wandb.login()

  # optuna_db = f'tmp/optuna/{time.time_ns()//100000000}'
  optuna_db = f'sqlite:///optuna/experiments.db'
  # os.makedirs(log_dir, exist_ok=True)

  study = optuna.create_study(
    storage=optuna_db,
    # study_name=f'study_{time.time_ns()//100000000}',
    study_name='study_16812733147',
    direction='maximize',
    # pruner=optuna.pruners.SuccessiveHalvingPruner()
    load_if_exists=True,
    )
  study.optimize(run_experiment, n_trials=4, show_progress_bar=True)
  # study.save(os.path.join(log_dir, 'study.pkl'))

  print(f'Study name: {study.study_name}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--load-from', type=str, default=None)
  args = parser.parse_args()

  main(**vars(args))