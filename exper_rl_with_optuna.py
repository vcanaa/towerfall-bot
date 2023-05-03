import argparse
import json
import logging
import os
import time
from typing import Any

from optuna.trial import Trial
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import optuna
from envs import (FollowCloseTargetCurriculum)
from envs.predefined_envs import create_simple_move_env
from trainer import TrainCallback


class NoLevelFormatter(logging.Formatter):
  '''Class documentation here'''
  def format(self, record):
    return record.getMessage()

logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(NoLevelFormatter())


export_wandb = True
if export_wandb:
  from wandb.integration.sb3 import WandbCallback
  from wandb.wandb_run import Run

  import wandb

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


def run_experiment(trial: Trial) -> float:
  logging.info('STARTING EXPERIMENT')
  # n_steps = trial.suggest_int('ppo_params/n_steps', 200, 400, step=100)
  configs = get_configs(trial)

  env = create_simple_move_env(configs)

  log_dir = f'tmp/{trial.study.study_name}/{trial.number}'
  os.makedirs(log_dir, exist_ok=True)

  monitored_env = Monitor(env, os.path.join(log_dir))

  logging.info('CREATING MODEL')
  model = PPO(
    env=monitored_env,
    verbose=2,
    tensorboard_log=os.path.join(log_dir, 'tensorboard'),
    **configs['ppo_params'],
  )

  # configs['learn_params']['total_timesteps'] = max(n_steps * 10, objective.max_total_steps)
  # configs['learn_params']['total_timesteps'] = 1000
  configs['learn_params']['total_timesteps'] = 50000
  # configs['learn_params']['total_timesteps'] = 400000
  logging.info('###############################################')
  logging.info(f"Starting to train for {configs['learn_params']['total_timesteps']} timesteps...")

  logging.info(f'Creating wandb run for trial {trial.study.study_name}/{trial.number}')
  # run = None
  if export_wandb:
    run: Run = init_wandb(configs, trial.study.study_name, f'trial_{trial.number}')

  assert isinstance(env.objective, FollowCloseTargetCurriculum)
  try:
    callbacks = []
    train_callback = TrainCallback(
      trial,
      configs['ppo_params']['n_steps'],
      env.objective.n_episodes,
      log_dir=log_dir)
    callbacks.append(train_callback)
    if export_wandb:
      wandb_callback = WandbCallback(
          # gradient_save_freq=100,
          # model_save_freq=100,
          # model_save_path=f"{log_dir}/models",
          verbose=2
        )
      callbacks.append(wandb_callback)

    with open(os.path.join(log_dir, 'hparams.json'), 'w') as file:
      file.write(json.dumps(configs, indent=2))

    logging.info(f'PARAMS: {configs}')
    logging.info('LEARNING')
    model = model.learn(
      progress_bar=True,
      callback=callbacks,
      **configs['learn_params'])
    model.logger.dump()
    logging.info('FINISHED')
  finally:
    if run:
      logging.info('FINISHING W&B RUN')
      run.finish()
  return train_callback.best_mean_reward


def evaluate(load_from: str):
  logging.info(f'Loading experiment from {load_from}')
  with open(os.path.join(load_from, 'hparams.json'), 'r') as file:
    configs = json.load(file)

  env = Monitor(create_simple_move_env(configs))

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


def main(load_from=None, n_trials=1):
  if load_from:
    evaluate(load_from)
    return

  if export_wandb:
    wandb.login()

  # optuna_db = f'tmp/optuna/{time.time_ns()//100000000}'
  optuna_db = f'sqlite:///optuna/experiments.db'
  # os.makedirs(log_dir, exist_ok=True)

  study = optuna.create_study(
    storage=optuna_db,
    study_name=f'study_{time.time_ns()//100000000}',
    # study_name='study_16812733147',
    direction='maximize',
    # pruner=optuna.pruners.SuccessiveHalvingPruner()
    load_if_exists=True,
    )
  study.optimize(run_experiment, n_trials=n_trials, show_progress_bar=True)
  # study.save(os.path.join(log_dir, 'study.pkl'))

  logging.info(f'Study name: {study.study_name}')

def get_configs(trial: Trial):
  # n_steps = trial.suggest_int('ppo_params/n_steps', 1024, 2048, step=1024)
  n_steps = 1024
  max_distance = 320
  configs = dict[str, Any] (
    ppo_params= dict(
      policy = 'MultiInputPolicy',
      n_steps = n_steps,
      # batch_size = trial.suggest_int('ppo_params/batch_size', 64, 128, step=64),
      batch_size = 64,
      # learning_rate= trial.suggest_float('ppo_params/learning_rate', 1e-4, 1e-2, log=True),
      learning_rate= 1e-4,
      policy_kwargs= dict(
        # net_arch = [trial.suggest_int('ppo_params/layer_size', 128, 256, step=64)] * trial.suggest_int('ppo_params/depth', 1, 3)
        net_arch = [64, 64]
      ),
    ),
    grid_params=dict(
      # sight= trial.suggest_int('grid_params/sight', 30, 60, step=10),
      # sight = trial.suggest_int('grid_params/sight', max_distance, 2*max_distance, step=max_distance),
      sight = 0,
      add_grid = False,
    ),
    objective_params = dict(
      # bounty=trial.suggest_float('objective_params/bounty', 4, 8, step=2),
      bounty = 0,
      distance = 100,
      max_distance = max_distance,
      episode_max_len=60,
      # rew_dc=trial.suggest_int('objective_params/rew_dc', 1, 2)
      rew_dc=1
    ),
    learn_params = dict()
  )

  return configs

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--load-from', type=str, default=None)
  parser.add_argument('--n_trials', type=int, default=1)
  args = parser.parse_args()

  main(**vars(args))