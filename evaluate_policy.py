import argparse
import logging
import os
import numpy as np
import time
import json

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import load_results, Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import   ts2xy, plot_results
from envs import TowerfallBlankEnv, GridObservation, PlayerObservation, FollowTargetObjective, FollowCloseTargetCurriculum


from common import Connection, GridView

class NoLevelFormatter(logging.Formatter):
  def format(self, record):
    return record.getMessage()

logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(NoLevelFormatter())

_HOST = '127.0.0.1'
_PORT = 12024


def input_blocking_callback(locals, globals):
      import ipdb; ipdb.set_trace()


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
  evaluate_policy(model, 
                  env=env, 
                  n_eval_episodes=50, 
                  render=False, 
                  deterministic=False,
                  callback=input_blocking_callback)
  # logging.info('Deterministic=True')
  # evaluate_policy(model, env=env, n_eval_episodes=30, render=False, deterministic=True)
  # logging.info(f'Finished evaluation for {last_model}')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--load-from', type=str, required=True)
  args = parser.parse_args()
  evaluate(args.load_from)
