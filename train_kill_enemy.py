import argparse
import logging
import os
import time
from typing import Any, Optional


import common.logging_options as logging_options
from envs.predefined_envs import create_kill_enemy
from trainer.trainer import Trainer

logging_options.set_default()

def get_configs():
  configs = dict[str, Any] (
    ppo_params= dict(
      policy = 'MultiInputPolicy',
      n_steps = 1024,
      batch_size = 64,
      learning_rate= 2*1e-5,
      policy_kwargs= dict(
        # net_arch = [64, 64]
        net_arch = [128] * 2
      ),
    ),
    grid_params=dict(
      sight = 0,
      add_grid = False,
    ),
    objective_params = dict(
      enemy_type = 'slime',
      enemy_count = 2,
      min_distance=50,
      max_distance=120,
      bounty=5,
      episode_max_len=60*6
    ),
    learn_params = dict(),
    actions_params = dict(
      can_shoot = False,
      can_dash = True,
    )
  )

  return configs


def parse_load_from(load_from: str) -> tuple[str, str, Optional[str]]:
  load_from_split = load_from.split('/')[-3:]
  if len(load_from_split) == 2:
    project, name = load_from_split
    model = None
  elif len(load_from_split) == 3:
    project, name, model = load_from_split
  else:
    raise Exception("Can't evaluate with load_from: {load_from}")
  return project, name, model


def evaluate(load_from: str):
  project, name, model = parse_load_from(load_from)

  trainer = Trainer(export_wandb=True)
  if model:
    trainer.evaluate_model(create_kill_enemy, 15, project, name, model)
  else:
    logging.info('Evaluating all models')
    trainer.evaluate_all_models(create_kill_enemy, 5, project, name)


def main(total_steps: int, load_from: str, eval: bool):
  if eval:
    assert load_from is not None, 'Must specify load_from when evaluating'
    evaluate(load_from)
    return

  trainer = Trainer(export_wandb=True)
  configs = get_configs()
  project_name=f'study_{time.time_ns()//100000000}'
  trial_name = '1'
  record_path = os.path.join(trainer.get_trial_path(project_name, trial_name), 'replay')
  env = create_kill_enemy(configs, record_path=record_path)
  if load_from:
    load_project, load_trial, load_model = parse_load_from(load_from)
    assert load_model is not None, 'Must specify model when loading'
    trainer.fork_training(env, total_steps, configs, project_name, trial_name, load_project, load_trial, load_model)
  else:
    trainer.train(env, total_steps, configs, project_name, trial_name)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--total_steps', type=int, default=1000)
  parser.add_argument('--eval', type=bool, default=False)
  parser.add_argument('--load_from', type=str, default=None)
  args = parser.parse_args()

  main(**vars(args))