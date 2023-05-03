import argparse
import os
import time
from typing import Any


import common.logging_options as logging_options
from envs.predefined_envs import create_simple_move_env
from trainer.trainer import Trainer

logging_options.set_default()

def get_configs():
  configs = dict[str, Any] (
    ppo_params= dict(
      policy = 'MultiInputPolicy',
      n_steps = 1024,
      batch_size = 64,
      learning_rate= 1e-4,
      policy_kwargs= dict(
        net_arch = [64, 64]
      ),
    ),
    grid_params=dict(
      sight = 0,
      add_grid = False,
    ),
    objective_params = dict(
      bounty = 0,
      distance = 100,
      max_distance = 320,
      episode_max_len=60,
      rew_dc=1
    ),
    learn_params = dict(),
    actions_params = dict(
      can_shoot = False,
      can_dash = False,
    )
  )

  return configs


def main(total_steps: int):
  trainer = Trainer(export_wandb=False)
  configs = get_configs()
  project_name=f'study_{time.time_ns()//100000000}'
  trial_name = '1'
  record_path = os.path.join(trainer.get_trial_path(project_name, trial_name), 'replay')
  env = create_simple_move_env(configs, record_path=record_path)
  trainer.train(env, total_steps, configs, project_name, trial_name)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--total_steps', type=int, default=1000)
  args = parser.parse_args()

  main(**vars(args))