import sys

sys.path.insert(0, '.')

import logging
from typing import Any

from common import GridView, logging_options
from envs import (FollowCloseTargetCurriculum, GridObservation,
                  PlayerObservation, TowerfallBlankEnv)
from towerfall import Towerfall

logging_options.set_default()

def create_env(configs) -> TowerfallBlankEnv:
  grid_view = GridView(grid_factor=5)
  objective = FollowCloseTargetCurriculum(grid_view, **configs['objective_params'])
  towerfall = Towerfall(config=dict(
    mode='sandbox',
    level='2',
    agents=[dict(type='remote', team='blue', archer='green')]
  ), verbose=1)
  env = TowerfallBlankEnv(
    towerfall=towerfall,
    observations= [
      GridObservation(grid_view, **configs['grid_params']),
      PlayerObservation()
    ],
    objective=objective,
    verbose=1)
  # check_env(env)
  return env


env = create_env(dict[str, Any] (
  grid_params=dict(
    sight = 60,
  ),
  objective_params = dict(
    bounty = 1,
    distance = 20,
    max_distance = 40,
    episode_max_len=60,
    rew_dc=1
  ),
  learn_params = dict()
))

logging.info('Evaluating')
n_episodes = 500
env.reset()
for ep in range(n_episodes):
  obs, rew, done, info = env.step(env.action_space.sample())
  if done:
    env.reset()

