import sys
sys.path.insert(0, 'C:/Program Files (x86)/Steam/steamapps/common/TowerFall/aimod')

import logging

from envs import TowerfallBlankEnv, FollowCloseTargetCurriculum, GridObservation, PlayerObservation, TowerfallProcessProvider

from common import GridView

from typing import Any

class NoLevelFormatter(logging.Formatter):
  def format(self, record):
    return record.getMessage()

logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(NoLevelFormatter())

def create_env(configs) -> TowerfallBlankEnv:
  grid_view = GridView(grid_factor=5)
  objective = FollowCloseTargetCurriculum(grid_view, **configs['objective_params'])
  towerfall_provider = TowerfallProcessProvider('default')
  towerfall = towerfall_provider.get_process(config=dict(
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
    actions=
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

