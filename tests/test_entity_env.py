import sys
sys.path.insert(0, 'C:/Program Files (x86)/Steam/steamapps/common/TowerFall/aimod')

import random
from entity_gym.env import GlobalCategoricalAction, GlobalCategoricalActionSpace
from entity_envs.entity_env import TowerfallEntityEnvImpl
from envs.connection_provider import TowerfallProcessProvider


import logging

from typing import Any

class NoLevelFormatter(logging.Formatter):
  def format(self, record):
    return record.getMessage()

logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(NoLevelFormatter())

def create_env() -> TowerfallEntityEnvImpl:
  towerfall_provider = TowerfallProcessProvider('test-entity-env')
  towerfall = towerfall_provider.get_process(
    fastrun=True,
    config=dict(
      mode='sandbox',
      level='2',
      fps=90,
    agents=[dict(type='remote', team='blue', archer='green')]
  ), verbose=1)
  env = TowerfallEntityEnvImpl(
    verbose=0)
  return env


env = create_env()

logging.info('Evaluating')
n_episodes = 500
env.reset()
for ep in range(n_episodes):
  actions = {}
  for action_name, action in env.action_space().items():
    assert isinstance(action, GlobalCategoricalActionSpace)
    idx = random.randint(0, len(action.index_to_label) - 1)
    actions[action_name] = GlobalCategoricalAction(idx, action.index_to_label[idx])
  obs = env.act(actions)
  if obs.done:
    env.reset()

