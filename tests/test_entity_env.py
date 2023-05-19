import sys

sys.path.insert(0, '.')

import logging
import random

from entity_gym.env import (GlobalCategoricalAction,
                            GlobalCategoricalActionSpace)

from common import logging_options
from entity_envs.entity_env import TowerfallEntityEnvImpl

logging_options.set_default()

def create_env() -> TowerfallEntityEnvImpl:
  env = TowerfallEntityEnvImpl(verbose=0)
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

