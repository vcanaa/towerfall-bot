import os
import logging

from common import Connection
from envs import EnvMovementExpert

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.ppo import PPO

class NoLevelFormatter(logging.Formatter):
  def format(self, record):
    return record.getMessage()


logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(NoLevelFormatter())

connection = Connection('127.0.0.1', 12024)
env = EnvMovementExpert(grid_factor=10, connection=connection)
check_env(env)

model_path = 'rl_models/test.model'

# if os.path.exists(model_path):
#   model = PPO.load(model_path, env=env)
# else:
model = PPO(
  env=env,
  batch_size=128,
  policy="MultiInputPolicy",
  policy_kwargs={'net_arch': [256, 256]},
  verbose=1)

while True:
  model.learn(total_timesteps=100)
  for k, v in model.logger.name_to_value:
    logging.info('%s: %s', k, v)
  model.save(model_path)
