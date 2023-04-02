import os
import logging

from common import Connection
from envs import EnvMovement

from stable_baselines3.common.env_checker import check_env

from stable_baselines3.ppo import PPO

class NoLevelFormatter(logging.Formatter):
  def format(self, record):
    return record.getMessage()


logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(NoLevelFormatter())

connection = Connection('127.0.0.1', 12024)
env = EnvMovement(grid_factor=2, sight=50, connection=connection)
check_env(env)

model_path = 'rl_models/test.model'

# if os.path.exists(model_path):
#   model = PPO.load(model_path)
# else:
model = PPO(
  env=env,
  batch_size=128,
  policy="MultiInputPolicy",
  policy_kwargs={'net_arch': [256, 256]},
  verbose=1)

while True:
  model.learn(total_timesteps=1000)
  # model.save(model_path)
