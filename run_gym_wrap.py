import os

from common import Connection, EnvWrap

from stable_baselines3.common.env_checker import check_env

# from stable_baselines3.dqn import DQN
from stable_baselines3.ppo import PPO

connection = Connection('127.0.0.1', 9000)
env = EnvWrap(grid_factor=4, sight=50, connection=connection)
# check_env(env)

model_path = 'models/test.model'

if os.path.exists(model_path):
  model = PPO.load(model_path)
else:
  model = PPO(
    env=env,
    batch_size=128,
    policy="MultiInputPolicy",
    policy_kwargs={'net_arch': [256, 256]},
    verbose=1)

while True:
  model.learn(total_timesteps=1000)
  model.save('test')
