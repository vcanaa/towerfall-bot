import sys

from envs import TowerFallMovementEnv
sys.path.insert(0, '../..')
from common import Connection

import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


_HOST = '127.0.0.1'
_PORT = 12024

connection = Connection(_HOST, _PORT)


env = TowerFallMovementEnv(grid_factor=2, sight=50, connection=connection)
# env = make_vec_env("CartPole-v1", n_envs=4)

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=25000)

# model.save("ppo_cartpole")
# del model # remove to demonstrate saving and loading
# model = PPO.load("ppo_cartpole")

print('Starting to play')
obs = env.reset()
while True:
    print('Obs:', obs)
    # action, _states = model.predict(obs)
    action = env.action_space.sample()
    obs, rewards, dones, info = env.step(action)
    print(obs, rewards, dones, info)
    env.render()
