import sys

from envs import TowerfallBlankEnv, GridObservation, PlayerObservation, FollowTargetObjective
sys.path.insert(0, '../..')
from common import Connection, GridView

import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


_HOST = '127.0.0.1'
_PORT = 12024

connection = Connection(_HOST, _PORT)
grid_view = GridView(grid_factor=5)
env = TowerfallBlankEnv(
  connection=connection,
  observations= [
    GridObservation(grid_view),
    PlayerObservation()
  ],
  objective=FollowTargetObjective(grid_view))

print('Starting to play')
obs = env.reset()
while True:
    print('Obs:', obs)
    # action, _states = model.predict(obs)
    action = env.action_space.sample()
    obs, rewards, dones, info = env.step(action)
    print(obs, rewards, dones, info)
    env.render()
