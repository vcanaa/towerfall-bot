import logging

from common import Connection, GridView
from envs import TowerfallBlankEnv, GridObservation, PlayerObservation, FollowTargetObjective, FollowCloseTargetCurriculum

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.ppo import PPO

class NoLevelFormatter(logging.Formatter):
  def format(self, record):
    return record.getMessage()


logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(NoLevelFormatter())

connection = Connection('127.0.0.1', 12024)

grid_view = GridView(grid_factor=5)
env = TowerfallBlankEnv(
  connection=connection,
  observations= [
    GridObservation(grid_view),
    PlayerObservation()
  ],
  objective=FollowCloseTargetCurriculum(grid_view))
check_env(env)

model = PPO(
    env=env,
    batch_size=128,
    policy="MultiInputPolicy",
    policy_kwargs={'net_arch': [256, 256]},
    verbose=1)

while True:
  model.learn(total_timesteps=2000)
