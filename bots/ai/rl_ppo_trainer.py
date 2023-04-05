import argparse
import logging
import os
import sys

sys.path.insert(0, '../..')
from envs import TowerfallBlankEnv, GridObservation, PlayerObservation, FollowTargetObjective
from common import Connection, GridView

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

class NoLevelFormatter(logging.Formatter):
  def format(self, record):
    return record.getMessage()

logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(NoLevelFormatter())

_HOST = '127.0.0.1'
_PORT = 12024


# TODO: make this more configurable
configs = {
    "ppo_params": {
        "n_steps": 256,
        "batch_size": 64,
    },
    "policy_kwargs": {
        "net_arch": [256, 256]
    },
    "total_timesteps": 1024
}

def main(load_from=None, save_to=None):
    connection = Connection(_HOST, _PORT)

    grid_view = GridView(grid_factor=5)
    env = TowerfallBlankEnv(
      connection=connection,
      observations= [
        GridObservation(grid_view),
        PlayerObservation()
      ],
      objective=FollowTargetObjective(grid_view))
    check_env(env)

    if  load_from is not None and os.path.exists(load_from):
        logging.info(f'Loading model from {load_from}')
        model = PPO.load(load_from)
    else:
        model = PPO(
            env=env,
            batch_size=configs['ppo_params']['batch_size'],
            n_steps=configs['ppo_params']['n_steps'],
            policy="MultiInputPolicy",
            policy_kwargs=configs['policy_kwargs'],
            verbose=1
        )

    logging.info('###############################################')
    logging.info(f'Starting to train for {configs["total_timesteps"]} timesteps...')


    model.learn(
       total_timesteps=configs['total_timesteps'],
       progress_bar=True)

    if save_to is not None:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        logging.info(f'Saving model to {save_to}')
        model.save(save_to)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-from', type=str, default=None)
    parser.add_argument('--save-to', type=str, default='rl_models/test.model')
    args = parser.parse_args()

    main(**vars(args))
