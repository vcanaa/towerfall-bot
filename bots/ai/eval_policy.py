import argparse
import logging
import os
import sys

sys.path.insert(0, '../..')
from envs import TowerfallMovementEnv
from common import Connection

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy


class NoLevelFormatter(logging.Formatter):
  def format(self, record):
    return record.getMessage()

logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(NoLevelFormatter())

_HOST = '127.0.0.1'
_PORT = 12024


def main(load_from, eval_policy=False):

    if not os.path.exists(load_from):
        raise Exception(f'Could not find model at {load_from}')

    logging.info(f'Loading model from {load_from}')
    model = PPO.load(load_from)

    # TODO: why is get_env() returning None?
    # model.get_env()
    env = TowerfallMovementEnv(grid_factor=2, sight=50, connection=Connection(_HOST, _PORT))
    check_env(env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    if eval_policy:
        logging.info('Evaluating policy...')
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        logging.info(f'mean_reward={mean_reward:.2f} +/- {std_reward:.2f}')

    # Enjoy trained agent
    vec_env = env
    # vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        print(obs, action)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-from', type=str, default='rl_models/test.model')
    parser.add_argument('--eval-policy', action='store_true', default=False)
    args = parser.parse_args()

    main(args.load_from, args.eval_policy)
