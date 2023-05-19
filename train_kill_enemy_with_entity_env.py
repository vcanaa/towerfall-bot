import logging
from enn_trainer import TrainConfig, State, init_train_state, train
import hyperstate
from common import logging_options

from entity_envs.entity_env import TowerfallEntityEnvImpl
from towerfall import Towerfall


logging_options.set_default()


@hyperstate.stateful_command(TrainConfig, State, init_train_state)
def main(state_manager: hyperstate.StateManager) -> None:
  try:
    train(state_manager=state_manager, env=TowerfallEntityEnvImpl)
  finally:
    logging.info('Closing all Towerfall processes')
    Towerfall.close_all()


if __name__ == "__main__":
  main()