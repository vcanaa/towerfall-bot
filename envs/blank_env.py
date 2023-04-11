import logging

from gym import spaces

from common import Connection

from .base_env import TowerfallEnv
from .actions import TowerfallActions
from .observations import TowerfallObservation
from .objectives import TowerfallObjective

from typing import Tuple, Optional


class TowerfallBlankEnv(TowerfallEnv):
  '''A blank environment that can be customized with the addition of observations and an objective.'''
  def __init__(self,
      connection: Connection,
      observations: list[TowerfallObservation],
      objective: TowerfallObjective,
      actions: Optional[TowerfallActions]=None):
    super(TowerfallBlankEnv, self).__init__(connection, actions)
    print('Initializing TowerfallBlankEnv')
    obs_space = {}
    self.components = list(observations)
    self.components.append(objective)
    self.objective = objective
    self.objective.env = self
    for obs in self.components:
      print('Extending obs space {type(obs)}')
      obs.extend_obs_space(obs_space)
    self.observation_space = spaces.Dict(obs_space)
    logging.info(str(self.observation_space))

  def _is_reset_valid(self) -> bool:
    assert self.me
    return self.objective.is_reset_valid(self.state_scenario, self.me, self.entities)

  def _send_reset(self):
    reset_inst = self.objective.get_reset_instruction()
    self.connection.write_reset(**reset_inst)

  def _post_reset(self) -> dict:
    obs_dict = {}
    for obs in self.components:
      obs.post_reset(self.state_scenario, self.me, self.entities, obs_dict)
    # logging.info(f"reset: {str(obs_dict)}")
    return obs_dict

  def _post_step(self) -> Tuple[object, float, bool, object]:
    obs_dict = {}
    for obs in self.components:
      obs.post_step(self.me, self.entities, self.command, obs_dict)
    # logging.info(f"step: {str(obs_dict)}")
    return obs_dict, self.objective.rew, self.objective.done, {}
