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
    obs_space = {}
    self.components = list(observations)
    self.components.append(objective)
    self.objective = objective
    self.objective.env = self
    for obs in self.components:
      obs.extend_obs_space(obs_space)
    self.observation_space = spaces.Dict(obs_space)
    logging.info(str(self.observation_space))

  def _handle_reset(self) -> dict:
    obs_dict = {}
    assert self.me
    for obs in self.components:
      obs.handle_reset(self.state_scenario, self.me, self.entities, obs_dict)
    return obs_dict

  def _handle_step(self) -> Tuple[object, float, bool, object]:
    obs_dict = {}
    assert self.me
    for obs in self.components:
      obs.handle_step(self.me, self.entities, obs_dict)
    return obs_dict, self.objective.rew, self.objective.done, {}
