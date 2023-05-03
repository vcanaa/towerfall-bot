import logging

from gym import spaces

from .base_env import TowerfallEnv
from .actions import TowerfallActions
from .observations import TowerfallObservation
from .objectives import TowerfallObjective
from .connection_provider import TowerfallProcess

from typing import Tuple, Optional


class TowerfallBlankEnv(TowerfallEnv):
  '''A blank environment that can be customized with the addition of observations and an objective.'''
  def __init__(self,
      towerfall: TowerfallProcess,
      observations: list[TowerfallObservation],
      objective: TowerfallObjective,
      actions: Optional[TowerfallActions]=None,
      record_path: Optional[str]=None,
      verbose: int = 0):
    super(TowerfallBlankEnv, self).__init__(towerfall, actions, record_path, verbose)
    logging.info('Initializing TowerfallBlankEnv')
    obs_space = {}
    self.components = list(observations)
    self.components.append(objective)
    self.objective = objective
    self.objective.env = self
    for obs in self.components:
      obs.extend_obs_space(obs_space)
    self.observation_space = spaces.Dict(obs_space)
    logging.info('Action space: %s', str(self.action_space))
    logging.info('Observation space: %s', str(self.observation_space))

  def _is_reset_valid(self) -> bool:
    return self.objective.is_reset_valid(self.state_scenario, self.me, self.entities)

  def _send_reset(self):
    reset_entities = self.objective.get_reset_entities()
    self.towerfall.send_reset(reset_entities, verbose=self.verbose)

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
    # logging.info(f'reward: {self.objective.rew}')
    return obs_dict, self.objective.rew, self.objective.done, {}
