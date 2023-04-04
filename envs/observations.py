import numpy as np
import logging
import random

# from abc import ABC, abstractmethod

from gym import spaces, Space

from common import Entity, GridView, WIDTH, HEIGHT, Vec2, grid_pos

from typing import Sequence, Optional, Tuple
from numpy.typing import NDArray


# class TowerfallObservation(ABC):
#   # def __init__(self, child: Optional['TowerfallObservation'] = None):
#   #   self.obs_space: Space
#   #   self.child = child

#   # @abstractmethod
#   # def handle_reset(self, state_scenario: dict, player: Entity, entities: list[Entity]):
#   #   '''Hook for a gym reset call.'''
#   #   raise NotImplementedError

#   # @abstractmethod
#   # def handle_step(self, player: Entity, entities: list[Entity]) -> Tuple[NDArray, float, bool, object]:
#   #   '''Hook for a gym step call.'''
#   #   raise NotImplementedError

#   @abstractmethod
#   def extend_obs_space(self):
#     raise NotImplementedError()

#   @abstractmethod
#   def extend_obs(self):
#     raise NotImplementedError()

class PlayerObservation():
  def __init__(self, exclude: Optional[Sequence[str]] = None):
    self.exclude = exclude

  def extend_obs_space(self, obs_space_dict: dict[str, Space]):
    '''Adds the new definitions to observations to obs_space.'''
    def try_add_obs(key, value):
      if self.exclude and key in self.exclude:
        if key in obs_space_dict.keys():
          raise Exception(f'Observation space already has {key}')
        obs_space_dict[key] = value
    try_add_obs('dodgeCooldown', spaces.Discrete(2))
    try_add_obs('dodging', spaces.Discrete(2))
    try_add_obs('facing', spaces.Discrete(2))
    try_add_obs('onGround', spaces.Discrete(2))
    try_add_obs('onWall', spaces.Discrete(2))
    try_add_obs('target', spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32))
    try_add_obs('vel', spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float32))

  def extend_obs(self, player: Entity, obs_dict: dict):
    '''Adds the new observations to obs_dict.'''
    def try_add_obs(key, value):
      if self.exclude and key in self.exclude:
        return
      obs_dict[key] = value

    try_add_obs('dodgeCooldown', int(player['dodgeCooldown']))
    try_add_obs('dodging', int(player['state']=='dodging'))
    try_add_obs('facing', (player['facing'] + 1) // 2) # -1,1 -> 0,1
    try_add_obs('onGround', int(player['onGround']))
    try_add_obs('onWall', int(player['onWall']))
    try_add_obs('vel', player.v.array() / 5)


class GridObservation():
  def __init__(self, state_scenario: dict, grid_view: GridView, sight: Optional[Tuple[int, int]] = None):
    self.gv = grid_view
    self.gv.set_scenario(state_scenario)
    self.sight = sight
    m, n = self.gv.view_sight_length(sight)
    self.obs_space = spaces.MultiBinary((2*m, 2*n))

  def extend_obs_space(self, obs_space_dict: dict[str, Space]):
    if 'grid' in obs_space_dict:
      raise Exception('Observation space already has \'grid\'')
    obs_space_dict['grid'] = self.obs_space

  def extend_obs(self, player: Entity, entities: list[Entity], obs_dict: dict):
    obs_dict['grid'] = self.gv.view(self.sight)


