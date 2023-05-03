import numpy as np

from abc import ABC, abstractmethod

from gym import spaces, Space

from common import Entity, GridView, JUMP, DASH, SHOOT

from typing import Sequence, Optional, Tuple, Union


class TowerfallObservation(ABC):
  '''
  Base class for observations.
  '''
  @abstractmethod
  def extend_obs_space(self, obs_space_dict: dict[str, Space]):
    '''Adds the new definitions to observations to obs_space.'''
    raise NotImplementedError()

  @abstractmethod
  def post_reset(self, state_scenario: dict, player: Optional[Entity], entities: list[Entity], obs_dict: dict):
    '''Hook for a gym reset call. Adds observations to obs_dict.'''
    raise NotImplementedError

  @abstractmethod
  def post_step(self, player: Optional[Entity], entities: list[Entity], command: str, obs_dict: dict):
    '''Hook for a gym step call. Adds observations to obs_dict.'''
    raise NotImplementedError


class PlayerObservation(TowerfallObservation):
  '''Observation for the player state.'''
  def __init__(self, exclude: Optional[Sequence[str]] = None):
    self.exclude = exclude

  def extend_obs_space(self, obs_space_dict: dict[str, Space]):
    def try_add_obs(key, value):
      if self.exclude and key in self.exclude:
        return
      if key in obs_space_dict.keys():
        raise Exception(f'Observation space already has {key}')
      obs_space_dict[key] = value

    try_add_obs('prev_jump', spaces.Discrete(2))
    try_add_obs('prev_dash', spaces.Discrete(2))
    try_add_obs('prev_shoot', spaces.Discrete(2))

    try_add_obs('dodgeCooldown', spaces.Discrete(2))
    try_add_obs('dodging', spaces.Discrete(2))
    try_add_obs('facing', spaces.Discrete(2))
    try_add_obs('onGround', spaces.Discrete(2))
    try_add_obs('onWall', spaces.Discrete(2))
    try_add_obs('vel', spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float32))

  def post_reset(self, state_scenario: dict, player: Optional[Entity], entities: list[Entity], obs_dict: dict):
    self._extend_obs(player, '', obs_dict)

  def post_step(self, player: Optional[Entity], entities: list[Entity], command: str, obs_dict):
    self._extend_obs(player, command, obs_dict)

  def _extend_obs(self, player: Optional[Entity], command: str, obs_dict: dict):
    def try_add_obs(key, value):
      if self.exclude and key in self.exclude:
        return
      obs_dict[key] = value

    try_add_obs('prev_jump', int(JUMP in command))
    try_add_obs('prev_dash', int(DASH in command))
    try_add_obs('prev_shoot', int(SHOOT in command))
    if not player:
      try_add_obs('dodgeCooldown', 0)
      try_add_obs('dodging', 0)
      try_add_obs('facing', 0)
      try_add_obs('onGround', 0)
      try_add_obs('onWall', 0)
      try_add_obs('vel', np.zeros(2))
      return

    try_add_obs('dodgeCooldown', int(player['dodgeCooldown']))
    try_add_obs('dodging', int(player['state']=='dodging'))
    try_add_obs('facing', (player['facing'] + 1) // 2) # -1,1 -> 0,1
    try_add_obs('onGround', int(player['onGround']))
    try_add_obs('onWall', int(player['onWall']))
    try_add_obs('vel', np.clip(player.v.numpy() / 5, -2, 2))


class GridObservation(TowerfallObservation):
  '''
  Shows a cropped view of the grid around the player.
  '''
  def __init__(self, grid_view: GridView, sight: Optional[Union[Tuple[int, int], int]] = None, add_grid: bool = True):
    self.gv = grid_view
    if isinstance(sight, int):
      sight = (sight, sight)

    self.sight = sight
    m, n = self.gv.view_sight_length(sight)
    self.obs_space = spaces.MultiBinary((2*m, 2*n))
    self.frame = 0
    self.add_grid = add_grid

  def extend_obs_space(self, obs_space_dict: dict[str, Space]):
    if self.add_grid:
      if 'grid' in obs_space_dict:
        raise Exception('Observation space already has \'grid\'')
      obs_space_dict['grid'] = self.obs_space

  def post_reset(self, state_scenario: dict, player: Optional[Entity], entities: list[Entity], obs_dict: dict):
    self.frame = 0
    self.gv.set_scenario(state_scenario)
    self._update_grid(player, entities)

    self._extend_obs(obs_dict)

  def post_step(self, player: Optional[Entity], entities: list[Entity], command: str, obs_dict: dict):
    self._update_grid(player, entities)
    self._extend_obs(obs_dict)
    self.frame += 1

  def _update_grid(self, player: Optional[Entity], entities: list[Entity]):
    if player:
      self.gv.update(entities, player)
    else:
      assert self.prev_player
      self.gv.update(entities, self.prev_player)
    self.prev_player = player

  def _extend_obs(self, obs_dict: dict):
    if self.add_grid:
      obs_dict['grid'] = self.gv.view(self.sight)
