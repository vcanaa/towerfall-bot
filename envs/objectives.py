import logging
import random
import numpy as np

from abc import abstractmethod

from gym import spaces, Space

from common import Entity, GridView, WIDTH, HEIGHT, HW, HH, Vec2, grid_pos

from .base_env import TowerfallEnv
from .observations import TowerfallObservation

from typing import Optional, Tuple
from numpy.typing import NDArray


class TowerfallObjective(TowerfallObservation):
  def __init__(self):
    self.done: bool
    self.rew: float
    self.env: TowerfallEnv

  def is_reset_valid(self, state_scenario: dict, player: Entity, entities: list[Entity]) -> bool:
    return True

  def get_reset_instruction(self) -> dict:
    '''Specifies how the environment needs to be reset.'''
    return {}


class FollowTargetObjective(TowerfallObjective):
  '''
  Specifies observation and rewards associated with moving to a target location.

  :param grid_view: Used to detect collisions when resetting the target.
  :param bounty: Reward received when reaching the location
  :param episode_max_len: Amount of frames after which the episode ends
  :param rew_dc: Agent loses this amount of reward per frame, in order to force it to get the target faster
  '''
  def __init__(self, grid_view: GridView, bounty: int=50, episode_max_len: int=60*2, rew_dc=1):
    super(FollowTargetObjective, self).__init__()
    self.gv = grid_view
    self.bounty = bounty
    self.episode_max_len = episode_max_len
    self.episode_len = 0
    self.rew_dc = rew_dc
    self.obs_space = spaces.Box(low=-2, high = 2, shape=(2,), dtype=np.float32)

  def extend_obs_space(self, obs_space_dict: dict[str, Space]):
    if 'target' in obs_space_dict:
      raise Exception('Observation space already has \'target\'')
    obs_space_dict['target'] = self.obs_space

  def post_reset(self, state_scenario: dict, player: Entity, entities: list[Entity], obs_dict: dict, target: Optional[Tuple[float, float]] = None):
    if target:
      self.set_target(player, *target)
    else:
      self._set_random_target(player)
    obs_dict['target'] = self.obs_target

  def post_step(self, player: Entity, entities: list[Entity], command: str, obs_dict: dict):
    self._update_reward(player)
    self.episode_len += 1
    self.env.draws({
      'type': 'line',
      'start': player['pos'],
      'end': self.target['pos'],
      'color': [1,1,1],
      'thick': 4
    })
    obs_dict['target'] = self.obs_target

  def _update_reward(self, player: Entity):
    displ = self._get_target_displ(player)
    disp_len = displ.length()
    self.rew = self.prev_disp_len - disp_len
    if disp_len < player.s.y / 2:
      # Reached target. Gets big reward
      self.rew += self.bounty
      self.done = True
      logging.info('Done. Reached target.')
    if self.episode_len > self.episode_max_len:
      self.done = True
      logging.info('Done. Timeout.')
    self.prev_disp_len = disp_len

  def set_target(self, player: Entity, x, y):
    self.target = Entity(e = {
      'pos': {'x': x, 'y': y},
      'vel': {'x': 0, 'y': 0},
      'size':{'x': 5, 'y': 5},
      'isEnemy': False,
      'type': 'fake'
    })
    displ = self._get_target_displ(player)
    self.obs_target: NDArray = np.array([displ.x / HW, displ.y / HH], dtype=np.float32)
    self.prev_disp_len = displ.length()
    self.done = False
    self.episode_len = 0

  def _set_random_target(self, player: Entity):
    while True:
      x = random.randint(0, WIDTH)
      y = random.randint(0, HEIGHT)
      i, j = grid_pos(Vec2(x, y), self.gv.csize)
      # logging.info('(i, j): ({} {})'.format(i, j))
      if not self.gv.fixed_grid10[i][j]:
        break
    # logging.info('New target: (x, y): ({} {})'.format(x, y))
    self.set_target(player, x, y)

  def _get_target_displ(self, player: Entity):
    '''Gets the displacement of of the target from the player.'''
    displ = self.target.p.copy()
    displ.sub(player.p)
    return displ