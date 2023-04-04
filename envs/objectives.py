import numpy as np
import logging
import random

# from abc import ABC, abstractmethod

from gym import spaces, Space

from common import Entity, GridView, WIDTH, HEIGHT, Vec2, grid_pos

from typing import Sequence, Optional, Tuple
from numpy.typing import NDArray


HW = WIDTH // 2
HH = HEIGHT // 2

class FollowTargetObjective():
  def __init__(self, env, grid_view: GridView, bounty: int=50, episode_max_len: int=60*5):
    self.env = env
    self.gv = grid_view
    self.bounty = bounty
    self.episode_max_len = episode_max_len
    self.episode_len = 0
    self.obs_space = spaces.Box(low=-2, high = 2, shape=(2,), dtype=np.float32)

  def extend_obs_space(self, obs_space_dict: dict[str, Space]):
    if 'target' in obs_space_dict:
      raise Exception('Observation space already has \'target\'')
    obs_space_dict['grid'] = self.obs_space

  def handle_reset(self, player: Entity):
    self._set_new_target(player)
    displ = self._get_target_displ(player)
    self.obs_target: NDArray = np.array([displ.x / HW, displ.y / HH], dtype=np.int8)
    self.done = False
    self.episode_len = 0

  def handle_update(self, player: Entity):
    self._update_reward(player)
    self.episode_len += 1
    self.env.draws({
      'type': 'line',
      'start': player['pos'],
      'end': self.target['pos'],
      'color': [1,1,1],
      'thick': 4
    })

  def extend_obs(self, obs_dict: dict):
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
    self.obs_target: NDArray = np.array([displ.x / HW, displ.y / HH], dtype=np.float32)
    if self.done:
      self._set_new_target(player)

  def _set_new_target(self, player: Entity):
    while True:
      x = random.randint(0, WIDTH)
      y = random.randint(0, HEIGHT)
      i, j = grid_pos(Vec2(x, y), self.gv.csize)
      # logging.info('(i, j): ({} {})'.format(i, j))
      if not self.gv.fixed_grid10[i][j]:
        break
    # logging.info('New target: (x, y): ({} {})'.format(x, y))
    self.target = Entity(e = {
      'pos': {'x': x, 'y': y},
      'vel': {'x': 0, 'y': 0},
      'size':{'x': 5, 'y': 5},
      'isEnemy': False,
      'type': 'fake'
    })
    # New target is only used in the next loop.
    self.prev_disp_len = self._get_target_displ(player).length()

  def _get_target_displ(self, player: Entity):
    '''Gets the displacement of of the target from the player.'''
    displ = self.target.p.copy()
    displ.sub(player.p)
    return displ