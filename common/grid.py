import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .common import Entity, Vec2, bounded, grid_pos
from .constants import HEIGHT, HH, HW, WIDTH


def plot_grid(grid: NDArray, name: str):
  H = grid.transpose()

  fig = plt.figure(figsize=(6, 3.2))

  ax = fig.add_subplot(111)
  ax.set_title(name)
  im = plt.imshow(H)
  im.set_clim(0, 1)
  ax.set_aspect('equal')
  ax.invert_yaxis()
  plt.show()


def fill_grid(e: Entity, grid: NDArray, shouldLog = False):
  factor = WIDTH / grid.shape[0]
  if WIDTH % grid.shape[0] != 0:
    raise Exception('fillGrid requires factor to be integer: {}'.format(factor))
  if HEIGHT / grid.shape[1] != factor:
    raise Exception('Invalid aspect rate for grid: {}'.format(grid.shape))
  bot_left = e.bot_left()
  top_right = e.top_right()
  x1 = int(bounded(bot_left.x // factor, 0, grid.shape[0]))
  x2 = int(bounded(top_right.x // factor, 0, grid.shape[0]))
  y1 = int(bounded(bot_left.y // factor, 0, grid.shape[1]))
  y2 = int(bounded(top_right.y // factor, 0, grid.shape[1]))
  if shouldLog:
    logging.info("{} {} {} {}".format(x1,x2,y1,y2))
  if x2 > x1:
    if y2 > y1:
      grid[x1:x2, y1:y2] = 1
    else:
      grid[x1:x2, y1:grid.shape[1]] = 1
      grid[x1:x2, 0:y2] = 1
  else:
    if y2 > y1:
      grid[x1:grid.shape[0], y1:y2] = 1
      grid[0:x2, y1:y2] = 1
    else:
      grid[x1:grid.shape[0], y1:grid.shape[1]] = 1
      grid[x1:grid.shape[0], 0:y2] = 1
      grid[0:x2, 0:grid.shape[1]] = 1
      grid[0:x2, 0:y2] = 1


def crop_grid(bot_left: Vec2, top_right: Vec2, grid: NDArray) -> NDArray:
  factor = WIDTH / grid.shape[0]
  if WIDTH % grid.shape[0] != 0:
    raise Exception('fillGrid requires factor to be integer: {}'.format(factor))
  if HEIGHT / grid.shape[1] != factor:
    raise Exception('Invalid aspect rate for grid: {}'.format(grid.shape))
  x1 = int(bot_left.x // factor)
  x2 = int(top_right.x // factor)
  y1 = int(bot_left.y // factor)
  y2 = int(top_right.y // factor)

  # print(x1, y1, x2, y2)
  # make region touch the grid
  if x2 < 0:
    x = x2 % grid.shape[0]
    x1 += x - x2
    x2 = x
  if x1 >= grid.shape[0]:
    x = x1 % grid.shape[0]
    x2 += x - x1
    x1 = x
  if y2 < 0:
    y = y2 % grid.shape[1]
    y1 += y - y2
    y2 = y
  if y1 >= grid.shape[1]:
    y = y1 % grid.shape[1]
    y2 += y - y1
    y1 = y

  if x2 > grid.shape[0]:
    x1 -= grid.shape[0]
    x2 -= grid.shape[0]
  if y2 > grid.shape[1]:
    y1 -= grid.shape[1]
    y2 -= grid.shape[1]

  if x1 < 0:
    if y1 < 0:
      return np.concatenate([
          np.concatenate([grid[x1:, y1:], grid[x1:, :y2]], axis=1),
          np.concatenate([grid[:x2, y1:], grid[:x2, :y2]], axis=1),
        ],
        axis=0)
    return np.concatenate([grid[x1:, y1:y2], grid[:x2, y1:y2]], axis=0)
  if y1 < 0:
    return np.concatenate([grid[x1:x2, y1:], grid[x1:x2, :y2]], axis=1)
  return grid[x1:x2, y1:y2]


class GridView():
  '''This is a representation of the scenario to show what parts of the screen are empty or occupied.
  It is capable of adjusting the resolution of the occupation matrix and recentering at different positions.'''

  def __init__(self, grid_factor: int):
    self.gf: int = grid_factor

  def set_scenario(self, game_state: Dict[str, Any]):
    # logging.info(f'Setting scenario in GridView {game_state["grid"]}')
    self.fixed_grid10 = np.array(game_state['grid'])
    self.csize: int = int(game_state['cellSize'])
    self.fixed_grid: NDArray = np.zeros((WIDTH // self.gf, HEIGHT // self.gf), dtype=np.int8)
    for i in range(self.fixed_grid10.shape[0]):
      for j in range(self.fixed_grid10.shape[1]):
        if self.fixed_grid10[i][j] == 1:
          self.fixed_grid[
            self.csize*i // self.gf:self.csize*(i+1) // self.gf,
            self.csize*j // self.gf:self.csize*(j+1) // self.gf] = 1


  def update(self, entities: List[Entity], me: Entity):
    '''To be called every frame to fill the empty spaces with the entities.'''
    self.grid = self.fixed_grid.copy()
    for e in entities:
      # TODO Add more entities to this
      if e.type == 'crackedWall':
        fill_grid(e, self.grid)
    # self._plot_grid(a, 'a')
    self.shifted_grid = np.roll(self.grid, -int(me.p.x - HW) // self.gf, axis=0)
    self.shifted_grid = np.roll(self.shifted_grid, -int(me.p.y - HH) // self.gf, axis=1)
    # logging.info('obs_grid: {}'.format(self.obs_grid.shape))
    # self._plot_grid(self.obs_grid, 'obs_grid')
    # print(self.obs_grid)

  def ray(self, pos: Vec2, step: Vec2, max: float) -> float:
    # This is not tested
    p = pos.copy()
    dist = float('-inf')
    for i in range(int(max) // self.csize + 1):
      p.add(step)
      i, j = grid_pos(p, self.csize)
      if self.fixed_grid10[i, j]:
        if step.x > 0:
          dist = (int(p.x) // self.csize) * self.csize - pos.x
          break
        if step.x < 0:
          dist = pos.x + (int(p.x) // self.csize + 1) * self.csize
          break
        if step.y > 0:
          dist = (int(p.y) // self.csize) * self.csize - pos.y
          break
        if step.y < 0:
          dist = pos.y + (int(p.y) // self.csize + 1) * self.csize
          break
    return min(max, dist)


  def view_sight_length(self, sight: Optional[Union[int, Tuple[int, int]]]) -> Tuple[int, int]:
    '''Gets the length of view using the specified sight.'''
    if not sight:
      return HW // self.gf, HH // self.gf
    if isinstance(sight, int):
      m = n = min(sight // self.gf, HH // self.gf )
    else:
      m = min(sight[0] // self.gf, HW // self.gf)
      n = min(sight[1] // self.gf, HH // self.gf)
    return m, n


  def view(self, sight: Optional[Union[int, Tuple[int, int]]]):
    '''Gets a view of the grid using the specified sight'''
    if not sight:
      return self.shifted_grid
    m, n = self.view_sight_length(sight)
    W, H = self.fixed_grid.shape
    return self.shifted_grid[
        W // 2 - m: W // 2 + m,
        H // 2 - n: H // 2 + n]

  def is_cell_blocked(self, i: int, j: int) -> bool:
    '''Checks whether a cell is blocked by walls'''
    m, n = self.fixed_grid10.shape
    i %= m
    j %= n
    return self.fixed_grid10[i, j]

  def is_region_collision(self, bot_left: Vec2, top_right: Vec2):
    # TODO make crop do the wrap around the edges so we dont end up with empty crops
    crop = crop_grid(bot_left, top_right, self.grid)
    if crop.shape[1] == 0:
      return True
    try:
      return crop.max() != 0
    except:
      print(f'crop: {crop} {len(crop)} {crop.shape} {bot_left} {top_right}')
      raise Exception()

  def is_clean_path(self, p1: Vec2, p2: Vec2) -> bool:
    dp = p2 - p1
    dmax = int(dp.length() / self.gf)
    dp.set_length(self.gf)
    p = p1.copy()
    for i in range(dmax):
      pgrid = grid_pos(p, self.gf)
      if self.grid[pgrid] > 0:
        return False
      p.add(dp)
    return True
