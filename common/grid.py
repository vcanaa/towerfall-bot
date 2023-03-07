import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray

from .common import WIDTH, HEIGHT, Entity, bounded, log

from typing import List


def plot_grid(grid: NDArray, name: str):
  H = grid.transpose()

  fig = plt.figure(figsize=(6, 3.2))

  ax = fig.add_subplot(111)
  ax.set_title(name)
  plt.imshow(H)
  ax.set_aspect('equal')
  ax.invert_yaxis()
  plt.show()


def fill_grid(e: Entity, grid: NDArray, shouldLog = False):
  factor = WIDTH / grid.shape[0]
  if WIDTH % grid.shape[0] != 0:
    raise Exception('fillGrid requires factor to be integer: {}'.format(factor))
  if HEIGHT / grid.shape[1] != factor:
    raise Exception('Invalid aspect rate for grid: {}'.format(grid.shape))
  topLeft = e.topLeft()
  botRight = e.bottomRight()
  x1 = int(bounded(topLeft.x // factor, 0, grid.shape[0]))
  x2 = int(bounded(botRight.x // factor, 0, grid.shape[0]))
  y1 = int(bounded(topLeft.y // factor, 0, grid.shape[1]))
  y2 = int(bounded(botRight.y // factor, 0, grid.shape[1]))
  if shouldLog:
    log("{} {} {} {}".format(x1,x2,y1,y2))
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


class GridView():
  def __init__(self, grid_factor: int):
    self.gf: int = grid_factor

  def set_scenario(self, game_state: dict):
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
    self.grid = self.fixed_grid.copy()
    for e in entities:
      if e.type == 'crackedWall':
        fill_grid(e, self.grid)
    # self._plot_grid(a, 'a')
    self.shifted_grid = np.roll(self.grid, -int(me.p.x - WIDTH // 2) // self.gf, axis=0)
    self.shifted_grid = np.roll(self.shifted_grid, -int(me.p.y - HEIGHT // 2) // self.gf, axis=1)
    # log('obs_grid: {}'.format(self.obs_grid.shape))
    # self._plot_grid(self.obs_grid, 'obs_grid')
    # print(self.obs_grid)


  def view_length(self, sight: int):
    return sight // self.gf


  def view(self, sight: int):
    n: int = self.view_length(sight)
    W, H = self.fixed_grid.shape
    return self.shifted_grid[
      W // 2 - n: W // 2 + n,
      H // 2 - n: H // 2 + n,]
