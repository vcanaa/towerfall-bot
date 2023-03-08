import numpy as np
import tkinter as tk
from typing import List
from numpy.typing import NDArray

from typing import Tuple

class Grid:
  def __init__(self, size: Tuple[int, int], canvas: tk.Canvas, fill: str = 'red', stipple: str = 'gray25'):
    self.canvas = canvas
    self.grid_rects: List[int] = []
    self.is_visible = False
    self.grid = np.zeros([0])
    self.size = size
    self.fill = fill
    self.stipple = stipple

  def _delete(self):
    for r in self.grid_rects:
      self.canvas.delete(r)


  def _create_rects(self):
    m: int = self.grid.shape[0]
    n: int = self.grid.shape[1]
    cellx = self.size[0] // m
    celly = self.size[1] // n
    for i in range(m):
      for j in range(n):
        if self.grid[i][j] == 1:
          self.grid_rects.append(
            self.canvas.create_rectangle(
                i*cellx, (n-j)*celly,(i+1)*cellx,(n - j -1)*celly,
                fill=self.fill))


  def update(self, grid: NDArray):
    self._delete()
    self.grid = grid
    if self.is_visible:
      self._create_rects()


  def show(self):
    if self.is_visible:
      return

    self._create_rects()
    self.is_visible = True


  def hide(self):
    self._delete()
    self.is_visible = False


  def toggle(self):
    if self.is_visible:
      self.hide()
    else:
      self.show()
