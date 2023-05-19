import sys

sys.path.insert(0, '.')

from typing import List

import numpy as np
from numpy.typing import NDArray

from common import Vec2, crop_grid

m = 8
n = 6
grid = np.arange(m*n).reshape((m, n))
print(grid)

def print_crop(g):
  s = ['[\n']
  for i in range(g.shape[0]):
    s.append('  [')
    for j in range(g.shape[1]):
      s.append(f'{g[i, j]:3d}'.rjust(2))
      s.append(',')
    s.append('],\n')
  s.append(']')
  print(''.join(s))


def test_combinations(bl: Vec2, tr: Vec2, grid: NDArray, expected: List[List[int]]):
  for dx in range(-320, 321, 320):
    for dy in range(-240, 241, 240):
      d = Vec2(dx, dy)
      a = crop_grid(bl + d, tr + d, grid)
      assert np.array_equal(a, expected), f'{bl}, {tr}\n {a}'


test_combinations(
  Vec2(0, 0),
  Vec2(160, 160),
  grid,
  [
    [  0,  1,  2,  3,],
    [  6,  7,  8,  9,],
    [ 12, 13, 14, 15,],
    [ 18, 19, 20, 21,],
  ])

test_combinations(
  Vec2(-80, 0),
  Vec2(80, 160),
  grid,
  [
    [36, 37, 38, 39,],
    [42, 43, 44, 45,],
    [ 0,  1,  2,  3,],
    [ 6,  7,  8,  9,]
  ])

test_combinations(
  Vec2(-80, -80),
  Vec2(80, 80),
  grid,
  [
    [ 40, 41, 36, 37,],
    [ 46, 47, 42, 43,],
    [  4,  5,  0,  1,],
    [ 10, 11,  6,  7,],
  ])

test_combinations(
  Vec2(0, -80),
  Vec2(160, 80),
  grid,
  [
    [  4,  5,  0,  1,],
    [ 10, 11,  6,  7,],
    [ 16, 17, 12, 13,],
    [ 22, 23, 18, 19,],
  ])
