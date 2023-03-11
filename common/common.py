import sys
import random

from math import sqrt

from .entity import Vec2, Entity
from .constants import WIDTH, HEIGHT

from typing import Optional, Tuple
from numpy.typing import NDArray


def reply(msg: Optional[str] = None):
  if msg:
    sys.stdout.write(msg)
  sys.stdout.write('\n')
  sys.stdout.flush()


def press(b: str):
  sys.stdout.write(b)


def has_arrows(ent: Entity) -> bool:
  return len(ent['arrows']) > 0


def chance(i: int) -> bool:
  if i == 1:
    return True
  return random.randint(0, i - 1) == 0


def distance(p1: Vec2, p2: Vec2) -> float:
  return sqrt(distance2(p1, p2))


def distance2(p1: Vec2, p2: Vec2) -> float:
  return (p1.x - p2.x)**2 + (p1.y - p2.y)**2


def diff(p1: Vec2, p2: Vec2) -> Vec2:
  return Vec2(p1.x - p2.x, p1.y - p2.y)


def bounded(v, left, right):
  if v < left:
    return v + right - left
  if v >= right:
    return v - (right - left)
  return v


def grid_pos(p: Vec2, cell_size: int) -> Tuple[int, int]:
  return (int(p.x % WIDTH) // cell_size, int(p.y % HEIGHT) // cell_size)


def is_clean_path(p1: Vec2, p2: Vec2, grid: NDArray, cell_size: int) -> bool:
  dp = diff(p2, p1)
  dp.set_length(5)
  p = p1.copy()
  for i in range(int(dp.length())):
    pgrid = grid_pos(p, cell_size)
    if grid[pgrid[0]][pgrid[1]] == 1:
      return False
    p.add(dp)
  return True


def rand_double_region(a: float, b: float):
  x = random.randint(int(a), int(b))
  return x if random.randint(0, 1) else -x

