import sys
import random

from math import sqrt

from .entity import Vec2, Entity

from typing import Optional


def reply(msg: Optional[str] = None):
  if msg:
    sys.stdout.write(msg)
  sys.stdout.write('\n')
  sys.stdout.flush()


def log(msg: str):
  sys.stderr.write(msg)
  sys.stderr.write('\n')
  sys.stderr.flush()


def press(b: str):
  sys.stdout.write(b)


def hasArrows(ent: Entity) -> bool:
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

