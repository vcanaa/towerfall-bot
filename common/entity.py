from __future__ import annotations

import sys

from typing import Any, List
from math import sqrt


class Entity:
  def __init__(self, e: dict):
    self.p: Vec2 = vec2_from_dict(e['pos'])
    self.v: Vec2 = vec2_from_dict(e['vel'])
    self.s: Vec2 = vec2_from_dict(e['size'])
    self.isEnemy: bool = e['isEnemy']
    self.type: str = e['type']
    self.e: Any = e


  def __getitem__(self, key):
    return self.e[key]


  def topLeft(self) -> Vec2:
    return Vec2(self.p.x - self.s.x / 2, self.p.y - self.s.y / 2)


  def bottomRight(self) -> Vec2:
    return Vec2(self.p.x + self.s.x / 2, self.p.y + self.s.y / 2)


class Vec2:
  def __init__(self, x: float, y: float):
    self.x: float = x
    self.y: float = y


  def __str__(self):
    return '({}, {})'.format(self.x, self.y)


  def set_length(self, l: float):
    d = self.length()
    self.x *= l/d
    self.y *= l/d


  def length(self):
    return sqrt(self.x**2 + self.y**2)


  def copy(self):
    return Vec2(self.x, self.y)


  def add(self, v: Vec2):
    self.x += v.x
    self.y += v.y

  def sub(self, v: Vec2):
    self.x -= v.x
    self.y -= v.y

  def mul(self, f: float):
    self.x *= f
    self.y *= f

  def div(self, f: float):
    self.x /= f
    self.y /= f


def vec2_from_dict(p: dict) -> Vec2:
  try:
    return Vec2(p['x'], p['y'])
  except KeyError:
    sys.stderr.write(str(p))
    raise


def to_entities(entities: List[dict]) -> List[Entity]:
  result = []
  for e in entities:
    result.append(Entity(e))

  return result


def is_arrow_pickup(e: Entity) -> bool:
  return e.type == 'item' and e['itemType'].startswith('arrow')


def is_stuck_arrow(e: Entity) -> bool:
  return e.type == 'arrow' and e['state'] == 'stuck'
