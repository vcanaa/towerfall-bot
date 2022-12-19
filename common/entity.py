import sys

from typing import Any, List

class Entity:
  def __init__(self, e: dict):
    self.p: Vec2 = Vec2(e['pos'])
    self.v: Vec2 = Vec2(e['vel'])
    self.s: Vec2 = Vec2(e['size'])
    self.isEnemy: bool = e['isEnemy']
    self.type: str = e['type']
    self.e: Any = e

  def __getitem__(self, key):
    return self.e[key]


class Vec2:
  def __init__(self, p: dict):
    try:
      self.x: float = p['x']
      self.y: float = p['y']
    except KeyError:
      sys.stderr.write(str(p))


def to_entities(entities: List[dict]) -> List[Entity]:
  result = []
  for e in entities:
    result.append(Entity(e))

  return result


def is_arrow_pickup(e: Entity) -> bool:
  return e.type == 'item' and e['itemType'].startswith('arrow')
