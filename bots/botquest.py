import json
import random
import logging

from common import *

from .bot import Bot

from typing import Any, Dict, List, Optional, Tuple

_HOST = "127.0.0.1"
_PORT = 12024

class BotQuest(Bot):
  def __init__(self):
    super(BotQuest, self).__init__()
    self.gv = GridView(1)
    self.state_init: Any
    self.state_update: Any
    self.control: Controls = Controls()
    self.entities: List[Entity]
    self.shootcd: float = 0
    self._connection = Connection(_HOST, _PORT)


  def __del__(self):
    del self._connection


  def run(self):
    while True:
      self.update()


  def update(self):
    game_state = json.loads(self._connection.read())
    if game_state['type'] == 'init':
      self.handle_init(game_state)

    if game_state['type'] == 'scenario':
      self.handle_scenario(game_state)

    if game_state['type'] == 'update':
      self.handle_update(game_state)


  def handle_init(self, state: Dict[str, Any]):
    logging.info("handle_init")
    self.state_init = state
    random.seed(state['index'])
    self._connection.write('.')


  def handle_scenario(self, state: Dict[str, Any]):
    logging.info("handle_scenario")
    self.state_scenario = state
    self.gv.set_scenario(state)
    self._connection.write('.')


  def get_player(self, entities: List[Entity]):
    for e in entities:
      if e.type == 'archer':
        if e['playerIndex'] == self.state_init['index']:
          self.me: Entity = e
          return e


  def get_closest_enemy(self, entities: List[Entity]) -> Tuple[Optional[Entity], Optional[GPath]]:
    enemies = [e for e in entities if e['isEnemy']]
    path_grid = PathGrid(self.gv.csize)
    return path_grid.get_closest_entity(vec2_from_dict(self.me['pos']), enemies, self.gv.fixed_grid10)


  def get_closest_stuck_arrow(self) -> Tuple[Optional[Entity], Optional[GPath]]:
    stuck_arrows = [e for e in self.entities if e.type == 'arrow' and is_stuck_arrow(e)]
    path_grid = PathGrid(self.gv.csize)
    return path_grid.get_closest_entity(vec2_from_dict(self.me['pos']), stuck_arrows, self.gv.fixed_grid10)


  def get_dist(self, p: Vec2):
    return distance(self.me.p, p)


  def fight_with_arrows(self, enemy: Entity, path_to_enemy: GPath):
    self.target = enemy
    self.path_to_target = path_to_enemy
    dist = distance(self.me.p, enemy.p)

    self.control.aim(diff(enemy.p, self.me.p))
    if self.shootcd <= 0:
      if is_clean_path(self.me.p, enemy.p, self.gv.fixed_grid10, self.gv.csize):
        self.shoot()
    else:
      self.fight_without_arrows(enemy, path_to_enemy)


  def is_above(self, ent: Entity, margin: float = 0):
    return self.me.p.y + margin <= ent.p.y


  def chase(self, pos: Vec2):
    if not self.me['onGround']:
      self.control.aim(diff(pos, self.me.p))
    else:
      if self.me.p.x < pos.x:
        self.control.right()
      else:
        self.control.left()


  def flee(self, pos: Vec2):
    if not self.me['onGround']:
      self.control.aim(diff(self.me.p, pos))
    else:
      if self.me.p.x < pos.x:
        self.control.left()
      else:
        self.control.right()


  def fight_without_arrows(self, enemy: Entity, path_to_enemy: GPath):
    self.target = enemy
    self.path_to_target = path_to_enemy
    enemy_dist: float = self.get_dist(enemy.p)
    arrow, path_to_arrow = self.get_closest_stuck_arrow()

    if arrow and path_to_arrow:
      self.target = arrow
      self.path_to_target = path_to_arrow
      self.chase(path_to_arrow.checkpoint.pos)
      return

    if enemy_dist > self.me.s.y * 4:
      self.chase(enemy.p)
      return
    if self.is_above(enemy, -enemy.s.y):
      self.flee(enemy.p)
    else:
      self.chase(enemy.p)


  def handle_update(self, state):
    self.update_lock.acquire()
    try:
      self.shootcd -= 16

      self.target = None
      self.path_to_target = None
      self.entities: List[Entity] = to_entities(state['entities'])
      self.get_player(self.entities)

      if self.me == None:
        raise Exception('No me?')

      self.gv.update(self.entities, self.me)

      if chance(10):
        self.control.dash()

      if chance(20):
        self.control.jump()

      enemy, path_to_enemy = self.get_closest_enemy(self.entities)
      if not enemy or not path_to_enemy:
        arrow, path_to_arrow = self.get_closest_stuck_arrow()
        if arrow and path_to_arrow:
          self.target = arrow
          self.path_to_target = path_to_arrow
          self.chase(path_to_arrow.checkpoint.pos)
        self.control.reply(self._connection)
        return

      if has_arrows(self.me):
        self.fight_with_arrows(enemy, path_to_enemy)
      else:
        self.fight_without_arrows(enemy, path_to_enemy)

      self.control.reply(self._connection)
    finally:
      self.update_lock.release()


  def shoot(self):
    self.control.shoot()
    self.shootcd = 600


  def get_entities(self) -> List[Tuple[str, object]]:
    result = []
    if self.target:
      result.append(('*target', self.target, self.path_to_target))
    result.extend([(e.type, e) for e in self.entities])
    return result

  def grid(self):
    return self.gv.fixed_grid10
