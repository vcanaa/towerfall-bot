import sys
import json
import random

from multiprocessing import shared_memory

import numpy as np

from numpy.typing import ArrayLike, NDArray
from threading import Lock

from common import reply, hasArrows, chance, distance, distance2, Controls, Entity, Vec2, to_entities, is_arrow_pickup, log

from typing import Any, List, Optional

class QuestBot:
  def __init__(self):
    self.grid: ArrayLike
    self.cgrid: ArrayLike # Centered grid
    self.stateInit: Any
    self.stateUpdate: Any
    self.me: Optional[Entity] = None
    self.control: Controls = Controls()
    self.entities: List[Entity]
    self.widget_update_cd: int = 0
    self.update_lock = Lock()
    self.shm = shared_memory.SharedMemory(name='towerfallScreen')


  def run(self):
    while True:
      self.update()


  def update(self):
    gameState = json.loads(sys.stdin.readline())
    if gameState['type'] == 'init':
      self.handleInit(gameState)

    if gameState['type'] == 'scenario':
      self.handleScenario(gameState)

    if gameState['type'] == 'update':
      self.handleUpdate(gameState)


  def handleInit(self, state: dict):
    log("handleInit")
    self.stateInit = state
    random.seed(state['index'])
    reply()


  def handleScenario(self, state: dict):
    log("handleScenario")
    self.grid = np.array(state['grid'])
    reply()


  def getPlayer(self, entities: List[Entity]):
    for e in entities:
      if e.type == 'archer':
        if e['playerIndex'] == self.stateInit['index']:
          self.me = e
          return e


  def getClosestEnemy(self, entities: List[Entity]):
    closest: Optional[Entity] = None
    minDist = 99999999
    for e in entities:
      if not e.isEnemy:
        continue
      dist = distance2(self.me.p, e.p)
      if (dist < minDist):
        closest = e
        minDist = dist
    return closest


  def getClosestArrowPickup(self) -> Optional[Entity]:
    closest: Optional[Entity] = None
    minDist = 99999999
    for e in self.entities:
      if not is_arrow_pickup(e):
        continue
      dist = distance2(self.me.p, e.p)
      if (dist < minDist):
        closest = e
        minDist = dist
    return closest


  def getDist(self, p: Vec2):
    return distance(self.me.p, p)


  def fightWithArrows(self, enemy: Entity):
    dist = distance(self.me.p, enemy.p)
    height = self.me.s.y
    self.chase(enemy.p)

    if dist < height * 5:
      if chance(20) == 0:
        self.control.shoot()
        if abs(enemy.p.y - self.me.p.y) >= abs(enemy.p.x - self.me.p.x):
          if self.isAbove(enemy):
            self.control.up()
          else:
            self.control.down()


  def isAbove(self, ent: Entity, margin: float = 0):
    return self.me.p.y + margin <= ent.p.y


  def chase(self, pos: Vec2):
    if self.me.p.x < pos.x:
      self.control.right()
    else:
      self.control.left()

    if self.me['onGround']:
      if self.me.p.y < pos.y:
        self.control.up()
      else:
        self.control.down()


  def flee(self, pos: Vec2):
    if self.me.p.x < pos.x:
      self.control.left()
    else:
      self.control.right()


  def fightWithoutArrows(self, enemy: Entity):
    enemy_dist: float = self.getDist(enemy.p)
    arrow: Optional[Entity] = self.getClosestArrowPickup()

    if arrow:
      self.chase(arrow.p)
      return

    if enemy_dist > self.me.s.y * 4:
      self.chase(enemy.p)
      return
    if self.isAbove(enemy, -enemy.s.y):
      self.flee(enemy.p)
    else:
      self.chase(enemy.p)


  def adjustEntitiesPos(self):
    x = self.me.e['pos']['x']
    y = self.me.e['pos']['y']
    for e in self.entities:
      e.p.x -= x
      e.p.y -= y


  def handleUpdate(self, state):
    self.update_lock.acquire()
    try:
      self.entities: List[Entity] = to_entities(state['entities'])
      self.getPlayer(self.entities)

      # self.cgrid = np.roll(self.grid, int(self.me.p.x / 10), axis=0)
      # self.adjustEntitiesPos()

      if self.me == None:
        reply()
        return

      if chance(10):
        self.control.dash()

      if chance(20):
        self.control.jump()

      enemy = self.getClosestEnemy(self.entities)
      if not enemy:
        reply()
        return

      if hasArrows(self.me):
        self.fightWithArrows(enemy)
      else:
        self.fightWithoutArrows(enemy)

      self.control.reply()
    finally:
      self.update_lock.release()


  def get_game_screen(self) -> NDArray[np.uint8]:
    self.update_lock.acquire()
    screen_data = np.frombuffer(self.shm.buf, dtype=np.uint8)
    self.update_lock.release()
    return screen_data
