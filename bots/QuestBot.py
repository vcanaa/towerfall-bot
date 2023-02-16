import sys
import json
import random

from multiprocessing import shared_memory

import numpy as np

from numpy.typing import ArrayLike, NDArray
from threading import Lock

from math import cos, sin

from common import *

from typing import Any, List, Optional, Tuple

class QuestBot:
  def __init__(self):
    self.cgrid: ArrayLike # Centered grid
    self.stateInit: Any
    self.stateUpdate: Any
    self.control: Controls = Controls()
    self.entities: List[Entity]
    self.widget_update_cd: int = 0
    self.update_lock = Lock()
    self.shm = shared_memory.SharedMemory(name='towerfallScreen')
    self.shootcd: float = 0
    self.pause = False

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
    self.fixed_grid = np.array(state['grid'])
    reply()


  def getPlayer(self, entities: List[Entity]):
    for e in entities:
      if e.type == 'archer':
        if e['playerIndex'] == self.stateInit['index']:
          self.me: Entity = e
          return e


  def getClosestEnemy(self, entities: List[Entity]) -> Tuple[Optional[Entity], Optional[Path]]:
    enemies = [e for e in entities if e['isEnemy']]
    pathGrid = PathGrid()
    return pathGrid.getClosestEntity(vec2_from_dict(self.me['pos']), enemies, self.grid)


  def getClosestStuckArrow(self) -> Tuple[Optional[Entity], Optional[Path]]:
    stuck_arrows = [e for e in self.entities if e.type == 'arrow' and is_stuck_arrow(e)]
    pathGrid = PathGrid()
    return pathGrid.getClosestEntity(vec2_from_dict(self.me['pos']), stuck_arrows, self.grid)


  def getDist(self, p: Vec2):
    return distance(self.me.p, p)


  def fightWithArrows(self, enemy: Entity, pathToEnemy: Path):
    self.target = enemy
    self.pathToTarget = pathToEnemy
    dist = distance(self.me.p, enemy.p)
    # height = self.me.s.y
    # self.chase(enemy.p)

    self.control.aim(diff(enemy.p, self.me.p))
    # if dist < height * 5:
    if self.shootcd <= 0:
      # aim = vec2_from_dict(self.me.e['aimDirection'])
      if isCleanPath(self.me.p, enemy.p, self.grid):
        self.shoot()

    else:
      self.fightWithoutArrows(enemy, pathToEnemy)
    # else:
    #   self.fightWithoutArrows(enemy)


  def isAbove(self, ent: Entity, margin: float = 0):
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


  def fightWithoutArrows(self, enemy: Entity, pathToEnemy: Path):
    self.target = enemy
    self.pathToTarget = pathToEnemy
    enemy_dist: float = self.getDist(enemy.p)
    arrow, pathToArrow = self.getClosestStuckArrow()

    if arrow and pathToArrow:
      self.target = arrow
      self.pathToTarget = pathToArrow
      self.chase(pathToArrow.checkpoint.pos)
      return

    if enemy_dist > self.me.s.y * 4:
      self.chase(enemy.p)
      return
    if self.isAbove(enemy, -enemy.s.y):
      self.flee(enemy.p)
    else:
      self.chase(enemy.p)


  def adjustEntitiesPos(self):
    x = self.me['pos']['x']
    y = self.me['pos']['y']
    for e in self.entities:
      e.p.x -= x
      if (e.p.x < -160):
        e.p.x += 320
      if (e.p.x >= 160):
        e.p.x -= 320
      e.p.y -= y
      if (e.p.y < -120):
        e.p.y += 240
      if (e.p.y >= 120):
        e.p.y -= 240


  def fillGrid(self, e: Entity, grid):
    topLeft = e.topLeft()
    botRight = e.bottomRight()
    grid[bounded(int(topLeft.x/10), 0, 32)][bounded(int(topLeft.y/10), 0, 24)] = 1
    grid[bounded(int(botRight.x/10 - 0.001), 0, 32)][bounded(int(topLeft.y/10), 0, 24)] = 1
    grid[bounded(int(topLeft.x/10), 0, 32)][bounded(int(botRight.y/10 - 0.001), 0, 24)] = 1
    grid[bounded(int(botRight.x/10 - 0.001), 0, 32)][bounded(int(botRight.y/10 - 0.001), 0, 24)] = 1


  def updateGrid(self):
    self.grid = self.fixed_grid.copy()
    for e in self.entities:
      if e.type != 'crackedWall':
        continue
      self.fillGrid(e, self.grid)


  def handleUpdate(self, state):
    self.update_lock.acquire()
    try:
      self.shootcd -= 16

      self.target = None
      self.pathToTarget = None
      self.entities: List[Entity] = to_entities(state['entities'])
      self.getPlayer(self.entities)
      # self.adjustEntitiesPos()

      self.updateGrid()

      if self.me == None:
        reply()
        return

      if chance(10):
        self.control.dash()

      if chance(20):
        self.control.jump()

      enemy, pathToEnemy = self.getClosestEnemy(self.entities)
      if not enemy or not pathToEnemy:
        arrow, pathToArrow = self.getClosestStuckArrow()
        if arrow and pathToArrow:
          self.target = arrow
          self.pathToTarget = pathToArrow
          self.chase(pathToArrow.checkpoint.pos)
        reply()
        return

      if hasArrows(self.me):
        self.fightWithArrows(enemy, pathToEnemy)
      else:
        self.fightWithoutArrows(enemy, pathToEnemy)

      self.control.reply()
    finally:
      self.update_lock.release()


  def get_game_screen(self) -> NDArray[np.uint8]:
    self.update_lock.acquire()
    screen_data = np.frombuffer(self.shm.buf, dtype=np.uint8)
    self.update_lock.release()
    return screen_data


  def shoot(self):
    if self.target and self.target.type == 'arrow':
      raise Exception('Shooting at arrows?')
    # self.pause = True
    self.control.shoot()
    self.shootcd = 600


  def get_entities(self) -> List[Tuple[str, object]]:
    result = []
    if self.target:
      result.append(('*target', self.target, self.pathToTarget))
    result.extend([(e.type, e) for e in self.entities])
    return result