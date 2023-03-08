import json
import random

from multiprocessing import shared_memory

import numpy as np

from numpy.typing import NDArray
from threading import Lock


from math import cos, sin

from common import *
from .trainer import *

from typing import Any, List, Optional, Tuple

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 9000  # The port used by the server

class QuestBotRL:
  def __init__(self):
    self.stateInit: Any
    self.stateUpdate: Any
    self.entities: List[Entity]
    self.widget_update_cd: int = 0
    self.update_lock = Lock()
    self.shm = shared_memory.SharedMemory(name='towerfallScreen')
    self.pause = False
    self.connection = Connection('127.0.0.1', 9000)
    self.gv = GridView(1)
    self.control: Controls = Controls(is_human=True)
    self.should_reset = False


  def __del__(self):
    logging.info('del bot')
    if hasattr(self, 'connection'):
      del self.connection


  def run(self):
    try:
      self.connection.read()
      self.connection.write_instruction('config')
      while True:
        self.update()
    finally:
      self.control.stop()


  def reset(self):
    self.should_reset = True


  def stop(self):
    self.control.stop()


  def update(self):
    gameState = json.loads(self.connection.read())
    if self.should_reset:
      self.connection.write_instruction('config')
      self.should_reset = False
      return

    if gameState['type'] == 'init':
      self.handleInit(gameState)

    if gameState['type'] == 'scenario':
      self.handleScenario(gameState)

    if gameState['type'] == 'update':
      self.handleUpdate(gameState)


  def handleInit(self, state: dict):
    logging.info("handleInit")
    self.stateInit = state
    random.seed(state['index'])
    self.connection.write('.')


  def handleScenario(self, state: dict):
    logging.info("handleScenario")
    self.stateScenario = state
    self.gv.set_scenario(state)
    self.connection.write('.')


  def getPlayer(self, entities: List[Entity]):
    for e in entities:
      if e.type == 'archer':
        if e['playerIndex'] == self.stateInit['index']:
          self.me: Entity = e
          return e


  def handleUpdate(self, state):
    self.update_lock.acquire()
    try:
      self.entities: List[Entity] = to_entities(state['entities'])
      self.getPlayer(self.entities)
      self.gv.update(self.entities, self.me)



      if self.me == None:
        raise Exception('no me')

      self.control.reply(self.connection)
    finally:
      self.update_lock.release()


  def get_game_screen(self) -> NDArray[np.uint8]:
    self.update_lock.acquire()
    screen_data = np.frombuffer(self.shm.buf, dtype=np.uint8)
    self.update_lock.release()
    screen_data = screen_data.reshape(240, 320, 4)
    return screen_data


  def get_entities(self) -> List[Tuple[str, object]]:
    result = []
    result.extend([(e.type, e) for e in self.entities])
    return result


  def grid(self):
    return self.gv.fixed_grid10
