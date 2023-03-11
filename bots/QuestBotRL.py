import os
import json
import random
import time

from multiprocessing import shared_memory
from pathlib import Path

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
    self.replay: GameReplay


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
    game_state = json.loads(self.connection.read())
    if self.should_reset:
      self.connection.write_instruction('config')
      self.should_reset = False
      return

    if game_state['type'] == 'init':
      self.handle_init(game_state)

    if game_state['type'] == 'scenario':
      self.handle_scenario(game_state)

    if game_state['type'] == 'update':
      self.handle_update(game_state)


  def handle_init(self, state: dict):
    logging.info("handle_init")
    if hasattr(self, 'replay'):
      dir_path = Path(os.path.join('replays', 'only_side_moves'))
      dir_path.mkdir(parents=True, exist_ok=True)
      file_name: str = '{}.json'.format(time.time_ns()//100000000)
      self.replay.save(os.path.join(dir_path, file_name))
    self.replay = GameReplay()
    self.replay.handle_init(state)
    self.stateInit = state
    random.seed(state['index'])
    self.connection.write('.')


  def handle_scenario(self, state: dict):
    logging.info("handle_scenario")
    self.replay.handle_scenario(state)
    self.stateScenario = state
    self.gv.set_scenario(state)
    self.connection.write('.')


  def getPlayer(self, entities: List[Entity]):
    for e in entities:
      if e.type == 'archer':
        if e['playerIndex'] == self.stateInit['index']:
          self.me: Entity = e
          return e


  def handle_update(self, state):
    self.update_lock.acquire()
    try:
      self.replay.handle_update(state)
      self.entities: List[Entity] = to_entities(state['entities'])
      self.getPlayer(self.entities)
      self.gv.update(self.entities, self.me)

      if 6 in self.control.pressed_keys:
        self.connection.write_instruction('config')
        self.should_reset = False
        return

      self.control.freeze()
      # self.create_input()

      self.replay.handle_actions(self.control.get_command())

      if self.me == None:
        raise Exception('no me')

      self.control.reply(self.connection)
    finally:
      self.update_lock.release()


  def create_input(self):
    m, n = self.me.s.tupleint()
    sight = (m//2 + 1, n//2 + 1)
    immediate_wall = self.gv.view(sight)
    left_wall = immediate_wall[0, :]
    right_wall = immediate_wall[-1, :]
    bot_wall = immediate_wall[:, 0]
    top_wall = immediate_wall[:, -1]

    if hasattr(self, 'input_state'):
      output_state = np.concatenate([
        self.me.v.array()
      ])

    self.input_state = np.concatenate([
      left_wall,
      right_wall,
      bot_wall,
      top_wall,
      self.control.hor(),
      self.control.ver(),
      self.control.jump_state(),
      self.control.dash_state(),
      self.me.v.array()
    ])

    if 4 in self.control.pressed_keys:
      logging.info('sight %s', sight)
      logging.info('input_state %s', self.input_state.shape)
      plot_grid(immediate_wall, 'immediate_wall')
      # plot_grid(top_wall, 'top_wall')
      # plot_grid(bot_wall, 'bot_wall')
      # plot_grid(left_wall, 'left_wall')
      # plot_grid(right_wall, 'right_wall')


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
