import json
import logging
from math import atan2, pi
from threading import Lock, Thread
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pyjoystick.sdl2 import Joystick, Key, run_event_loop

from common import Vec2, reply
from towerfall import Connection

pi8 = pi / 8


def get_command(command_set: set[str]):
  s = ''
  # print(command_set)
  for c in command_set:
    s += c
  return s


KEYMAP = {
  0: 'j',
  2: 's',
  3: 'c',
  # 4 share
  # 4 options
  10: 'z',
  11: 'u',
  12: 'd',
  13: 'l',
  14: 'r'
}



class Controls:
  def __init__(self, is_human: bool = False):
    self.past: set[str] = set()
    self.curr: set[str] = set()
    self.key_up_event: set[int] = set()
    self.pressed_keys: set[int] = set()
    self.lock = Lock()
    self.freeze_lock = Lock()
    self.is_human = is_human

    if is_human:
      self.is_alive = True

      def alive():
        return self.is_alive

      def print_add(joy: Joystick):
        logging.info('Joystick added {}'.format(joy))

      def print_remove(joy: Joystick):
        logging.info('Joystick removed {}'.format(joy))

      def key_received(key: Key):
        self.freeze_lock.acquire()
        self.lock.acquire()
        try:
          if key.keytype == Key.AXIS:
            return
          # logging.info('Key: %s %s', key.number, key.value)
          if key.value:
            self.pressed_keys.add(key.number)
            if key.number in KEYMAP:
              self.curr.add(KEYMAP[key.number])
          else:
            if key.number in self.pressed_keys:
              self.key_up_event.add(key.number)
            self.pressed_keys.discard(key.number)
            if key.number in KEYMAP:
              self.curr.discard(KEYMAP[key.number])

        finally:
          self.freeze_lock.release()
          self.lock.release()

      def listen_joystick():
        logging.info('Listening joystick')
        run_event_loop(print_add, print_remove, key_received, alive)

      self.thr = Thread(target=listen_joystick).start()


  def consume_key_up_event(self, key_number: int) -> bool:
    r = False
    if key_number in self.key_up_event:
      r = True
      self.key_up_event.remove(key_number)
    return r

  def parse_command(self, command: str):
    self._swap()
    self.curr.clear()
    for c in command:
      self.curr.add(c)


  def freeze(self):
    self.freeze_lock.acquire()


  def stop(self):
    if hasattr(self, 'is_alive'):
      logging.info('stop Controls')
      self.is_alive = False


  def try_press(self, c):
    if c in self.past:
      return
    self.curr.add(c)

  def jump(self):
    self.try_press('j')

  def dash(self):
    self.try_press('z')

  def shoot(self):
    self.try_press('s')

  def right(self):
    self.curr.add('r')

  def left(self):
    self.curr.add('l')

  def up(self):
    self.curr.add('u')

  def down(self):
    self.curr.add('d')

  def direction(self) -> NDArray:
    return np.array([self.hor(), self.ver()])

  def hor(self) -> float:
    r = 0
    if 'l' in self.curr:
      r -= 1
    if 'r' in self.curr:
      r += 1
    return r

  def ver(self) -> float:
    r = 0
    if 'd' in self.curr:
      r -= 1
    if 'u' in self.curr:
      r += 1
    return r

  def key_state(self, k) -> NDArray:
    return np.array([
      1 if k in self.curr else 0,
      1 if k in self.past else 0
    ])

  def jump_state(self) -> NDArray:
    return self.key_state('j')

  def dash_state(self) -> NDArray:
    return self.key_state('z')

  def shoot_state(self) -> NDArray:
    return self.key_state('s')

  def aim(self, dir: Vec2):
    a = atan2(dir.y, dir.x)
    if a < -5*pi8 or a > 5*pi8:
      self.left()
    if a > -7*pi8 and a < -pi8:
      self.down()
    if a > -3*pi8 and a < 3*pi8:
      self.right()
    if a > pi8 and a < 7*pi8:
      self.up()

  def get_command(self):
    return get_command(self.curr)

  def _swap(self):
    aux = self.curr
    self.curr = self.past
    self.past = aux

  def _msg(self) -> str:
    msg = json.dumps({
      'type': 'command',
      'command': get_command(self.curr)
    })
    return msg

  def reply(self, conn: Optional[Connection] = None):
    self.lock.acquire()
    try:
      if conn:
        conn.write(self._msg())
      else:
        reply(self._msg())

      if not self.is_human:
        self._swap()
        self.curr.clear()
    finally:
      if self.freeze_lock.locked():
        self.freeze_lock.release()
      self.lock.release()
