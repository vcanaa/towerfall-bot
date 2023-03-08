import sys
import json

from math import atan2, pi
from threading import Thread, Lock

from pyjoystick.sdl2 import Key, Joystick, run_event_loop

from common import reply, Vec2, log

from .connection import Connection

from typing import Optional


pi8 = pi / 8


def _get_command(command_set: set[str]):
  s = ''
  # print(command_set)
  for c in command_set:
    s += c
  return s


KEYMAP = {
  0: 'j',
  2: 's',
  3: 'c',
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
    self.lock = Lock()
    self.is_human = is_human

    if is_human:
      self.is_alive = True

      def alive():
        return self.is_alive

      def print_add(joy: Joystick):
        log('Added {}'.format(joy))

      def print_remove(joy: Joystick):
        log('Removed {}'.format(joy))

      def key_received(key: Key):
        self.lock.acquire()
        try:
          if key.keytype == Key.AXIS:
            return
          log('Key: {}'.format(key.value))
          if key.number not in KEYMAP:
            return
          keynum = KEYMAP[key.number]
          if key.value:
            self.curr.add(keynum)
          else:
            if keynum not in self.curr:
              return
            self.curr.remove(keynum)
        finally:
          self.lock.release()

      def listen_joystick():
        log('Listening joystick')
        run_event_loop(print_add, print_remove, key_received, alive)

      self.thr = Thread(target=listen_joystick).start()


  def stop(self):
    if hasattr(self, 'is_alive'):
      log('stop Controls')
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

  def _swap(self):
    aux = self.curr
    self.curr = self.past
    self.past = aux

  def _msg(self) -> str:
    msg = json.dumps({
      'type': 'command',
      'command': _get_command(self.curr)
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
      self.lock.release()
