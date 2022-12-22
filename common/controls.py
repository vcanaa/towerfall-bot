import sys

from math import atan2, pi

from common import reply, Vec2

pi8 = pi / 8

class Controls:
  def __init__(self):
    self.past: set[str] = set()
    self.curr: set[str] = set()

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

  def __print(self):
    for c in self.curr:
      sys.stdout.write(c)

  def __swap(self):
    aux = self.curr
    self.curr = self.past
    self.past = aux

  def reply(self):
    self.__print()
    self.__swap()
    self.curr.clear()
    reply()