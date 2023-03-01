from threading import Semaphore
from common import log

import numpy as np

from numpy.typing import NDArray

from gym import spaces


class EnvWrap:
  def __init__(self):
    # self.num_envs = 1
    self.step_sem = Semaphore(0)
    self.update_sem = Semaphore(1)
    self.actions_sem = Semaphore(0)
    self.obs: NDArray
    self.rew: float
    self.actions: NDArray
    self.unwrapped = self
    self.observation_spaces = spaces.Box(low=0, high=1, shape=(50, ), dtype=np.uint8)


  def reset(self):
    raise NotImplementedError('reset is not implemented')


  def step(self, actions: NDArray):
    try:
      self.step_sem.acquire()
      log('step')
      self.actions = actions
      return self.obs, self.rew, self.done, None
    finally:
      self.actions_sem.release()
      self.update_sem.release()


  def get_actions(self) -> NDArray:
    self.actions_sem.acquire()
    return self.actions


  def update(self, obs: NDArray, rew: float, done: bool):
    try:
      self.update_sem.acquire()
      log('update')
      self.obs = obs
      self.rew = rew
      self.done = done
    finally:
      self.step_sem.release()

