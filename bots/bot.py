import numpy as np

from threading import Lock
from multiprocessing import shared_memory

from common import WIDTH, HEIGHT
from numpy.typing import NDArray

_CHANNELS = 4
_SCREEN_MEMORY_NAME = 'towerfallScreen'

class Bot():
  def __init__(self):
    self.update_lock = Lock()
    self.shm = shared_memory.SharedMemory(name=_SCREEN_MEMORY_NAME)
    self.is_paused: bool = False

  def stop(self):
    pass

  def grid(self):
    raise NotImplementedError()

  def reset(self):
    pass

  def update(self):
    pass

  def get_entities(self):
    return []

  def get_game_screen(self) -> NDArray[np.uint8]:
    self.update_lock.acquire()
    screen_data = np.frombuffer(self.shm.buf, dtype=np.uint8)
    self.update_lock.release()
    screen_data = screen_data.reshape(HEIGHT, WIDTH, _CHANNELS)
    return screen_data
