import json
import logging
import numpy as np
import matplotlib.pyplot as plt

from common import Connection, Entity, to_entities, fill_grid, rand_double_region, grid_pos, Vec2, WIDTH, HEIGHT

from typing import List
from numpy.typing import NDArray

from gym import spaces, Env

from stable_baselines3.common.env_checker import check_env

from .grid import GridView

BUTTONS = ['u', 'd', 'l', 'r', 'j', 'z']


class EnvWrap(Env):
  def __init__(self, grid_factor: int, sight: int, connection: Connection):
    self.gv = GridView(grid_factor)
    self.sight = sight
    n = self.gv.view_length(sight)
    self.obs: dict[str,object]
    self.rew: float
    # self.action_space = spaces.MultiBinary(len(BUTTONS))
    self.action_space = spaces.Box(
      low=np.array([-1, -1, 0, 0], dtype=np.int8),
      high=np.array([1, 1, 1, 1], dtype=np.int8) , shape=(4,), dtype=np.int8)
    self.observation_space = spaces.Dict({
        'grid': spaces.MultiBinary((2*n, 2*n)),
        'target': spaces.Box(low=-2*n, high = 2*n, shape=(2,), dtype=np.int8)
    })
    # print(self.observation_space.sample())
    # print(self.observation_space)
    self.connection = connection
    self.connection.read()


  def reset(self):
    # logging.info('reset')
    self._write_instruction('config', pos={'x': 160, 'y': 80})

    game_state = self._read_game_state()
    # print(game_state['type'])
    assert game_state['type'] == 'init'
    self.index = game_state['index']
    # logging.info('index: {}'.format(self.index))
    self.connection.write('.')
    self.frame = 0

    game_state = self._read_game_state()
    # print(game_state['type'])
    assert game_state['type'] == 'scenario'
    self.gv.set_scenario(game_state)
    # logging.info('grid: {}'.format(self.fixed_grid.shape))
    self.connection.write('.')

    game_state = self._read_game_state()
    # print(game_state['type'])
    assert game_state['type'] == 'update'
    self._handle_update(game_state)
    # logging.info('entities: {}'.format(len(self.entities)))
    self._set_new_target()
    displ = self._get_target_displ()
    self.obs_target: NDArray = np.array([displ.x, displ.y], dtype=np.int8)
    # print(self.obs_target)
    self.done = False
    return {
      'grid': self.obs_grid,
      'target': self.obs_target
    }


  def step(self, actions: NDArray):
    # logging.info('step')
    command = ''
    if actions[0] == -1:
      command += 'l'
    elif actions[0] == 1:
      command += 'r'
    if actions[1] == -1:
      command += 'd'
    elif actions[1] == 1:
      command += 'u'
    if actions[2] == 1:
      command += 'j'
    if actions[3] == 1:
      command += 'z'

    self.connection.write(json.dumps({
      'type': 'command',
      'command': command
    }))
    game_state = self._read_game_state()
    self._handle_update(game_state)
    self._update_reward()
    return {
      'grid': self.obs_grid,
      'target': self.obs_target
    }, self.rew, self.done, {}


  def render(self, mode='human'):
    # logging.info('render')
    pass


  def _update_me(self, entities: List[Entity]):
    for e in entities:
      if e.type == 'archer':
        if e['playerIndex'] == self.index:
          self.me: Entity = e


  def _read_game_state(self):
    return json.loads(self.connection.read())


  def _write_instruction(self, type, command='', pos=None):
    resp = {
      'type': type,
      'command': command
    }
    if pos:
      resp['pos'] = pos
    self.connection.write(json.dumps(resp))


  def _update_obs_grid(self):
    self.gv.update(self.entities, self.me)
    self.obs_grid = self.gv.view(self.sight)


  def _set_new_target(self):
    while True:
      x = self.me.p.x + rand_double_region(0.5*self.sight, self.sight)
      y = self.me.p.y + rand_double_region(0.5*self.sight, self.sight)
      # logging.info('(x, y): ({} {})'.format(x, y))
      i, j = grid_pos(Vec2(x, y), self.gv.csize)
      # logging.info('(i, j): ({} {})'.format(i, j))
      if not self.gv.fixed_grid[i][j]:
        break
    self.target = Entity(e = {
      'pos': {'x': x, 'y': y},
      'vel': {'x': 0, 'y': 0},
      'size':{'x': 5, 'y': 5},
      'isEnemy': False,
      'type': 'fake'
    })
    # New target is only used in the next loop.
    self.prev_disp_len = self._get_target_displ().length()
    self.bonus_rew = self.prev_disp_len


  def _get_target_displ(self):
    '''Gets the displacement of of the target from the player.'''
    displ = self.target.p.copy()
    displ.sub(self.me.p)
    return displ


  def _update_reward(self):
    displ = self._get_target_displ()
    disp_len = displ.length()
    self.rew = self.prev_disp_len - disp_len
    if disp_len < 2:
      # Reached target. Gets big reward
      self.rew += self.bonus_rew
      self.done = True
      print('Done. Reached target.')
    if abs(displ.x) > self.sight or abs(displ.y) > self.sight:
      # Target considered out of reach. Fail
      self.done = True
      print('Done. Target out of reach. {} {} {}'.format(self.target.p, self.me.p, displ))
    self.prev_disp_len = disp_len
    self.obs_target: NDArray = np.array([displ.x, displ.y], dtype=np.int8)
    if self.done:
      self._set_new_target()


  def _handle_update(self, state: dict):
    self.entities = to_entities(state['entities'])
    self._update_me(self.entities)
    self._update_obs_grid()
