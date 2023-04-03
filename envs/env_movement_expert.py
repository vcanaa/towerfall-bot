import logging
import numpy as np
import random

from gym import spaces

from common import Connection, GridView, Vec2, Entity, to_entities, rand_double_region, grid_pos, WIDTH, HEIGHT

from envs import TowerFallEnvWrapper
from .actions import TowerfallActions

from numpy.typing import NDArray
from typing import Tuple, Optional

HW = WIDTH // 2
HH = HEIGHT // 2

class TowerfallMovementExpertEnv(TowerFallEnvWrapper):
  '''In each episode of this environment, the agent has to move from point A to point B.
  For every frame positive reward is given for getting closer to target and negative reward is given for distantiating from target.
  When reaching the target, agent receives a bigger bounty.'''
  def __init__(self, connection: Connection, actions: Optional[TowerfallActions] = None, grid_factor: int = 2, bounty=50, episode_max_len: int=60*5):
    super(TowerfallMovementExpertEnv, self).__init__(connection, actions)
    self.gv = GridView(grid_factor)
    m, n = self.gv.view_sight_length(None)
    self.episode_max_len = episode_max_len
    self.bounty = bounty
    # logging.info('m, n: %s, %s', m, n)
    self.obs: dict[str,object]
    self.rew: float
    self.draws = []
    self.observation_space = spaces.Dict({
        'dodgeCooldown': spaces.Discrete(2),
        'dodging': spaces.Discrete(2),
        'facing': spaces.Discrete(2),
        'grid': spaces.MultiBinary((2*m, 2*n)),
        'onGround': spaces.Discrete(2),
        'onWall': spaces.Discrete(2),
        'target': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        'vel': spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float32),
    })
    logging.info(str(self.observation_space))

  def _handle_reset(self, state_scenario: dict, state_update: dict) -> dict:
    self.gv.set_scenario(state_scenario)
    self.entities = to_entities(state_update['entities'])
    self.me = self._get_own_archer(self.entities)
    self._update_obs_grid()

    self._set_new_target()
    displ = self._get_target_displ()
    self.obs_target: NDArray = np.array([displ.x / HW, displ.y / HH], dtype=np.int8)
    self.done = False
    self.episode_len = 0
    return self._get_obs()

  def _handle_step(self, state_update: dict) -> Tuple[object, float, bool, object]:
    self.entities = to_entities(state_update['entities'])
    self.me = self._get_own_archer(self.entities)
    self._update_obs_grid()
    self._update_reward()
    self.draws.clear()
    self.episode_len += 1
    assert self.me
    self.draws.append({
      'type': 'line',
      'start': self.me['pos'],
      'end': self.target['pos'],
      'color': [1,1,1],
      'thick': 4
    })

    return self._get_obs(), self.rew, self.done, {}

  def _get_obs(self):
    assert self.me
    return {
      'dodgeCooldown': int(self.me['dodgeCooldown']),
      'dodging': int(self.me['state']=='dodging'),
      'facing': (self.me['facing'] + 1) // 2, # -1,1 -> 0,1
      'grid': self.obs_grid,
      'onGround': int(self.me['onGround']),
      'onWall': int(self.me['onWall']),
      'target': self.obs_target,
      'vel': self.me.v.array() / 5
    }

  def _update_reward(self):
    assert self.me
    displ = self._get_target_displ()
    disp_len = displ.length()
    self.rew = self.prev_disp_len - disp_len
    if disp_len < self.me.s.y / 2:
      # Reached target. Gets big reward
      self.rew += self.bounty
      self.done = True
      logging.info('Done. Reached target.')
    if self.episode_len > self.episode_max_len:
      self.done = True
      logging.info('Done. Timeout.')
    self.prev_disp_len = disp_len
    self.obs_target: NDArray = np.array([displ.x / HW, displ.y / HH], dtype=np.float32)
    if self.done:
      self._set_new_target()

  def _update_obs_grid(self):
    assert self.me
    self.gv.update(self.entities, self.me)
    self.obs_grid = self.gv.view(None)

  def _set_new_target(self):
    assert self.me
    while True:
      x = random.randint(0, WIDTH)
      y = random.randint(0, HEIGHT)
      i, j = grid_pos(Vec2(x, y), self.gv.csize)
      # logging.info('(i, j): ({} {})'.format(i, j))
      if not self.gv.fixed_grid10[i][j]:
        break
    # logging.info('New target: (x, y): ({} {})'.format(x, y))
    self.target = Entity(e = {
      'pos': {'x': x, 'y': y},
      'vel': {'x': 0, 'y': 0},
      'size':{'x': 5, 'y': 5},
      'isEnemy': False,
      'type': 'fake'
    })
    # New target is only used in the next loop.
    self.prev_disp_len = self._get_target_displ().length()

  def _get_target_displ(self):
    '''Gets the displacement of of the target from the player.'''
    assert self.me
    displ = self.target.p.copy()
    displ.sub(self.me.p)
    return displ

  def _get_draws(self) -> list[dict]:
    return self.draws