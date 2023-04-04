import logging
import numpy as np

from gym import spaces, Wrapper

from common import Connection, GridView, Vec2, Entity, to_entities, rand_double_region, grid_pos

from .gym_wrapper import TowerfallEnv
from .actions import TowerfallActions
from .observations import PlayerObservation

from numpy.typing import NDArray
from typing import Tuple, Optional

class TowerfallMovementEnv(TowerfallEnv):
  def __init__(self,
      connection: Connection,
      actions: Optional[TowerfallActions] = None,
      grid_factor: int = 2,
      sight: int = 50):
    """A gym environment for TowerFall Ascension.
    Args:
    TODO: """

    super(TowerfallMovementEnv, self).__init__(connection, actions)
    self.gv = GridView(grid_factor)
    self.sight = sight
    n, _ = self.gv.view_sight_length(sight)
    self.obs: dict[str,object]
    self.rew: float
    obs_space = {
        'grid': spaces.MultiBinary((2*n, 2*n)),
        'target': spaces.Box(low=-2*n, high = 2*n, shape=(2,), dtype=np.int8)
    }
    self.player_obs = PlayerObservation()
    self.player_obs.extend_obs_space(obs_space)
    self.observation_space = spaces.Dict(obs_space)

  def _handle_reset(self, state_scenario: dict) -> dict:
    self.gv.set_scenario(state_scenario)
    self._update_obs_grid()

    self._set_new_target()
    displ = self._get_target_displ()
    self.obs_target: NDArray = np.array([displ.x, displ.y], dtype=np.int8)
    self.done = False
    obs = {
      'grid': self.obs_grid,
      'target': self.obs_target
    }
    assert self.me
    self.player_obs._extend_obs(self.me, obs)
    return obs

  def _handle_step(self) -> Tuple[object, float, bool, object]:
    self._update_obs_grid()
    self._update_reward()
    assert self.me
    self.draws({
      'type': 'line',
      'start': self.me['pos'],
      'end': self.target['pos'],
      'color': [1,1,1],
      'thick': 4
    })

    obs = {
      'grid': self.obs_grid,
      'target': self.obs_target
    }
    self.player_obs._extend_obs(self.me, obs)
    return obs, self.rew, self.done, {}

  def _update_reward(self):
    assert self.me
    displ = self._get_target_displ()
    disp_len = displ.length()
    self.rew = self.prev_disp_len - disp_len
    if disp_len < self.me.s.y / 2:
      # Reached target. Gets big reward
      self.rew += self.bonus_rew
      self.done = True
      logging.debug('Done. Reached target.')
    if abs(displ.x) > self.sight or abs(displ.y) > self.sight:
      # Target considered out of reach. Fail
      self.done = True
      logging.debug('Done. Target out of reach. {} {} {}'.format(self.target.p, self.me.p, displ))
    self.prev_disp_len = disp_len
    self.obs_target: NDArray = np.array([displ.x, displ.y], dtype=np.int8)
    if self.done:
      self._set_new_target()

  def _update_obs_grid(self):
    assert self.me
    self.gv.update(self.entities, self.me)
    self.obs_grid = self.gv.view(self.sight)

  def _set_new_target(self):
    assert self.me
    while True:
      x = self.me.p.x + rand_double_region(0.5*self.sight, self.sight)
      # y = self.me.p.y + rand_double_region(0.5*self.sight, self.sight)
      y = self.me.p.y
      # logging.info('(x, y): ({} {})'.format(x, y))
      i, j = grid_pos(Vec2(x, y), self.gv.csize)
      # logging.info('(i, j): ({} {})'.format(i, j))
      if not self.gv.fixed_grid10[i][j]:
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
    assert self.me
    displ = self.target.p.copy()
    displ.sub(self.me.p)
    return displ
