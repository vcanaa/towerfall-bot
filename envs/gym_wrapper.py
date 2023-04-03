import json

from abc import ABC, abstractmethod

from common import Connection, Entity

from .actions import TowerfallActions

from typing import List, Optional, Tuple
from numpy.typing import NDArray

from gym import Env

class TowerFallEnvWrapper(Env, ABC):
  def __init__(self, connection: Connection, actions: Optional[TowerfallActions] = None):
    self.connection = connection
    if actions:
      self.actions = actions
    else:
      self.actions = TowerfallActions()
    self.action_space = self.actions.action_space
    self.connection.read()

  # @abstractmethod
  # def _actions_to_command(self, actions: NDArray) -> str:
  #   '''Converts the action array into the commands complying to Towerfall API.'''
  #   raise NotImplementedError

  @abstractmethod
  def _handle_reset(self, state_scenario: dict, state_update: dict):
    '''Hook for a gym reset call.'''
    raise NotImplementedError

  @abstractmethod
  def _handle_step(self, state_update: dict) -> Tuple[NDArray, float, bool, object]:
    '''Hook for a gym step call.'''
    raise NotImplementedError

  @abstractmethod
  def _get_draws(self) -> list[dict]:
    raise NotImplementedError

  def reset(self) -> Tuple[NDArray, object]:
    '''Gym reset'''
    self.connection.write_instruction('config', pos={'x': 160, 'y': 80})

    state_init = self._read_game_state()
    assert state_init['type'] == 'init'
    self.index = state_init['index']
    self.connection.write('.')
    self.frame = 0

    self.state_scenario = self._read_game_state()
    assert self.state_scenario['type'] == 'scenario'
    self.connection.write('.')

    state_update = self._read_game_state()
    assert state_update['type'] == 'update'
    return self._handle_reset(self.state_scenario, state_update)

  def step(self, actions: NDArray) -> Tuple[NDArray, float, bool, object]:
    '''Gym step'''
    command = self.actions._actions_to_command(actions)

    self.connection.write(json.dumps({
      'type': 'command',
      'command': command,
      'draws': self._get_draws()
    }))
    state_update = self._read_game_state()
    assert state_update['type'] == 'update'
    return self._handle_step(state_update)

  def _get_own_archer(self, entities: List[Entity]) -> Optional[Entity]:
    '''Iterates over all entities to find the archer that matches the index specified in init.'''
    for e in entities:
      if e.type == 'archer':
        if e['playerIndex'] == self.index:
          return e
    return None

  def _read_game_state(self):
    return json.loads(self.connection.read())

  def render(self, mode='human'):
    '''This is a no-op since the game is rendered independenly by MonoGame/XNA'''
    pass