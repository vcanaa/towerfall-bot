import json
import logging

from abc import ABC, abstractmethod

from common import Connection, Entity, to_entities

from .actions import TowerfallActions
from .connection_provider import TowerfallProcess

from typing import List, Optional, Tuple, Any
from numpy.typing import NDArray

from gym import Env


class TowerfallEnv(Env, ABC):
  '''
  Interacts with the Towerfall.exe process to create an interface with the agent that follows the gym API.
  Inherit from this class to choose the appropriate observations and reward functions.

  param connection: The connection with the game.
  param actions: The actions that the agent can take. If None, the default actions are used.
  '''
  def __init__(self,
      towerfall: TowerfallProcess,
      actions: Optional[TowerfallActions] = None,
      verbose: int = 0):
    print('Initializing TowerfallEnv')
    self.towerfall = towerfall
    self.verbose = verbose
    self.connection = self.towerfall.join(timeout=5, verbose=self.verbose)
    if actions:
      self.actions = actions
    else:
      self.actions = TowerfallActions()
    self.action_space = self.actions.action_space
    self._draw_elems = []
    self.is_init_sent = False
    print('Initialized TowerfallEnv')

  def _is_reset_valid(self) -> bool:
    '''
    Use this to make check if the initiallization is valid.
    This is useful to collect information about the environment to programmatically construct a sequence of tasks, then
    reset the environment again with the proper reseet instructions.

    Returns:
      True if the reset is valid, False otherwise, then the environment will be reset again.
    '''
    return True

  def _send_reset(self):
    '''
    Sends the reset instruction to the game. Overwrite this to change the starting conditions.
    Returns:
      True if hard reset, False if soft reset.
    '''
    self.towerfall.send_reset(verbose=self.verbose)

  @abstractmethod
  def _post_reset(self) -> Tuple[NDArray, dict]:
    '''
    Hook for a gym reset call. Subclass should populate and return the same as a reset in gym API.

    Returns:
      A tuple of (observation, info)
    '''
    raise NotImplementedError

  @abstractmethod
  def _post_step(self) -> Tuple[NDArray, float, bool, dict]:
    '''
    Hook for a gym step call. Subclass should populate this to return the same as a step in gym API.

    Returns:
      A tuple of (observation, reward, done, info)
    '''
    raise NotImplementedError

  def draws(self, draw_elem):
    '''
    Draws an element on the screen. This is useful for debugging.
    '''
    self._draw_elems.append(draw_elem)

  def reset(self) -> Tuple[NDArray, dict]:
    '''
    Gym reset. This is called by the agent to reset the environment.
    '''
    logging.info('Resetting environment')
    while True:
      self._send_reset()
      if not self.is_init_sent:
        state_init = self.connection.read_json()
        assert state_init['type'] == 'init', state_init['type']
        self.index = state_init['index']
        self.connection.write_json(dict(type='result', success=True))

        self.state_scenario = self.connection.read_json()
        assert self.state_scenario['type'] == 'scenario', self.state_scenario['type']
        self.connection.write_json(dict(type='result', success=True))
        self.is_init_sent = True
      else:
        self.connection.write_json(dict(type='commands', command="", id=self.state_update['id']))

      self.frame = 0
      self.state_update = self.connection.read_json()
      assert self.state_update['type'] == 'update', self.state_update['type']
      self.entities = to_entities(self.state_update['entities'])
      self.me = self._get_own_archer(self.entities)
      if self._is_reset_valid():
        break

    return self._post_reset()

  def step(self, actions: NDArray) -> Tuple[NDArray, float, bool, object]:
    '''
    Gym step. This is called by the agent to take an action in the environment.
    '''
    command = self.actions._actions_to_command(actions)

    resp: dict[str, Any] = dict(
      type='commands',
      command=command,
      id=self.state_update['id']
    )
    # if self._draw_elems:
    #   resp['draws'] = self._draw_elems
    self.connection.write_json(resp)
    self._draw_elems.clear()
    self.state_update = self.connection.read_json()
    assert self.state_update['type'] == 'update'
    self.entities = to_entities(self.state_update['entities'])
    self.me = self._get_own_archer(self.entities)
    self.command = command
    return self._post_step()

  def _get_own_archer(self, entities: List[Entity]) -> Optional[Entity]:
    '''
    Iterates over all entities to find the archer that matches the index specified in init.
    '''
    for e in entities:
      if e.type == 'archer':
        if e['playerIndex'] == self.index:
          return e
    return None

  # def _read_game_state(self):
  #   # logging.info('Reading game state')
  #   self.connection.read_json()
  #   return json.loads(self.connection.read())

  def render(self, mode='human'):
    '''
    This is a no-op since the game is rendered independenly by MonoGame/XNA.
    '''
    pass