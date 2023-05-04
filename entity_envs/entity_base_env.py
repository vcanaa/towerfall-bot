from abc import abstractmethod
import logging
from typing import Any, List, Mapping, Optional, Dict

from entity_gym.env import GlobalCategoricalAction, Environment, Observation, Action, ActionName, ObsSpace, ActionSpace, GlobalCategoricalActionSpace
from entity_gym.env import Entity as EntityGym

from common.constants import DASH, DOWN, JUMP, LEFT, RIGHT, SHOOT, UP
from common.entity import Entity, to_entities

from envs.connection_provider import TowerfallProcess

class TowerfallEntityEnv(Environment):
  def __init__(self,
      towerfall: TowerfallProcess,
      record_path: Optional[str] = None,
      verbose: int = 0):
    logging.info('Initializing TowerfallEntityEnv')
    self.towerfall = towerfall
    self.verbose = verbose
    self.connection = self.towerfall.join(timeout=5, verbose=self.verbose)
    self.connection.record_path = record_path
    self._draw_elems = []
    self.is_init_sent = False

    logging.info('Initialized TowerfallEnv')

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
  def _post_reset(self) -> Observation:
    '''
    Hook for a gym reset call. Subclass should populate and return the same as a reset in gym API.

    Returns:
      A tuple of (observation, info)
    '''
    raise NotImplementedError

  @abstractmethod
  def _post_observe(self) -> Observation:
    raise NotImplementedError

  def draws(self, draw_elem):
    '''
    Draws an element on the screen. This is useful for debugging.
    '''
    self._draw_elems.append(draw_elem)

  def reset(self) -> Observation:
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

  @classmethod
  def obs_space(cls) -> ObsSpace:
    return ObsSpace(
      global_features=["y", "arrow_count"],
      entities={
        "enemies": EntityGym(features=["x", "y", "facing"]),
        "arrows": EntityGym(features=["x", "y", "stuck"]),
    })

  @classmethod
  def action_space(cls) -> Dict[ActionName, ActionSpace]:
    return {
        "hor": GlobalCategoricalActionSpace(
            [".", "l", "r"],
        ),
        "ver": GlobalCategoricalActionSpace(
            [".", "d", "u"],
        ),
        "shoot": GlobalCategoricalActionSpace(
            [".", "s"],
        ),
        "dash": GlobalCategoricalActionSpace(
            [".", "d"],
        ),
        "jump": GlobalCategoricalActionSpace(
            [".", "j"],
        ),
    }

  def observe(self) -> Observation:
    self.state_update = self.connection.read_json()
    assert self.state_update['type'] == 'update'
    self.entities = to_entities(self.state_update['entities'])
    self.me = self._get_own_archer(self.entities)
    obs = self._post_observe()
    # logging.info(f'Observation: {obs.__dict__}')
    return obs

  def _actions_to_command(self, actions: Mapping[ActionName, Action]) -> str:
    command = ''
    hor: Action = actions['hor']
    assert isinstance(hor, GlobalCategoricalAction)
    if hor.index == 1:
      command += LEFT
    elif hor.index == 2:
      command += RIGHT

    ver: Action = actions['hor']
    assert isinstance(ver, GlobalCategoricalAction)
    if ver.index == 1:
      command += DOWN
    elif ver.index == 2:
      command += UP

    jump: Action = actions['jump']
    assert isinstance(jump, GlobalCategoricalAction)
    if jump.index == 1:
      command += JUMP

    dash: Action = actions['dash']
    assert isinstance(dash, GlobalCategoricalAction)
    if dash.index == 1:
      command += DASH

    shoot: Action = actions['shoot']
    assert isinstance(shoot, GlobalCategoricalAction)
    if shoot.index == 1:
      command += SHOOT
    return command

  def act(self, actions: Mapping[ActionName, Action]) -> Observation:
    command = self._actions_to_command(actions)

    resp: dict[str, Any] = dict(
      type='commands',
      command=command,
      id=self.state_update['id']
    )
    if self._draw_elems:
      resp['draws'] = self._draw_elems
    self.connection.write_json(resp)
    self._draw_elems.clear()

    return self.observe()

  def _get_own_archer(self, entities: List[Entity]) -> Optional[Entity]:
    for e in entities:
      if e.type == 'archer':
        if e['playerIndex'] == self.index:
          return e
    return None