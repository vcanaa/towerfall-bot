from gym import spaces

from numpy.typing import NDArray

_LEFT = 'l'
_RIGHT = 'r'
_DOWN = 'd'
_UP = 'u'
_JUMP = 'j'
_DASH = 'z'
_SHOOT = 's'

class TowerfallActions:
  '''Does conversion between gym actions and Towerfall API actions.
  Allows enabling or disabling certain actions'''
  def __init__(self,
      can_jump=True,
      can_dash=True,
      can_shoot=True):

    self.can_jump = can_jump
    self.can_dash = can_dash
    self.can_shoot = can_shoot

    actions = [3,3]
    self.action_map = {} # maps key to index
    if can_jump:
      self.action_map[_JUMP] = len(actions)
      actions.append(2)
    if can_dash:
      self.action_map[_DASH] = len(actions)
      actions.append(2)
    if can_shoot:
      self.action_map[_SHOOT] = len(actions)
      actions.append(2)
    self.action_space = spaces.MultiDiscrete(actions)

  def _actions_to_command(self, actions: NDArray) -> str:
    command = ''
    if actions[0] == 0:
      command += _LEFT
    elif actions[0] == 2:
      command += _RIGHT
    if actions[1] == 0:
      command += _DOWN
    elif actions[1] == 2:
      command += _UP

    if self.can_jump and actions[self.action_map[_JUMP]]:
      command += _JUMP
    if self.can_dash and actions[self.action_map[_DASH]]:
      command += _DASH
    if self.can_shoot and actions[self.action_map[_SHOOT]]:
      command += _SHOOT
    return command