from gym import spaces

from common.constants import *

from numpy.typing import NDArray

class TowerfallActions:
  '''
  Does conversion between gym actions and Towerfall API actions.
  Allows enabling or disabling certain actions.
  '''
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
      self.action_map[JUMP] = len(actions)
      actions.append(2)
    if can_dash:
      self.action_map[DASH] = len(actions)
      actions.append(2)
    if can_shoot:
      self.action_map[SHOOT] = len(actions)
      actions.append(2)
    self.action_space = spaces.MultiDiscrete(actions)

  def _actions_to_command(self, actions: NDArray) -> str:
    '''
    Converts a list of actions to a command string.
    '''
    command = ''
    if actions[0] == 0:
      command += LEFT
    elif actions[0] == 2:
      command += RIGHT
    if actions[1] == 0:
      command += DOWN
    elif actions[1] == 2:
      command += UP

    if self.can_jump and actions[self.action_map[JUMP]]:
      command += JUMP
    if self.can_dash and actions[self.action_map[DASH]]:
      command += DASH
    if self.can_shoot and actions[self.action_map[SHOOT]]:
      command += SHOOT
    return command