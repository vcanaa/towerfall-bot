import json

from typing import Any, Dict, List

class GameReplay:
  def __init__(self):
    self.state_init: Dict[str, Any]
    self.state_scenario: Dict[str, Any]
    self.state_update: List[Dict[str, Any]] = []
    self.actions: List[str] = []

  def handle_init(self, state):
    self.state_init = state

  def handle_scenario(self, state):
    self.state_scenario = state

  def handle_update(self, state):
    self.state_update.append(state)

  def handle_actions(self, actions):
    self.actions.append(actions)

  def save(self, filepath: str):
    with open(filepath, 'w') as stream:
      stream.write(json.dumps(self.state_init))
      stream.write('\n')
      stream.write(json.dumps(self.state_scenario))
      stream.write('\n')
      i = 0
      for state in self.state_update:
        stream.write(json.dumps(state))
        stream.write('\n')
        if i < len(self.actions):
          stream.write(json.dumps(self.actions[i]))
          stream.write('\n')
          i += 1

  def load(self, filepath: str):
    with open(filepath, 'r') as stream:
      self.state_init = json.loads(stream.readline())
      self.state_scenario = json.loads(stream.readline())
      while True:
        line: str = stream.readline()
        if not line:
          break
        self.state_update.append(json.loads(line))
        line = stream.readline()
        if not line:
          break
        self.actions.append(json.loads(line))
