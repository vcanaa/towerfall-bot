import sys
sys.path.insert(0, 'C:/Program Files (x86)/Steam/steamapps/common/TowerFall/aimod')

import time
import timeit
import logging
import random

from common import Connection
from envs import TowerfallProcessProvider, TowerfallProcess

from typing import Any

_VERBOSE = 0
_TIMEOUT = 4

class NoLevelFormatter(logging.Formatter):
  def format(self, record):
    return record.getMessage()

logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(NoLevelFormatter())


process_provider = TowerfallProcessProvider('test')

_starting_x = [64, 256, 128, 192]

def get_random_command() -> str:
  s = ''
  p = 0.1
  keys = ['u', 'd', 'l', 'r', 'j', 'z', 's']
  for key in keys:
    if random.random() < p:
      s += key
  return s


def get_config(agent_count: int) -> dict[str, Any]:
  return dict(
    mode='sandbox',
    level='1',
    fps=30,
    agents=[dict(type='remote', team='blue', archer='green')]*agent_count)


def get_process(agent_count: int) -> TowerfallProcess:
  return process_provider.get_process(
    fastrun=True,
    # nographics=True,
    config=get_config(agent_count))


def join(towerfall: TowerfallProcess, agent_count: int) -> list[Connection]:
  connections = []
  for i in range(agent_count):
    conn = towerfall.join(timeout=_TIMEOUT, verbose=_VERBOSE)
    conn.log_cap = 100
    connections.append(conn)
  return connections


def reset(towerfall: TowerfallProcess, agent_count: int) -> list[dict]:
  y = 110
  entities = [dict(type='archer', pos=dict(x=_starting_x[i], y=y)) for i in range(agent_count)]
  entities.append(dict(type='slime', pos=dict(x=160, y=y)))
  towerfall.send_reset(entities, verbose=_VERBOSE)
  return entities


def receive_init(connections: list[Connection]):
  for i in range(len(connections)):
    # init
    state_init = connections[i].read_json()
    assert state_init['type'] == 'init', state_init['type']
    connections[i].write_json(dict(type='result', success=True))

  for i in range(len(connections)):
    # scenario
    state_scenario = connections[i].read_json()
    assert state_scenario['type'] == 'scenario', state_scenario['type']
    connections[i].write_json(dict(type='result', success=True))


def receive_update(connections: list[Connection], entities: list[dict], length: int):
  now = time.time()
  for j in range(length):
    for i in range(len(connections)):
      # update
      state_update = connections[i].read_json()
      assert state_update['type'] == 'update', state_update['type']
      # if j == 0:
      #   pos = [e['pos'] for e in state_update['entities'] if e['type'] == 'archer' and e['playerIndex']==i][0]
      #   diff = abs(pos['x'] - entities[i]['pos']['x'])
      #   assert diff < 2, f"{pos['x']} != {entities[i]['pos']['x']}, diff = {diff}"
      connections[i].write_json(dict(type='commands', command=get_random_command(), id=state_update['id']))
  dt = time.time() - now
  logging.info(f'fps: {length/dt:.2f}')


def run_many_resets(towerfall: TowerfallProcess, agent_count: int, reset_count: int):
  connections = join(towerfall, agent_count)
  entities = reset(towerfall, agent_count)

  receive_init(connections)
  receive_update(connections, entities, length=20)

  for i in range(reset_count):
    entities = reset(towerfall, agent_count)
    receive_update(connections, entities, length=20)


def run_session():
  agent_count = 1
  reset_count = 5
  towerfall = get_process(agent_count)
  run_many_resets(towerfall, agent_count, reset_count)

  # agent_count = 1
  # towerfall.send_config(get_config(agent_count), verbose=_VERBOSE)
  # run_many_resets(towerfall, agent_count, reset_count)

  # agent_count = 3
  # towerfall.send_config(get_config(agent_count), verbose=_VERBOSE)
  # run_many_resets(towerfall, agent_count, reset_count)

  # agent_count = 4
  # towerfall.send_config(get_config(agent_count), verbose=_VERBOSE)
  # run_many_resets(towerfall, agent_count, reset_count)

  # process_provider.release_process(towerfall)


n_it = 1
elapsed_time = timeit.timeit(run_session, number=n_it) / n_it
print(f'Elapsed time: {elapsed_time:.2f} s')

# process_provider.close()