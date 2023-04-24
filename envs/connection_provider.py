import os
import json
import psutil
import signal
import logging
import time

from copy import deepcopy
from subprocess import Popen, PIPE

from common import Connection

from typing import Any, Optional


_HOST = '127.0.0.1'


class TowerfallProcess:
  '''
  Offers an interface with a Towerfall process.

  params pid: The process ID of the Towerfall process.
  params port: The port that the Towerfall process is listening on.
  params config: The current configuration of the Towerfall process.
  '''
  def __init__(self, pid: int, port: int, config: dict[str, Any] = {}):
    self.pid = pid
    self.port = port
    self.config: dict[str, Any] = config
    self.connections: list[Connection] = []

  def to_dict(self) -> dict[str, Any]:
    return dict(
      pid=self.pid,
      port=self.port,
      config=self.config
    )

  def join(self) -> Connection:
    connection = Connection(_HOST, self.port, timeout=2)
    connection.write_json(dict(type='join'))
    resp = connection.read_json()
    if resp['type'] != 'result':
      raise Exception(f'Unexpected response type: {resp["type"]}')
    if not resp['success']:
      raise Exception(f'Failed to join process {self.pid}: {resp["message"]}')
    self.connections.append(connection)

    def on_close():
      self.connections.remove(connection)
    connection.on_close = on_close
    return connection

  def send_reset(self, entities):
    resp = self.send_request_json(dict(type='reset', entities=entities), timeout=2)
    if resp['type'] != 'result':
      raise Exception(f'Unexpected response type: {resp["type"]}')
    if not resp['success']:
      raise Exception(f'Failed to reset process {self.pid}: {resp["message"]}')
    logging.info(f'Successfully reset process {self.pid}')

  def send_config(self, config):
    resp = self.send_request_json(dict(type='config', config=config), timeout=2)
    if resp['type'] != 'result':
      raise Exception(f'Unexpected response type: {resp["type"]}')
    if not resp['success']:
      raise Exception(f'Failed to config process {self.pid}: {resp["message"]}')
    logging.info(f'Successfully applied config to process {self.pid}')
    self.config = config

  def send_request_json(self, obj: dict[str, Any], timeout=0):
    connection = Connection(_HOST, self.port, timeout=timeout)
    connection.write_json(obj)
    return connection.read_json()


class TowerfallProcessProvider:
  '''
  Creates and manages Towerfall processes.

  params name: Name of the connection provider. Used to separate different connection providers states.
  '''
  def __init__(self, name: str):
    self.towerfall_path = 'C:/Program Files (x86)/Steam/steamapps/common/TowerFall'
    self.towerfall_path_exe = os.path.join(self.towerfall_path, 'TowerFall.exe')
    self.connection_path = os.path.join('.connection_provider', name)
    os.makedirs(self.connection_path, exist_ok=True)
    self.state_path = os.path.join(self.connection_path, 'state.json')
    self.processes = []
    if os.path.exists(self.state_path):
      with open(self.state_path, 'r') as file:
        for process_data in json.loads(file.read()):
          try:
            psutil.Process(process_data['pid'])
            self.processes.append(TowerfallProcess(**process_data))
          except psutil.NoSuchProcess:
            continue

    self._processes_in_use = set()

    self._save_state()

    self.default_config = dict(
      mode='sandbox',
      level='2',
      fastrun=False,
      agents=[
        dict(type='remote', team='blue', archer='green')
      ]
    )

  def get_process(self, config = None) -> TowerfallProcess:
    if not config:
      config = self.default_config

    # Try to find an existing process that is not in use
    selected_process = next((p for p in self.processes if p.pid not in self._processes_in_use), None)

    # If no process was found, start a new one
    if not selected_process:
      logging.info(f'Starting new process {self.towerfall_path_exe}')
      process = Popen([self.towerfall_path_exe, '--noconfig'], cwd=self.towerfall_path)
      port = self._get_port(process.pid)
      selected_process = TowerfallProcess(process.pid, port)
      self.processes.append(selected_process)
      self._save_state()

    selected_process.send_config(config)
    self._processes_in_use.add(selected_process.pid)
    self._save_state()
    return selected_process

  def release_process(self, process: TowerfallProcess):
    self._processes_in_use.remove(process.pid)

  def close(self):
    for process in self.processes:
      os.kill(process.pid, signal.SIGTERM)

  def _get_port(self, pid: int) -> int:
    port_path = os.path.join(self.towerfall_path, 'ports', str(pid))
    tries = 0
    print(f'Waiting for port file {port_path} to be created...')
    while not os.path.exists(port_path) and tries < 20:
      time.sleep(0.2)
      tries += 1
    with open(port_path, 'r') as file:
      return int(file.readline())

  def _save_state(self):
    with open(self.state_path, 'w') as file:
      file.write(json.dumps([p.to_dict() for p in self.processes], indent=2))

  def _match_config(self, config1, config2):
    return False
