import os
import json

from copy import deepcopy
from subprocess import Popen, PIPE

from common import Connection

class TowerfallConnectionProvider:
  def __init__(self):
    self.towerfall_path = 'C:/Program Files (x86)/Steam/steamapps/common/TowerFall Ascension/TowerFall.exe'
    self.config_path = '.towerfall_connections'
    os.makedirs(self.config_path)
    self.unused_ports = set(range(12024, 12124))
    self.unused_processes = set()
    self.default_config = dict(
      mode='sandbox',
      level='2',
      fastrun=False,
    )

  def get_connection(self, config=None):
    if not config:
      config = deepcopy(self.default_config)

    cconfig_path = os.path.join(self.config_path, len(self.connections), 'config.json')
    with open(cconfig_path, 'w') as file:
      file.write(json.dumps(config, indent=2))


    process = Popen([self.towerfall_path, '-config', cconfig_path])
    connection = Connection('localhost', self.unused_ports.pop())
    def on_close():
      self.unused_ports.add(connection.port)
      self.unused_processes.add(process)
    connection.on_close = on_close


