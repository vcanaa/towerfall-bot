from typing import Any, Optional

from common.grid import GridView
from entity_envs.entity_env import TowerfallEntityEnvImpl
from envs.connection_provider import TowerfallProcessProvider

def create_kill_enemy(configs: dict[str, Any], record_path: Optional[str]=None, verbose=0):
  towerfall_provider = TowerfallProcessProvider('default')
  towerfall = towerfall_provider.get_process(
    fastrun=True,
    config=dict(
    mode='sandbox',
    level='1',
    fps=90,
    agents=[dict(type='remote', team='blue', archer='green')]
  ))
  return TowerfallEntityEnvImpl(
    towerfall=towerfall,
    record_path=record_path,
    verbose=verbose)