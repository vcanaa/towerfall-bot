from typing import Optional

from entity_envs.entity_env import TowerfallEntityEnvImpl


def create_kill_enemy(record_path: Optional[str]=None, verbose=0):
  return TowerfallEntityEnvImpl(
    record_path=record_path,
    verbose=verbose)