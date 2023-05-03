from typing import Any, Optional

from common.grid import GridView
from envs.actions import TowerfallActions
from envs.blank_env import TowerfallBlankEnv
from envs.connection_provider import TowerfallProcessProvider
from envs.curriculums import FollowCloseTargetCurriculum
from envs.kill_enemy_objective import KillEnemyObjective
from envs.observations import GridObservation


def create_simple_move_env(configs: dict[str, Any], record_path: Optional[str]=None, verbose=0):
  grid_view = GridView(grid_factor=5)
  objective = FollowCloseTargetCurriculum(grid_view, **configs['objective_params'])
  towerfall_provider = TowerfallProcessProvider('default')
  towerfall = towerfall_provider.get_process(
    fastrun=True,
    config=dict(
    mode='sandbox',
    level='1',
    agents=[dict(type='remote', team='blue', archer='green')]
  ))
  return TowerfallBlankEnv(
    towerfall=towerfall,
    observations= [
      GridObservation(grid_view, **configs['grid_params']),
    ],
    objective=objective,
    actions=TowerfallActions(**configs['actions_params']),
    record_path=record_path,
    verbose=verbose)


def create_kill_enemy(configs: dict[str, Any], record_path: Optional[str]=None, verbose=0):
  objective = KillEnemyObjective(**configs['objective_params'])
  towerfall_provider = TowerfallProcessProvider('default')
  towerfall = towerfall_provider.get_process(
    fastrun=True,
    config=dict(
    mode='sandbox',
    level='1',
    fps=90,
    agents=[dict(type='remote', team='blue', archer='green')]
  ))
  return TowerfallBlankEnv(
    towerfall=towerfall,
    observations=[],
    objective=objective,
    actions=TowerfallActions(**configs['actions_params']),
    record_path=record_path,
    verbose=verbose)