from typing import Any, Dict, Optional

from common.grid import GridView
from envs.actions import TowerfallActions
from envs.blank_env import TowerfallBlankEnv
from envs.curriculums import FollowCloseTargetCurriculum
from envs.kill_enemy_objective import KillEnemyObjective
from envs.observations import GridObservation
from towerfall import Towerfall


def create_simple_move_env(configs: Dict[str, Any], record_path: Optional[str]=None, verbose=0):
  grid_view = GridView(grid_factor=5)
  objective = FollowCloseTargetCurriculum(grid_view, **configs['objective_params'])
  towerfall = Towerfall(
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


def create_kill_enemy(configs: Dict[str, Any], record_path: Optional[str]=None, verbose=0):
  objective = KillEnemyObjective(**configs['objective_params'])
  towerfall = Towerfall(
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