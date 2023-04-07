from gym import Space

from abc import ABC, abstractmethod

from common import GridView, Entity, Connection, Vec2, WIDTH, HEIGHT

from .objectives import FollowTargetObjective
from .observations import TowerfallObservation
from .objectives import TowerfallObjective

from typing import Tuple, Optional, Iterable


# class TowerfallCurriculum(TowerfallObjective):
#   def __init__(self, connection: Connection):
#     self.connection = connection

#   @abstractmethod
#   def reset(self):
#     raise NotImplemented


class FollowCloseTargetCurriculum(TowerfallObjective):
  def __init__(self, grid_view: GridView):
    self.gv = grid_view
    self.objective = FollowTargetObjective(grid_view)
    self.task_idx = 0
    self.start_ends: list[Tuple[Vec2, Vec2]] = []
    for start in self._pick_all_starts():
      for end in self._pick_all_ends(start):
        self.start_ends.append((start, end))

  def is_reset_valid(self, state_scenario: dict, player: Entity, entities: list[Entity]) -> bool:
    self.gv.set_scenario(state_scenario)
    self.gv.update(entities, player)
    hsize = player.s / 2
    if self.end.x < 0 or self.end.x > WIDTH or self.end.y < 0 or self.end.y > HEIGHT:
      return False
    if self.gv.is_region_collision(self.start - hsize, self.start + hsize):
      return False
    if self.gv.is_region_collision(self.end - hsize, self.end + hsize):
      return False
    return True

  def extend_obs_space(self, obs_space_dict: dict[str, Space]):
    self.objective.extend_obs_space(obs_space_dict)

  def get_reset_instruction(self):
    self.objective.env = self.env
    self.task_idx += 1
    self.start, self.end = self.start_ends[self.task_idx]
    return dict(pos = self.start.dict())

  def post_reset(self, state_scenario: dict, player: Entity, entities: list[Entity], obs_dict: dict):
    target = (self.end.x, self.end.y)
    self.objective.post_reset(state_scenario, player, entities, obs_dict, target)

  def post_step(self, player: Entity, entities: list[Entity], obs_dict: dict):
    self.objective.post_step(player, entities, obs_dict)
    self.rew = self.objective.rew
    self.done = self.objective.done

  def _pick_all_starts(self) -> Iterable[Vec2]:
    margin = 15
    for x in range(margin, WIDTH - margin, 10):
      for y in range(margin, HEIGHT - margin, 10):
        yield Vec2(x, y)

  def _pick_all_ends(self, start: Vec2) -> Iterable[Vec2]:
    dist = 30
    yield Vec2(start.x + dist, start.y + dist)
    # yield (start[0] - m, start[1] + m)
    # yield (start[0] - m, start[1] + m)
    # yield (start[0] - m, start[1] + m)