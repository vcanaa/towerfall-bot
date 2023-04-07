import logging
import json
import random

from gym import Space

from abc import ABC, abstractmethod

from common import GridView, Entity, Connection, Vec2, WIDTH, HEIGHT

from .objectives import FollowTargetObjective
from .objectives import TowerfallObjective

from typing import Tuple, Iterable


class MyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, Vec2):
      d = obj.dict()
      d['__Vec2__'] = True
      return d
    return super().default(obj)

class MyDecoder(json.JSONDecoder):
  def __init__(self, *args, **kwargs):
    super().__init__(object_hook=self.object_hook, *args, **kwargs)

  def object_hook(self, dct):
    if '__Vec2__' in dct:
      return Vec2(**dct)
    return dct

class FollowCloseTargetCurriculum(TowerfallObjective):
  '''Creates a series tasks where the agent needs to get to a closeby target.'''
  def __init__(self, grid_view: GridView):
    self.gv = grid_view
    self.objective = FollowTargetObjective(grid_view)
    self.task_idx = -1
    self.initialized = False

  def is_reset_valid(self, state_scenario: dict, player: Entity, entities: list[Entity]) -> bool:
    if self.initialized:
      return True
    self.gv.set_scenario(state_scenario)
    self.gv.update(entities, player)
    self.start_ends: list[Tuple[Vec2, Vec2]] = []
    hsize = player.s / 2
    for start in self._pick_all_starts():
      if self.gv.is_region_collision(start - hsize, start + hsize):
        logging.info(f'Start collides. {start}')
        continue
      if not self.gv.is_region_collision(start - hsize - Vec2(0, 10), start - hsize + Vec2(player.s.x, 0)):
        logging.info(f'No floor for start at {start}')
        continue
      for end in self._pick_all_ends(start):
        if end.x < 0 or end.x > WIDTH or end.y < 0 or end.y > HEIGHT:
          logging.info(f'End out of bounds. {end}')
        if self.gv.is_region_collision(end - hsize, end + hsize):
          logging.info(f'End collides. {end}')
          continue
        if not self.gv.is_clean_path(start, end):
          logging.info(f'No clean path between {start} and {end}')
          continue
        self.start_ends.append((start, end))

    filename = 'tasks.json'
    logging.info(f'Saving tasks to {filename}')
    with open(filename, 'w') as file:
      file.write(json.dumps(self.start_ends, cls=MyEncoder))

    random.shuffle(self.start_ends)
    self.initialized = True

  def extend_obs_space(self, obs_space_dict: dict[str, Space]):
    self.objective.extend_obs_space(obs_space_dict)

  def get_reset_instruction(self):
    if not self.initialized:
      return {}
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
    d = 30
    for dx in range(-d, d + 1, 2 * d):
      for dy in range(-d, d + 1, d):
        yield Vec2(start.x + dx, start.y + dy)
    for dy in range(-d, d + 1, 2*d):
      yield Vec2(start.x, start.y + dy)