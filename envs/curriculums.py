import logging
import json
import random
import os

import numpy as np

from gym import Space

from common import GridView, Entity, Vec2, WIDTH, HEIGHT

from .objectives import FollowTargetObjective
from .objectives import TowerfallObjective

from typing import Tuple, Iterable, Optional


class TaskEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, Vec2):
      d = obj.dict()
      d['__Vec2__'] = True
      return d
    return super().default(obj)


class TaskDecoder(json.JSONDecoder):
  def __init__(self, *args, **kwargs):
    super().__init__(object_hook=self.object_hook, *args, **kwargs)

  def object_hook(self, dct):
    if '__Vec2__' in dct:
      del dct['__Vec2__']
      return Vec2(**dct)
    return dct


class FollowCloseTargetCurriculum(TowerfallObjective):
  '''
  Creates a series of episodes where the agent needs to get to a near target.
  The starting positions are uniformly distributed around the empty spaces of the scenario.

  :param grid_view: Used to detect collisions when resetting the task.
  :param distance: Original distance from the target.
  :param max_distance: When maximum distance from the target is reached the episode ends.
  :param bounty: Reward received when reaching the location.
  :param episode_max_len: Amount of frames after which the episode ends.
  :param rew_dc: Agent loses this amount of reward per frame, in order to force it to get the target faster.
  '''
  def __init__(self, grid_view: Optional[GridView], distance=8, max_distance=16, bounty=10, episode_max_len=90, rew_dc=2):
    print('FollowCloseTargetCurriculum')
    self.gv = grid_view
    self.distance = distance
    self.max_distance = max_distance
    self.objective = FollowTargetObjective(grid_view, distance, max_distance, bounty, episode_max_len, rew_dc)
    self.task_idx = -1
    self.filename = f'FollowCloseTargetCurriculum_episodes_{distance}.json'
    self.initialized = False
    if os.path.exists(self.filename):
      with open(self.filename, 'r') as file:
        self.start_ends = json.loads(file.read(), cls=TaskDecoder)
      random.shuffle(self.start_ends)
      self.initialized = True
    # self.start_ends = self.start_ends[:1]
    self.max_total_steps = episode_max_len * self.n_episodes
    print('FollowCloseTargetCurriculum created')

  @property
  def n_episodes(self):
    if not hasattr(self, 'start_ends'):
      return 1
    return len(self.start_ends)

  def is_reset_valid(self, state_scenario: dict, player: Optional[Entity], entities: list[Entity]) -> bool:
    if self.initialized:
      return True

    assert player, 'player is required to create curriculum'

    assert self.gv, 'grid_view is required to create curriculum'
    # if self.gv is None:
    #   self.gv = GridView(5)
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

    logging.info(f'Saving tasks to {self.filename}')
    with open(self.filename, 'w') as file:
      file.write(json.dumps(self.start_ends, cls=TaskEncoder))

    random.shuffle(self.start_ends)
    self.initialized = True
    return False

  def extend_obs_space(self, obs_space_dict: dict[str, Space]):
    self.objective.extend_obs_space(obs_space_dict)

  def get_reset_entities(self) -> Optional[list[dict]]:
    if not self.initialized:
      return None
    self.objective.env = self.env
    self.task_idx += 1
    if self.task_idx == len(self.start_ends):
      # TODO generate report
      # Restart from the first task.
      self.task_idx = 0
    self.start, self.end = self.start_ends[self.task_idx]
    return [dict(type='archer', pos=self.start.dict())]

  def post_reset(self, state_scenario: dict, player: Optional[Entity], entities: list[Entity], obs_dict: dict):
    target = (self.end.x, self.end.y)
    self.objective.post_reset(state_scenario, player, entities, obs_dict, target)

  def post_step(self, player: Optional[Entity], entities: list[Entity], command: str, obs_dict: dict):
    self.objective.post_step(player, entities, command, obs_dict)
    self.rew = self.objective.rew
    self.done = self.objective.done

  def _pick_all_starts(self) -> Iterable[Vec2]:
    return [Vec2(160, 110)]
    # margin = 15
    # for x in range(margin, WIDTH - margin, 10):
    #   for y in range(margin, HEIGHT - margin, 10):
    #     yield Vec2(x, y)

  def _pick_all_ends(self, start: Vec2) -> Iterable[Vec2]:
    v = Vec2(-self.distance, 20)
    n = 10
    yield start + v
    for i in range(10):
      v.x += self.distance / n * 2
      yield start + v


