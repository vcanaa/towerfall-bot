import logging
import random

from entity_gym.env import Observation, GlobalCategoricalActionMask

from typing import Any, Optional

import numpy as np
from common.constants import HH, HW
from common.entity import Entity, Vec2
from entity_envs.entity_base_env import TowerfallEntityEnv

from envs.connection_provider import TowerfallProcess, TowerfallProcessProvider


class TowerfallEntityEnvImpl(TowerfallEntityEnv):
  def __init__(self,
      record_path: Optional[str]=None,
      verbose: int = 0):
    towerfall_provider = TowerfallProcessProvider('entity-env-trainer')
    towerfall = towerfall_provider.get_process(
      fastrun=True,
      reuse=False,
      config=dict(
      mode='sandbox',
      level='3',
      agents=[dict(type='remote', team='blue', archer='green')]
    ))
    super().__init__(towerfall, record_path, verbose)
    self.enemy_count = 2
    self.min_distance = 50
    self.max_distance = 100
    self.episode_max_len = 60*6
    self.action_mask = {
      'hor': GlobalCategoricalActionMask(np.array([[True, True, True]])),
      'ver': GlobalCategoricalActionMask(np.array([[True, True, True]])),
      'jump': GlobalCategoricalActionMask(np.array([[True, True]])),
      'dash': GlobalCategoricalActionMask(np.array([[True, True]])),
      'shoot': GlobalCategoricalActionMask(np.array([[True, True]])),
    }

  def _is_reset_valid(self) -> bool:
    return True

  def _send_reset(self):
    reset_entities = self._get_reset_entities()
    self.towerfall.send_reset(reset_entities, verbose=self.verbose)

  def _post_reset(self) -> Observation:
    assert self.me, 'No player found after reset'
    targets = list(e for e in self.entities if e['isEnemy'])
    self.prev_enemy_ids = set(t['id'] for t in targets)

    self.done = False
    self.episode_len = 0
    self.reward = 0
    self.prev_arrow_count = len(self.me['arrows'])
    return self._get_obs(targets, [])

  def _post_observe(self) -> Observation:
    targets = list(e for e in self.entities if e['isEnemy'])
    self._update_reward(targets)
    self.episode_len += 1
    arrows = list(e for e in self.entities if e['type'] == 'arrow')
    return self._get_obs(targets, arrows)

  def _get_reset_entities(self) -> Optional[list[dict]]:
    p = Vec2(160, 110)
    entities: list[dict[str, Any]] = [dict(type='archer', pos=p.dict())]
    for i in range(self.enemy_count):
      sign = random.randint(0, 1)*2 - 1
      d = random.uniform(self.min_distance, self.max_distance) * sign
      enemy = dict(
        type='slime',
        pos=(p + Vec2(d, -5)).dict(),
        facing=-sign)
      entities.append(enemy)
    return entities

  def _update_reward(self, enemies: list[Entity]):
    '''
    Updates the reward and checks if the episode is done.
    '''
    # Negative reward for getting killed or end of episode
    self.reward = 0
    if not self.me or self.episode_len >= self.episode_max_len:
      self.done = True
      self.reward -= 1

    # Positive reward for killing an enemy
    enemy_ids = set(t['id'] for t in enemies)
    for id in self.prev_enemy_ids - enemy_ids:
      self.reward += 1

    if self.me:
      arrow_count = len(self.me['arrows'])
      delta_arrow = arrow_count - self.prev_arrow_count
      if delta_arrow > 0:
        self.reward += delta_arrow * 0.2
      else:
        self.reward += delta_arrow * 0.1
      self.prev_arrow_count = arrow_count


    if self.reward != 0:
      logging.info(f'Reward: {self.reward}')

    self.prev_enemy_ids = enemy_ids
    if len(self.prev_enemy_ids) == 0:
      self.done = True

  def limit(self, x: float, a: float, b: float) -> float:
    return x+b-a if x < a else x-b+a if x > b else x

  def _get_obs(self, enemies: list[Entity], arrows: list[Entity]) -> Observation:
    if not self.me:
      return Observation(
        done=self.done,
        reward=self.reward,
        actions=self.action_mask,
        global_features=np.array([0, 0], dtype=np.float32),
        entities={
          'enemies': [],
          'arrows': []
        }
      )

    enemie_states = []
    for enemy in enemies:
      enemie_states.append(np.array(
        [
          self.limit((enemy.p.x - self.me.p.x) / HW, -1, 1),
          self.limit((enemy.p.y - self.me.p.y) / HH, -1, 1),
          enemy['facing']
        ],
        dtype=np.float32
      ))

    arrow_states = []
    for arrow in arrows:
      arrow_states.append(np.array(
        [
          self.limit((arrow.p.x - self.me.p.x) / HW, -1, 1),
          self.limit((arrow.p.y - self.me.p.y) / HH, -1, 1),
          1 if arrow['state'] == 'stuck' else 0
        ],
        dtype=np.float32
      ))

    return Observation(
      done=self.done,
      reward=self.reward,
      actions=self.action_mask,
      global_features=np.array([
          (self.me.p.y-110) / HH,
          self.prev_arrow_count,
        ], dtype=np.float32),
      entities={
        'enemies': enemie_states,
        'arrows': arrow_states
      }
    )
