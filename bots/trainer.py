from gym import Env
from common import *
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from threading import Thread

class MoveTrainer(Env):
  def __init__(self):
    self.done = False
    self.target = None
    self.env = EnvWrap()
    check_env(self.env)
    self.frames = 0
    self.move_model = DQN(
      env=self.env,
      batch_size=128,
      buffer_size=50000,
      exploration_final_eps=0.1,
      exploration_fraction=0.12,
      gamma=0.99,
      gradient_steps=-1,
      learning_rate=0.00063,
      learning_starts=0,
      policy="MlpPolicy",
      policy_kwargs={'net_arch': [256, 256]},
      target_update_interval=250,
      train_freq=4,
      verbose=1)

    def learn():
      self.move_model.learn(total_timesteps=1000)

    self.th_learn = Thread(target=learn)


  def update(self, grid: NDArray, me: Entity, controls: Controls):
    def rand_double_region(a: float, b: float):
      x = random.randint(int(a), int(b))
      return x if random.randint(0, 1) else x

    if self.target == None or self.done:
      while True:
        x = me.p.x + rand_double_region(30, 50)
        y = me.p.y + rand_double_region(30, 50)
        i, j = gridPos(Vec2(x, y))
        if not grid[i][j]:
          break
      self.target = Entity(e = {
        'pos': {'x': x, 'y': y},
        'size':{'x': 5, 'y': 5},
        'type': 'fake'
      })

    target_dist = distance(me.p, self.target.p)
    rew = 0
    if (self.frames > 0):
      if self.prev_target_dist > 0:
        rew = self.prev_target_dist - target_dist
    if target_dist < 5:
      rew += 50
      self.done = True
    if target_dist > 50:
      self.done = True
    self.prev_target_dist = target_dist
    self.env.update(obs=grid, rew=0, done=self.done)
    actions = self.env.get_actions()

    if actions[0]:
      controls.left()
    if actions[1]:
      controls.right()
    if actions[2]:
      controls.down()
    if actions[3]:
      controls.up()
    if actions[4]:
      controls.jump()
    if actions[5]:
      controls.dash()
