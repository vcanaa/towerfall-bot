import json
import logging
import os
from typing import Any, Callable, Optional, Tuple
from gym import Env

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import yaml

from envs.blank_env import TowerfallBlankEnv

from .train_callback import TrainCallback


class Trainer:
  '''Class that trains new models and resume training of existing models.'''
  def __init__(self, export_wandb: bool = True):
    self.export_wandb = export_wandb

  def init_wandb(self, configs: dict[str, Any], project_name: str, trial_name: str):
    from wandb.wandb_run import Run

    import wandb

    logging.info(f'Creating wandb run for trial {project_name}/{trial_name}')
    self.run = wandb.init(
        project=project_name,
        name=trial_name,
        tags=[],
        config=configs,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )
    assert type(self.run) is Run, f'Expected Run, got {type(self.run)}'
    logging.info(f'Initialized wandb run {self.run.name}, id:{self.run.id}')

  def get_wandb_callback(self):
    from wandb.integration.sb3 import WandbCallback
    return WandbCallback(verbose=2)

  def get_latest_model(self, trial_path: str) -> Optional[str]:
    model_names = os.listdir(os.path.join(trial_path, 'models'))
    latest_model = None
    last_step = -1
    for name in model_names:
      if name == 'model.zip':
        latest_model = 'model.zip'
        break
      step = int(name.replace('.zip', ''))
      if step > last_step:
        last_step = step
        latest_model = name
    return latest_model

  def load_from_trial(self, project_name: str, trial_name: str, model_name: Optional[str] = None, env: Optional[Env] = None) -> Tuple[PPO, dict[str, Any]]:
    trial_path = self.get_trial_path(project_name, trial_name)
    logging.info(f'Loading experiment from {trial_path}')
    with open(os.path.join(trial_path, 'hparams.json'), 'r') as file:
      configs = json.load(file)

    if model_name is None:
      last_model = self.get_latest_model(trial_path)
      assert last_model is not None, f'No model found in {trial_path}'
      model_path = os.path.join(trial_path, 'models', last_model)
    else:
      model_path = os.path.join(trial_path, 'models', model_name)

    if env:
      model = PPO.load(model_path, env)
    else:
      model = PPO.load(model_path)
    logging.info(f'Loaded model {model_path}')
    return model, configs

  def get_trial_path(self, project_name: str, trial_name: str):
    return f'tmp/{project_name}/{trial_name}'

  def _train_model(self, model, env: TowerfallBlankEnv, total_steps: int, configs: dict[str, Any], project_name: str, trial_name: str) -> float:
    trial_path = self.get_trial_path(project_name, trial_name)
    os.makedirs(trial_path, exist_ok=True)

    logging.info(f"Starting to train for {total_steps} timesteps...")
    try:
      if self.export_wandb:
        self.init_wandb(configs, project_name, trial_name)

      callbacks = []
      train_callback = TrainCallback(
        configs['ppo_params']['n_steps'],
        n_episodes = 100,
        log_dir=trial_path)
      callbacks.append(train_callback)
      if self.export_wandb:
        callbacks.append(self.get_wandb_callback())

      with open(os.path.join(trial_path, 'hparams.json'), 'w') as file:
        file.write(json.dumps(configs, indent=2))

      logging.info(yaml.dump(configs, indent=2))
      logging.info('Starting model learn')
      model.learn(
        progress_bar=True,
        callback=callbacks,
        total_timesteps=total_steps,
        **configs['learn_params'])
      model.logger.dump()
      logging.info('Finished model learn')
    finally:
      if hasattr(self, 'run'):
        assert self.run
        logging.info('Finishing W&B run.')
        self.run.finish()
    return train_callback.best_mean_reward

  def train(self, env: TowerfallBlankEnv, total_steps: int, configs: dict[str, Any], project_name: str, trial_name: str):
    trial_path = self.get_trial_path(project_name, trial_name)
    logging.info(f'Creating Monitor in {trial_path}')
    monitored_env = Monitor(env, os.path.join(trial_path, 'monitor'))
    model = PPO(
      env=monitored_env,
      verbose=2,
      tensorboard_log=os.path.join(trial_path, 'tensorboard'),
      **configs['ppo_params'],
    )

    return self._train_model(model, env, total_steps, configs, project_name, trial_name)

  def fork_training(self,
                    env: TowerfallBlankEnv,
                    total_steps: int,
                    configs: dict[str, Any],
                    project_name: str,
                    trial_name: str,
                    load_project_name: str,
                    load_trial_name: str,
                    load_model_name: str):
    trial_path = self.get_trial_path(project_name, trial_name)
    logging.info(f'Creating Monitor in {trial_path}')
    monitored_env = Monitor(env, os.path.join(trial_path, 'monitor'))

    model, _ = self.load_from_trial(load_project_name, load_trial_name, load_model_name, monitored_env)
    self._train_model(model, env, total_steps, configs, project_name, trial_name)

  def evaluate_model(self, env_fn: Callable[[dict[str, Any]], Env], n_episodes: int, project_name: str, trial_name: str, model_name: str):
    model, configs = self.load_from_trial(project_name, trial_name, model_name)
    env = env_fn(configs)
    evaluate_policy(model, env=env, n_eval_episodes=n_episodes, render=False, deterministic=False)

  def evaluate_all_models(self, env_fn: Callable[[dict[str, Any]], Env], n_episodes: int, project_name: str, trial_name: str):
    trial_path = self.get_trial_path(project_name, trial_name)
    logging.info(f'Loading experiment from {trial_path}')
    with open(os.path.join(trial_path, 'hparams.json'), 'r') as file:
      configs = json.load(file)

    model_names = os.listdir(os.path.join(trial_path, 'models'))
    model_numbers = []
    for name in model_names:
      try:
        step = int(name.replace('.zip', ''))
        model_numbers.append(step)
      except ValueError:
        continue

    model_numbers.sort()
    for model_number in model_numbers:
      model_path = os.path.join(trial_path, 'models', f'{model_number}.zip')
      model = PPO.load(model_path)
      logging.info(f'Loaded model {model_path}')
      env = env_fn(configs)
      evaluate_policy(model, env=env, n_eval_episodes=n_episodes, render=False, deterministic=False)