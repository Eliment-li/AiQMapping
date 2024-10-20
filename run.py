import datetime
import pathlib
import time
from copy import copy, deepcopy

import psutil
from gymnasium import register
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.models.configs import ModelConfig
from sympy import timed

from env.env_helper import  register_custom_env
from env.env_v1 import CircuitEnv_v1
from env.env_v2 import CircuitEnv_v2
from env.env_v3 import CircuitEnv_v3
from env.env_v4 import CircuitEnv_v4
from env.env_v5 import CircuitEnv_v5

from config import ConfigSingleton
import numpy as np

import ray
from ray import air, tune
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.tune.registry import get_trainable_cls

from evaluate import evaluate_policy
from utils.visualize.trace import show_trace
args = ConfigSingleton().get_config()


stop = {
    TRAINING_ITERATION: args.stop_iters,
    NUM_ENV_STEPS_SAMPLED_LIFETIME: args.stop_timesteps,
}

# todo move to config.yml
env_config={
    'debug':False,
    #'name':'Env_1'
}
def train_policy():
    cpus  = psutil.cpu_count(logical=False)
    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(env=CircuitEnv_v5,env_config=env_config)
        .framework('torch')
        .rollouts(num_rollout_workers=int(cpus*0.9)
                  , num_envs_per_worker=2
                  # ,remote_worker_envs=True
                  )
        .resources(num_gpus=1)
        .training(
            model={
                # Change individual keys in that dict by overriding them, e.g.
                "fcnet_hiddens": [256,64,128,64,32],
                #"fcnet_hiddens": [32,64,128,64,32],
                "fcnet_activation": "relu",
                "use_attention": False,
               # "use_attention": args.use_attention,
                # "attention_num_transformer_units": args.attention_num_transformer_units,
                # "attention_use_n_prev_actions": args.prev_n_actions,
                # "attention_use_n_prev_rewards": args.prev_n_rewards,
                # "attention_dim": args.attention_dim,
                # "attention_memory_inference": args.attention_memory_inference,
                # "attention_memory_training": args.attention_memory_training,
            },
            gamma=0.99,
        )
    )
    #stop = {"training_iteration": 100, "episode_reward_mean": 300}
    # config['model']['fcnet_hiddens'] = [32, 32]
    # automated run with Tune and grid search and TensorBoard
    print(config)
    tuner = tune.Tuner(
        args.run,
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop,
                                checkpoint_config=air.CheckpointConfig(
                                checkpoint_frequency=args.checkpoint_frequency,
                                checkpoint_at_end=args.checkpoint_at_end,
                                 ))
    )
    results = tuner.fit()
    checkpoint = results.get_best_result().checkpoint
    print("Training completed")
    return checkpoint

def train():
    best_result = train_policy()
    evaluate_policy(best_result)


if __name__ == '__main__':
    register_custom_env(args.env_version)
    # from ray.tune import register_env
    # def env_creator(env_config):
    #     return gym.make('Env_1')  # return an instance of your custom environment
    #
    # register_env("Env_1", env_creator)

    args = ConfigSingleton().get_config()
    try:
        ray.init(num_gpus=1, local_mode=args.local_mode)
        time.sleep(1)
        train()
        ray.shutdown()
    except Exception as e:
        print(e)
