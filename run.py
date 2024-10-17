import datetime
import pathlib
from copy import copy, deepcopy

from gymnasium import register
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.models.configs import ModelConfig
from env.env_v1 import CircuitEnv_v1

from config import ConfigSingleton

import argparse
import gymnasium as gym
import numpy as np
import os

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

from utils.visualize.trace import show_trace


def evaluate_policy(checkpoint):
    if not isinstance(checkpoint, str):
        checkpoint = checkpoint.to_directory()
    algo = Algorithm.from_checkpoint(checkpoint)
    env = gym.make('Env_1')
    obs, info = env.reset()

    episode_reward = 0.0
    #attention
    # num_transformers = config["model"]["attention_num_transformer_units"]
    # memory_inference = config["model"]["attention_memory_inference"]
    # attention_dim = config["model"]["attention_dim"]
    # init_state = state = [
    #     np.zeros([memory_inference, attention_dim], np.float32)
    #     for _ in range(num_transformers)
    # ]
    # trace
    trace = []
    trace.append(deepcopy(info['occupy']))
    done = False
    while not done:
        # Compute an action (`a`).
        a = algo.compute_single_action(
            observation=obs,
            explore=None,
            policy_id="default_policy",  # <- default value
        )
        # Send the computed action `a` to the env.

        obs, reward, done, truncated, info = env.step(a)
        #trace
        trace.append(deepcopy(info['occupy']))

        print('done = %r, reward = %r  info = %r \n' % (done, reward,info['occupy']))
        episode_reward += reward

        # Is the episode `done`? -> Reset.
        if done:
            print('env done = %r, reward = %r \n obs = \n {%r} ' % (done, reward, obs[-3:]))
            print(f"Episode done: Total reward = {episode_reward}")

            if not isinstance(checkpoint, str):
                checkpoint = checkpoint.path

            obs, info = env.reset()
            episode_reward = 0.0

    algo.stop()
    trace = np.array(trace)
    #print(trace)
    show_trace(trace.transpose())


# todo move to config.yml
env_config={
    'debug':False,
    'name':'Env_1'
}
def train_policy():
    config = (
        PPOConfig()
        .environment(env=CircuitEnv_v1,env_config=env_config)
        .framework('torch')
        # Switch both the new API stack flags to True (both False by default).
        # This enables the use of
        # a) RLModule (replaces ModelV2) and Learner (replaces Policy)
        # b) and automatically picks the correct EnvRunner (single-agent vs multi-agent)
        # and enables ConnectorV2 support.
        # .api_stack(
        #     enable_rl_module_and_learner=True,
        #     enable_env_runner_and_connector_v2=True,
        # )
        .resources(
            num_cpus_for_main_process=8,
        )
        # We are using a simple 1-CPU setup here for learning. However, as the new stack
        # supports arbitrary scaling on the learner axis, feel free to set
        # `num_learners` to the number of available GPUs for multi-GPU training (and
        # `num_gpus_per_learner=1`).
        .learners(
            num_learners=1,  # <- in most cases, set this value to the number of GPUs
            num_gpus_per_learner=1,  # <- set this to 1, if you have at least 1 GPU
        )
        # When using RLlib's default models (RLModules) AND the new EnvRunners, you should
        # set this flag in your model config. Having to set this, will no longer be required
        # in the near future. It does yield a small performance advantage as value function
        # predictions for PPO are no longer required to happen on the sampler side (but are
        # now fully located on the learner side, which might have GPUs available).
        .training(
            model={
                "use_attention": False,
            },
            gamma=0.99,
        )
    )
    #stop = {"training_iteration": 100, "episode_reward_mean": 300}
    # config['model']['fcnet_hiddens'] = [32, 32]
    # automated run with Tune and grid search and TensorBoard
    tuner = tune.Tuner(
        'PPO',
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop={"training_iteration": 20},
                                 checkpoint_config=air.CheckpointConfig(
                                     checkpoint_frequency=1,
                                     checkpoint_at_end=True,
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
    register(
        id='Env_1',
        # entry_point='core.envs.circuit_env:CircuitEnv',
        entry_point='env.env_v1:CircuitEnv_v1',
        max_episode_steps=2000,
    )
    from ray.tune import register_env
    def env_creator(env_config):
        return gym.make('Env_1')  # return an instance of your custom environment

    register_env("Env_1", env_creator)

    args = ConfigSingleton().get_config()
    try:
        ray.init(num_gpus=1, local_mode=args.local_mode)
        train()
        ray.shutdown()
    except Exception as e:
        print(e)
