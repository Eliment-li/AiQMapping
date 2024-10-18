import datetime
import pathlib
import time
from copy import copy, deepcopy

from gymnasium import register
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.models.configs import ModelConfig
from sympy import timed

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
args = ConfigSingleton().get_config()

def evaluate_policy(checkpoint):
    if not isinstance(checkpoint, str):
        checkpoint = checkpoint.to_directory()
    algo = Algorithm.from_checkpoint(checkpoint)
    env = gym.make('Env_1')
    obs, info = env.reset()

    episode_reward = 0.0
    #attention

    # In case the model needs previous-reward/action inputs, keep track of
    # these via these variables here (we'll have to pass them into the
    # compute_actions methods below).
    init_prev_a = prev_a = None
    init_prev_r = prev_r = None
    # Set attention net's initial internal state.
    # num_transformers = args.attention_num_transformer_units
    # memory_inference =  args.attention_memory_inference
    # attention_dim =  args.attention_dim
    # init_state = state = [
    #     np.zeros([memory_inference, attention_dim], np.float32)
    #     for _ in range(num_transformers)
    # ]
    # Do we need prev-action/reward as part of the input?
    # if args.prev_n_actions:
    #     init_prev_a = prev_a = np.array([0] * int(args.prev_n_actions))
    # if args.prev_n_rewards:
    #     init_prev_r = prev_r = np.array([0.0] * int(args.prev_n_rewards))

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
        #attention
        # a, state_out, _ = algo.compute_single_action(
        #     observation=obs,
        #     state=state,
        #     prev_action=prev_a,
        #     prev_reward=prev_r,
        #     explore=args.explore_during_inference,
        #     policy_id="default_policy",  # <- default value
        # )

        # Send the computed action `a` to the env.

        obs, reward, done, truncated, info = env.step(a)
        #trace
        trace.append(deepcopy(info['occupy']))

        print('done = %r, reward = %r  info = %r \n' % (done, reward,info['occupy']))
        episode_reward += reward

        # Is the episode `done`? -> Reset.
        if done:
            print('env done = %r, reward = %r \n occupy = \n {%r} ' % (done, reward, info['occupy']))
            print(f"Episode done: Total reward = {episode_reward}")

            # attention
            # state = init_state
            # prev_a = init_prev_a
            # prev_r = init_prev_r

            if not isinstance(checkpoint, str):
                checkpoint = checkpoint.path

            obs, info = env.reset()
            episode_reward = 0.0
        # attention
        # else:
        #     # Append the just received state-out (most recent timestep) to the
        #     # cascade (memory) of our state-ins and drop the oldest state-in.
        #     state = [
        #         np.concatenate([state[i], [state_out[i]]], axis=0)[1:]
        #         for i in range(num_transformers)
        #     ]
        #     if init_prev_a is not None:
        #         prev_a = a
        #     if init_prev_r is not None:
        #         prev_r = reward

    algo.stop()
    trace = np.array(trace)
    show_trace(trace.transpose())


# todo move to config.yml
env_config={
    'debug':False,
    'name':'Env_1'
}
def train_policy():

    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(env=CircuitEnv_v1,env_config=env_config)
        .framework('torch')
        .resources(
            num_cpus_for_main_process=8,
        )
        .training(
            model={
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
    print('train')
    best_result = train_policy()
    print('evaluation')
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
        time.sleep(3)
        train()
        ray.shutdown()
    except Exception as e:
        print(e)
