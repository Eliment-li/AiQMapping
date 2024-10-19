from copy import deepcopy

import numpy as np
from ray.rllib.algorithms import Algorithm
import gymnasium as gym
from sympy import pprint

from config import ConfigSingleton
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

        print('done = %r, action = %r, reward = %r,  info = %r \n' % (done,a, reward,info['occupy']))
        episode_reward += reward

        # Is the episode `done`? -> Reset.
        if done:
            print('env done = %r, action = %r, reward = %r  occupy =  {%r} ' % (done,a, reward, info['occupy']))
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
    pprint(trace.transpose())
    show_trace(trace.transpose())
