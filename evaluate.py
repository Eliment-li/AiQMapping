from datetime import datetime
from copy import deepcopy
import numpy as np
from ray.rllib.algorithms import Algorithm
import gymnasium as gym
from sympy import pprint

from config import ConfigSingleton
from env.env_helper import register_custom_env
from  utils.visualize.trace import show_trace, show_result
from utils.visualize.train import show_train_metric

args = ConfigSingleton().get_config()
max_step = 100
r_arr = []
dist_arr = []
nn_arr = []

#获取每一个 step 的数据，用于分析
def grap_metric(reward,info):
    dist_arr.append(info['distance'])
    r_arr.append(reward)
    nn_arr.append(info['nn'])


def evaluate_policy(results):

    if  not isinstance(results, str):
        checkpoint = results.get_best_result(metric='env_runners/episode_reward_mean', mode='max').checkpoint
        checkpoint = checkpoint.to_directory()
        print(f'best checkpoint: {checkpoint}')
        algo = Algorithm.from_checkpoint(path=checkpoint)
    else:
        algo = Algorithm.from_checkpoint(path=results)


    env = gym.make('Env_'+str(args.env_version))
    obs, info = env.reset()
    episode_reward = 0.0
    trace = []
    # trace
    trace.append(deepcopy(info['occupy']))
    for i in range(max_step):
        # Compute an action (`a`).
        a = algo.compute_single_action(
            observation=obs,
            explore=None,
            policy_id="default_policy",  # <- default value
        )
        obs, reward, done, truncated, info = env.step(a)
        #trace
        trace.append(deepcopy(info['occupy']))
        grap_metric(reward,info)
        print('done = %r, action = %r, reward = %r,  info = %r \n' % (done,a, reward,info['occupy']))
        episode_reward *=0.99
        episode_reward += reward

        # Is the episode `done`? -> Reset.
        if done:
            print('env done = %r, action = %r, reward = %r  occupy =  {%r} ' % (done,a, reward, info['occupy']))
            print(f"Episode done: Total reward = {episode_reward}")
            break

    algo.stop()
    trace = np.array(trace)
    pprint(trace.transpose())
    # if args.show_trace:
    #     show_trace(trace.transpose())
    show_result(trace[-1])
    show_train_metric([r_arr,dist_arr])

    #use attention
def evaluate_policyv2(results):
    if  not isinstance(results, str):
        checkpoint = results.get_best_result(metric='env_runners/episode_reward_mean', mode='max').checkpoint
        checkpoint = checkpoint.to_directory()
        print(f'best checkpoint: {checkpoint}')
        algo = Algorithm.from_checkpoint(path=checkpoint)
    else:
        algo = Algorithm.from_checkpoint(path=results)


    env = gym.make('Env_'+str(args.env_version))
    obs, info = env.reset()
    episode_reward = 0.0

    #attention start
    # In case the model needs previous-reward/action inputs, keep track of
    # these via these variables here (we'll have to pass them into the
    # compute_actions methods below).
    init_prev_a = prev_a = None
    init_prev_r = prev_r = None
    # Set attention net's initial internal state.
    num_transformers = int(args.attention_num_transformer_units)
    memory_inference =  int(args.attention_memory_inference)
    attention_dim =  int(args.attention_dim)
    init_state = state = [
        np.zeros([memory_inference, attention_dim], np.float32)
        for _ in range(num_transformers)
    ]
    # need prev-action/reward as part of the input?
    if args.prev_n_actions:
        init_prev_a = prev_a = np.array([0] * int(args.prev_n_actions))
    if args.prev_n_rewards:
        init_prev_r = prev_r = np.array([0.0] * int(args.prev_n_rewards))
    #attention end

    # trace
    trace = []
    trace.append(deepcopy(info['occupy']))
    for i in range(max_step):
        a, state_out, _ = algo.compute_single_action(
            observation=obs,
            state=state,
            prev_action=prev_a,
            prev_reward=prev_r,
            explore=args.explore_during_inference,
            policy_id="default_policy",  # <- default value
        )

        obs, reward, done, truncated, info = env.step(a)
        #trace
        trace.append(deepcopy(info['occupy']))
        grap_metric(reward, info)
        print('done = %r, action = %r, reward = %r,  info = %r \n' % (done,a, reward,info['occupy']))
        episode_reward *=args.gamma
        episode_reward += reward

        if done:
            print('env done = %r, action = %r, reward = %r  occupy =  {%r} ' % (done,a, reward, info['occupy']))
            print(f"Episode done: Total reward = {episode_reward}")
            print(f"CheckPoint = {checkpoint}")
            break
        else:
            # Append the just received state-out (most recent timestep) to the
            # cascade (memory) of our state-ins and drop the oldest state-in.
            state[0] = np.roll(state[0], -1, axis=0)
            state[0][-1, :] = state_out[0]

            if init_prev_a is not None:
                prev_a = np.roll(prev_a, -1,axis=0)
                prev_a[-1, :] = a
            if init_prev_r is not None:
                prev_r = np.roll(prev_r, -1)
                prev_r[-1] = a
    algo.stop()
    trace = np.array(trace)
    #pprint(trace.transpose())
    #save_array(trace,file)
    # if args.show_trace:
    #     show_trace(trace.transpose())
    show_result(trace[-1])
    show_train_metric([r_arr, nn_arr, dist_arr])

if __name__ == '__main__':
    register_custom_env(args.env_version)
    #evaluate_policy('/tmp/checkpoint_tmp_cfb4d4a183d2477c85bb76d4546b4c69')

    evaluate_policyv2(r'C:/Users/Administrator/ray_results/AiQMapping/PPO_b30b6_00002/checkpoint_000000')
