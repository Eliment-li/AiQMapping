from datetime import datetime
from copy import deepcopy
from pathlib import Path

import numpy as np
from ray.rllib.algorithms import Algorithm
import gymnasium as gym
from sympy import pprint

from config import ConfigSingleton
from env.env_helper import register_custom_env
from utils.file.file_util import get_root_dir
from  utils.visualize.trace import show_trace, show_result
import matplotlib.pyplot as plt
args = ConfigSingleton().get_config()
max_step = 20
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
    plot_evaluate([r_arr,dist_arr])

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
    trace = []
    trace.append(deepcopy(info['occupy']))
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
    plot_evaluate([r_arr, nn_arr, dist_arr])

# data[0]:  data[1]:
def plot_evaluate(data, save=True, max_length=20):
    '''
    # Example usage:
    data = [
        [10, 20, 30, 40, 50],
        [0, 342, 200, 500, 600],
        [3, 4, 5, 6, 7]
    ]
    label = ['Metric 1', 'Metric 2', 'Metric 3']

    plot_evaluate(data, label)
    '''

    # 创建一个图形和一个坐标轴
    fig, ax1 = plt.subplots()

    # Generate a colormap
    num_lines = len(data)
    colors = plt.cm.viridis(np.linspace(0, 1, num_lines))

    x1 = np.arange(len(data[0][:max_length]))
    y1 = data[0][:max_length]

    x2 = np.arange(len(data[1][:max_length]))
    y2 = data[1][:max_length]

    ax1.plot(x1, y1, label='reward', color='#5370c4', marker='o')
    ax1.set_xlabel('step')
    ax1.set_ylabel('reward', color='#5370c4')
    for x, y in zip(x1, y1):
        # Annotate each point with its value
        ax1.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

        # Draw a vertical line from each point to the x-axis
        ax1.axvline(x=x, ymin=0, ymax=(y - plt.ylim()[0]) / (plt.ylim()[1] - plt.ylim()[0]), color=colors[0],
                    linestyle='--', linewidth=0.5)

    # 创建第二个坐标轴，共享x轴
    ax2 = ax1.twinx()
    ax2.plot(x2, y2, label='distance', color='#f16569', marker='v')
    ax2.set_ylabel('distance', color='#f16569')
    for x, y in zip(x2, y2):
        # Annotate each point with its value
        ax2.annotate(f'{y:.0f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    # Add labels and legend
    plt.title('Evaluate Result')

    if save:
        path = Path(get_root_dir()) / 'data' / 'result' / (args.time_id + '.png')
        plt.savefig(path)
    # Show the plot
    # plt.show()


if __name__ == '__main__':
    #python evaluate.py /tmp/checkpoint_tmp_cfb4d4a183d2477c85bb76d4546b4c69
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('version', default="2", type=int, help='evaluate_policy version i.e. evaluate_policyv2')
    parser.add_argument('checkpoint', type=str, help='checkpoint path')
    #parser.add_argument('--verbose', action='store_true', help='enable verbose mode')
    command_args = parser.parse_args()


    register_custom_env(args.env_version)
    if command_args.version == 1:
        print(command_args.checkpoint)
        evaluate_policy(command_args.checkpoint)

    else:
        print(command_args.checkpoint)
        evaluate_policyv2(command_args.checkpoint)
