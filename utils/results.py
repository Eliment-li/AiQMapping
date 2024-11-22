import os
from datetime import datetime
from pathlib import Path
from pprint import pprint

import pandas as pd
from ray.tune import ResultGrid, tune
from ray.tune.registry import get_trainable_cls

from config import ConfigSingleton
from utils.file.file_util import get_root_dir

args = ConfigSingleton().get_config()

key_metric=[
'trial_id',
'iteration',
'episode_reward_min',
'episode_reward_max',
'episode_reward_mean',
'gamma',
'lr',
'fcnet_hiddens',
'fcnet_activation',
'date',
'env',
'config/model/post_fcnet_activation',
'config/model/use_attention',
'config/model/attention_num_transformer_units',
'config/model/attention_dim',
'config/model/attention_num_heads',
'config/model/attention_head_dim',
'config/model/attention_memory_inference',
'config/model/attention_memory_training',
'config/model/attention_position_wise_mlp_dim',
'config/model/attention_use_n_prev_actions',
'config/model/attention_use_n_prev_rewards',
'config/model/attention_init_gru_gate_bias',
'config/model/dim',
'config/explore',
'config/count_steps_by',
'env_runners/hist_stats/episode_reward',
]
perf_metric=[
'iterations_since_restore',
'env_runners/sampler_perf/mean_env_wait_ms',
'timers/sample_time_ms',
'timers/training_step_time_ms',
'perf/cpu_util_percent',
'perf/ram_util_percent',
'logdir',
]

head_map = {'config/lr': 'lr',
            'training_iteration': 'iteration',
            'env_runners/episode_reward_min': 'episode_reward_min',
            'env_runners/episode_reward_max': 'episode_reward_max',
            'env_runners/episode_reward_mean': 'episode_reward_mean',
            'config/gamma': 'gamma',
            'config/model/fcnet_hiddens': 'fcnet_hiddens',
            'config/model/fcnet_activation': 'fcnet_activation',
            'config/env': 'env',
}

def load_res(path):
    experiment_path = os.path.join(path, 'exp_name')
    print(f"Loading results from {experiment_path}...")

    restored_tuner = tune.Tuner.restore(experiment_path, trainable=get_trainable_cls(args.run))
    result_grid = restored_tuner.get_results()

# a function to change the head of given dataframe ,input is a map, key = old value, value = new value
def change_head(df, head_map):
    '''
        data = {
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': [7, 8, 9]
    }
    df = pd.DataFrame(data)
    change_head(df, {'a': 'A', 'b': 'B', 'c': 'C'})
    print(df)
    :param df:
    :param head_map:
    :return:
    '''
    df.columns = [head_map.get(col, col) for col in df.columns]
    return df



'''
filter the panda.dataframe
drop columns not in given list, order the colums using given list and return the new dataframe
params can be more than one list
'''
def filter_df(df, *params):
    """
       Filters the given pandas DataFrame by dropping columns not in the given lists,
       orders the columns using the given lists, and returns the new DataFrame.

       Args:
           df (pd.DataFrame): The DataFrame to filter.
           *params (List[str]): One or more lists of column names to keep and order.

       Returns:
           pd.DataFrame: The filtered and ordered DataFrame.
        example
               data = {
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': [7, 8, 9]
    }
    df = pd.DataFrame(data)
    print(df)
    print(filter_df(df, ['a', 'c']))
    print(filter_df(df, ['b', 'a'], ['c']))
    print(filter_df(df, ['c', 'b', 'a']))
    print(filter_df(df, ['c'], ['a', 'b']))
    print(filter_df(df, ['b'], ['c', 'a']))
       """
    columns_to_keep = [col for sublist in params for col in sublist]
    # Drop columns not in the given list
    df = df.drop(columns=[col for col in df.columns if col not in columns_to_keep])
    # Order the columns using the given list
    df = df[columns_to_keep]
    return df


def analysis_res(results:ResultGrid):
    output_path = Path(get_root_dir()) / 'data' / 'result' / (str(args.stop_iters) + '_' + args.time_id + '.output.csv')

    # best_result = results.get_best_result("mean_loss", "min")  # Get best result object
    # best_config = best_result.config  # Get best trial's hyperparameters
    # best_logdir = best_result.path  # Get best trial's result directory
    # best_checkpoint = best_result.checkpoint  # Get best trial's best checkpoint
    # best_metrics = best_result.metrics  # Get best trial's last results
    # best_result_df = best_result.metrics_dataframe  # Get best result as pandas dataframe

    #best_result_df.to_csv(output_path,mode='a', index=False)
    # Get a dataframe of results for a specific score or mode
    #df = results.get_dataframe(filter_metric="score", filter_mode="max")

    df = results.get_dataframe()
    df = change_head(df,head_map )
    df = filter_df(df, key_metric,perf_metric )
    df.to_csv(output_path,mode='x')

    best_result_df = results.get_dataframe(
        filter_metric="env_runners/episode_reward_mean", filter_mode="max"
    )
    best_result_df = change_head(best_result_df,head_map )
    best_result_df = filter_df(best_result_df, key_metric,perf_metric )
    best_result_df.to_csv(output_path,mode='a')
    '''

    results_df = results.get_dataframe()
    results_df[["training_iteration", "episode_reward_mean"]]

    print("Shortest training time:", results_df["time_total_s"].min())
    print("Longest training time:", results_df["time_total_s"].max())

   
    '''

def iterate_res(results:ResultGrid):
    # Iterate over results
    for i, result in enumerate(results):
        if result.error:
            print(f"Trial #{i} had an error:", result.error)
            continue

        print(
            f"Trial #{i} finished successfully with a mean accuracy metric of:",
            result.metrics["mean_accuracy"]
        )


# Example usage
if __name__ == '__main__':
    pass