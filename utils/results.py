import os
from datetime import datetime
from pathlib import Path
from pprint import pprint

from ray.tune import ResultGrid, tune
from ray.tune.registry import get_trainable_cls

from config import ConfigSingleton
from utils.file.file_util import get_root_dir

args = ConfigSingleton().get_config()

key_metric=[
'trial_id',
'date',
'training_iteration',
'env_runners/episode_reward_min',
'env_runners/episode_reward_max',
'env_runners/episode_reward_mean',
'env_runners/hist_stats/episode_reward',
'config/env',
'config/gamma',
'config/lr',
'config/model/fcnet_hiddens',
'config/model/fcnet_activation',
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

def load_res(path):
    experiment_path = os.path.join(path, 'exp_name')
    print(f"Loading results from {experiment_path}...")

    restored_tuner = tune.Tuner.restore(experiment_path, trainable=get_trainable_cls(args.run))
    result_grid = restored_tuner.get_results()
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
    datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    p = Path(get_root_dir())
    output_path = p / 'data' / 'result' / (str(args.stop_iters) + '_' + datetime_str + '.output.csv')

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
    df = filter_df(df, perf_metric, key_metric)
    df.to_csv(output_path,mode='x')

    best_result_df = results.get_dataframe(
        filter_metric="env_runners/episode_reward_mean", filter_mode="max"
    )
    best_result_df = filter_df(best_result_df, perf_metric, key_metric)
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