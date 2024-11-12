import os
from pprint import pprint

from ray.tune import ResultGrid, tune
from ray.tune.registry import get_trainable_cls

from config import ConfigSingleton

args = ConfigSingleton().get_config()

def load_res(path):
    experiment_path = os.path.join(path, 'exp_name')
    print(f"Loading results from {experiment_path}...")

    restored_tuner = tune.Tuner.restore(experiment_path, trainable=get_trainable_cls(args.run))
    result_grid = restored_tuner.get_results()

def analysis_res(results:ResultGrid):
    best_result = results.get_best_result()  # Get best result object
    best_config = best_result.config  # Get best trial's hyperparameters
    best_logdir = best_result.path  # Get best trial's result directory
    best_checkpoint = best_result.checkpoint  # Get best trial's best checkpoint
    best_metrics = best_result.metrics  # Get best trial's last results
    best_result_df = best_result.metrics_dataframe  # Get best result as pandas dataframe
    pprint(best_result_df)

    # Get a dataframe with the last results for each trial
    df_results = results.get_dataframe()

    # Get a dataframe of results for a specific score or mode
    #df = results.get_dataframe(filter_metric="score", filter_mode="max")
    df = results.get_dataframe()
    df.to_csv('d:/output.csv', index=False)
    print(df)
