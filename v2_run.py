import datetime
import sys
import time
from io import StringIO
from pathlib import Path
import psutil
from ray.rllib.algorithms import PPOConfig
from ray.tune import ResultGrid

from env.env_helper import  register_custom_env
from env.env_v11 import CircuitEnv_v11
from env.env_v12 import CircuitEnv_v12
from config import ConfigSingleton

import ray
from ray import air, tune
from ray.air.constants import TRAINING_ITERATION
from ray.tune.registry import get_trainable_cls

from utils.common_utils import parse_tensorboard, copy_folder
from evaluate import evaluate_policyv2
from utils.file.file_util import write, get_root_dir
from utils.results import analysis_res

args = ConfigSingleton().get_config()
args_pri = ConfigSingleton().get_config_private()

stop = {
    TRAINING_ITERATION: args.stop_iters,
    #NUM_ENV_STEPS_SAMPLED_LIFETIME: args.stop_timesteps,
}

env_config={
    'debug':False,
}
def train_policy() -> ResultGrid:
    cpus  = psutil.cpu_count(logical=True)
    config = PPOConfig()\
    .environment(env=CircuitEnv_v11, env_config=env_config)\
    .framework('torch')\
    .rollouts(num_rollout_workers=int(cpus * 0.7), num_envs_per_worker=2)\
    .resources(num_gpus=args.num_gpus)\
    .training(
        model={
            # "fcnet_hiddens":args.fcnet_hiddens ,
             "fcnet_hiddens": tune.grid_search(args.fcnet_hiddens_grid),
            "fcnet_activation": tune.grid_search(args.fcnet_activation),
            "use_attention": True,
                "attention_num_transformer_units": args.attention_num_transformer_units,
                "attention_use_n_prev_actions": args.prev_n_actions,
                "attention_use_n_prev_rewards": args.prev_n_rewards,
                "attention_dim": args.attention_dim,
                "attention_memory_inference": args.attention_memory_inference,
                "attention_memory_training": args.attention_memory_training,
        },
        # lr=tune.grid_search(args.lr_grid),
        gamma=tune.grid_search(args.gamma_grid),
        # step = iteration * 4000
        lr_schedule=tune.grid_search([
            [[0, 5.0e-5], [4000 * 100, 5.0e-5], [4000 * 200, 1.0e-5]],
            # [[0, 0.001], [1e9, 0.0005]],
        ]),
    )
    '''
    #use tune to test different lr_schedule
        lr_schedule: tune.grid_search([
        [[0, 0.01], [1e6, 0.00001]],
        [[0, 0.001], [1e9, 0.0005]],
    ]),
    '''
    #config["lr_schedule"]=[[0, 5e-5],[400000, 3e-5],[1200000, 1e-5]]


    #stop = {"training_iteration": 100, "episode_reward_mean": 300}
    # config['model']['fcnet_hiddens'] = [32, 32]
    # automated run with Tune and grid search and TensorBoard

    '''
    If  don’t specify a scheduler, Tune will use a first-in-first-out (FIFO) scheduler by default, 
    which simply passes through the trials selected by your search algorithm in the order they were 
    picked and does not perform any early stopping.
    '''
    tuner = tune.Tuner(
        args.run,
        #param_space=config.to_dict(),
        param_space=config,
        run_config=air.RunConfig(
                                name='AiQMapping',
                                stop=stop,
                                checkpoint_config=air.CheckpointConfig(
                                checkpoint_frequency=args.checkpoint_frequency,
                                checkpoint_at_end=args.checkpoint_at_end,
                               ),
                                #storage_path=str(args.storage_path)
        ),
        tune_config=tune.TuneConfig(
            metric="env_runners/episode_reward_mean",
            mode="max",
            # num_samples=5,
            trial_name_creator=trial_str_creator,
            trial_dirname_creator=trial_str_creator,
        ),

    )
    results = tuner.fit()
    return results
def trial_str_creator(trial):
    return "{}_{}".format(trial.trainable_name, trial.trial_id)
def train():
    print('train:')
    output = StringIO()
    original_stdout = sys.stdout
    try:
        # Redirect stdout to the StringIO object
        sys.stdout = output
        result = train_policy()
        evaluate_policyv2(result)
        # Get the output from the StringIO object
        captured_output = output.getvalue()
        #write to file
        wirte2file(captured_output)
        analysis_res(result)
    finally:
        # Revert stdout back to the original
        sys.stdout = original_stdout

    tensorboard = parse_tensorboard(captured_output)
    print(f'tensorboard: {tensorboard}')
    copy_folder(tensorboard, args_pri.tensorboard_dir + args.time_id)


def wirte2file(content):
    text_path =  Path(get_root_dir())/ 'data' / 'result' / (str(args.stop_iters) + '_' + args.time_id + '.txt')
    write(text_path, content)


if __name__ == '__main__':
    register_custom_env(args.env_version)
    try:
        ray.init(num_gpus=args.num_gpus, local_mode=args.local_mode)
        time.sleep(1)
        train()
        ray.shutdown()
    except Exception as e:
        print(e)
    finally:
        ray.shutdown()
