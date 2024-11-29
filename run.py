import sys
import time
from datetime import datetime
from io import StringIO

import psutil
from ray.rllib.algorithms import PPOConfig

from env.env_helper import  register_custom_env
from config import ConfigSingleton

import ray
from ray import air, tune
from ray.air.constants import TRAINING_ITERATION

from env.env_v12 import CircuitEnv_v12
from env.env_v13 import CircuitEnv_v13
from utils.common_utils import parse_tensorboard, copy_folder
from evaluate import evaluate_policy
from utils.results import analysis_res
from v2_run import wirte2file

args = ConfigSingleton().get_config()
args_pri = ConfigSingleton().get_config_private()
stop = {
    TRAINING_ITERATION: args.stop_iters,
    #NUM_ENV_STEPS_SAMPLED_LIFETIME: 1000,
}

# todo move to config.yml
env_config={
    'debug':False,
    #'name':'Env_1'
}
def train_policy():
    cpus  = psutil.cpu_count(logical=True)

    config = PPOConfig()\
    .environment(env=CircuitEnv_v13, env_config=env_config)\
    .framework('torch')\
    .rollouts(num_rollout_workers=int(cpus * 0.7), num_envs_per_worker=2)\
    .resources(num_gpus=args.num_gpus)\
    .training(
        model={
            # "fcnet_hiddens":args.fcnet_hiddens ,
            "fcnet_hiddens": tune.grid_search(args.fcnet_hiddens_grid), #args.fcnet_hiddens,
            "fcnet_activation": tune.grid_search(args.fcnet_activation),
            "use_attention": False,
        },

        #lr=tune.grid_search(args.lr_grid),
        gamma=tune.grid_search(args.gamma_grid),
        # step = iteration * 4000
        lr_schedule= tune.grid_search([
        [[0, 5.0e-5], [4000*100, 5.0e-5],[4000*200,1.0e-5]],
       # [[0, 0.001], [1e9, 0.0005]],
    ]),
    )
    '''
    #use tune to test different 
    '''
    #config["lr_schedule"]=[[0, 5e-5],[400000, 3e-5],[1200000, 1e-5]]

    tuner = tune.Tuner(
        args.run,
        param_space=config,
        run_config=air.RunConfig(stop=stop,
                                 verbose=0,
                                checkpoint_config=air.CheckpointConfig(
                                checkpoint_frequency=args.checkpoint_frequency,
                                checkpoint_at_end=args.checkpoint_at_end,
                                 )),
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
    output = StringIO()
    original_stdout = sys.stdout
    try:
        # Redirect stdout to the StringIO object
        sys.stdout = output
        results = train_policy()
        evaluate_policy(results)
        # Get the output from the StringIO object
        captured_output = output.getvalue()
        # write to filereassign_qxx_labels
        wirte2file(captured_output)
        analysis_res(results)
    except Exception as e:
        print(e)
    finally:
        # Revert stdout back to the original
        sys.stdout = original_stdout

    tensorboard = parse_tensorboard(captured_output)
    print(f'tensorboard: {tensorboard}')
    copy_folder(tensorboard, args_pri.tensorboard_dir + args.time_id)


if __name__ == '__main__':
    register_custom_env(args.env_version)
    # from ray.tune import register_env
    # def env_creator(env_config):
    #     return gym.make('Env_1')  # return an instance of your custom environment
    #
    # register_env("Env_1", env_creator)

    args = ConfigSingleton().get_config()
    try:
        ray.init(num_gpus=args.num_gpus, local_mode=args.local_mode)
        time.sleep(1)
        train()
        ray.shutdown()
    except Exception as e:
        print(e)
    finally:
        ray.shutdown()
