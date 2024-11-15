import sys
import time
from io import StringIO

import numpy as np
import psutil
from ray.rllib.algorithms import PPOConfig
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_LIFETIME

from env.env_helper import  register_custom_env
from env.env_v10 import CircuitEnv_v10
from config import ConfigSingleton

import ray
from ray import air, tune
from ray.air.constants import TRAINING_ITERATION
from ray.tune.registry import get_trainable_cls

from env.env_v12 import CircuitEnv_v12
from utils.common_utils import parse_tensorboard
from utils.evaluate import evaluate_policy
from utils.results import analysis_res
from v2_run import wirte2file

args = ConfigSingleton().get_config()

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
    # config = (
    #     get_trainable_cls(args.run)
    #     .get_default_config()
    #     .environment(env=CircuitEnv_v10,env_config=env_config)
    #     .framework('torch')
    #     .rollouts(num_rollout_workers=int(cpus*0.7)
    #               , num_envs_per_worker=2
    #               # ,remote_worker_envs=True
    #               )
    #     .resources(num_gpus=args.num_gpus)
    #     .training(
    #         #lr = tune.grid_search([0.001, 0.01, 0.1, 1.0]),
    #         model={
    #             # Change individual keys in that dict by overriding them, e.g.
    #             #"fcnet_hiddens":args.fcnet_hiddens ,
    #             #"fcnet_hiddens": [32,64,128,64,32],
    #             "fcnet_activation":args.fcnet_activation,
    #             "use_attention": False,
    #         },
    #         gamma=0.99,
    #     )
    # )

    config = PPOConfig()\
    .environment(env=CircuitEnv_v12, env_config=env_config)\
    .framework('torch')\
    .rollouts(num_rollout_workers=int(cpus * 0.7), num_envs_per_worker=2)\
    .resources(num_gpus=args.num_gpus)\
    .training(
        model={
            # "fcnet_hiddens":args.fcnet_hiddens ,
            "fcnet_hiddens": [128,256,256,128],
            "fcnet_activation": args.fcnet_activation,
            "use_attention": False,
        },
        #lr=tune.grid_search([5e-5, 3e-5,1e-5]),
        gamma=0.99,
    )
    '''
    #use tune to test different lr_schedule
        lr_schedule: tune.grid_search([
        [[0, 0.01], [1e6, 0.00001]],
        [[0, 0.001], [1e9, 0.0005]],
    ]),
    '''
    config["lr_schedule"]=[[0, 5e-5],[400000, 3e-5],[1200000, 1e-5]]

    tuner = tune.Tuner(
        args.run,
        param_space=config,
        run_config=air.RunConfig(stop=stop,
                                checkpoint_config=air.CheckpointConfig(
                                checkpoint_frequency=args.checkpoint_frequency,
                                checkpoint_at_end=args.checkpoint_at_end,
                                 ))
    )
    results = tuner.fit()
    return results

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

    finally:
        # Revert stdout back to the original
        sys.stdout = original_stdout

    tensorboard = parse_tensorboard(captured_output)
    print(f'tensorboard: {tensorboard}')

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
