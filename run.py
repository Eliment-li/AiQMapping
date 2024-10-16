import datetime
import random
import time
from pprint import pprint
from ray import tune, air
import gymnasium as gym
import ray
from gymnasium import register
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.models.configs import ModelConfig

from config import ConfigSingleton


def evaluate_policy(checkpoint):
    algo = Algorithm.from_checkpoint(checkpoint.to_directory())
    env = gym.make('CartPole-v1')
    obs, info = env.reset()
    num_episodes = 0
    episode_reward = 0.0
    #    while num_episodes < args.num_episodes_during_inference:

    while num_episodes < 1:
        # Compute an action (`a`).
        a = algo.compute_single_action(
            observation=obs,
            explore=None,
            policy_id="default_policy",  # <- default value
        )
        # Send the computed action `a` to the env.
        obs, reward, done, truncated, info = env.step(a)
        print('done = %r, reward = %r \n' % (done, reward))
        episode_reward += reward

        # Is the episode `done`? -> Reset.
        if done:
            print('env done = %r, reward = %r \n obs = \n {%r} ' % (done, reward, obs))
            print(f"Episode done: Total reward = {episode_reward}")

            if not isinstance(checkpoint, str):
                checkpoint = checkpoint.path

            obs, info = env.reset()
            num_episodes += 1
            episode_reward = 0.0

        algo.stop()

# todo move to config.yml
env_config={
    'debug':False
}
def train_policy():
    config = (
        PPOConfig()
        .environment(env="CartPole-v1",env_config=env_config)
        # Switch both the new API stack flags to True (both False by default).
        # This enables the use of
        # a) RLModule (replaces ModelV2) and Learner (replaces Policy)
        # b) and automatically picks the correct EnvRunner (single-agent vs multi-agent)
        # and enables ConnectorV2 support.
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .resources(
            num_cpus_for_main_process=1,
        )
        # We are using a simple 1-CPU setup here for learning. However, as the new stack
        # supports arbitrary scaling on the learner axis, feel free to set
        # `num_learners` to the number of available GPUs for multi-GPU training (and
        # `num_gpus_per_learner=1`).
        .learners(
            num_learners=0,  # <- in most cases, set this value to the number of GPUs
            num_gpus_per_learner=0,  # <- set this to 1, if you have at least 1 GPU
        )
        # When using RLlib's default models (RLModules) AND the new EnvRunners, you should
        # set this flag in your model config. Having to set this, will no longer be required
        # in the near future. It does yield a small performance advantage as value function
        # predictions for PPO are no longer required to happen on the sampler side (but are
        # now fully located on the learner side, which might have GPUs available).
        .training(model={"uses_new_env_runners": True})
    )
    config['model']['use_attention'] = True
    #stop = {"training_iteration": 100, "episode_reward_mean": 300}
    # config['model']['fcnet_hiddens'] = [32, 32]
    # automated run with Tune and grid search and TensorBoard
    tuner = tune.Tuner(
        'PPO',
        param_space=config.to_dict(),
        run_config=air.RunConfig(),
    )
    results = tuner.fit()

    print("Training completed")
    return results

def train():
    results = train_policy()
    checkpoint = results.get_best_result().checkpoint
    evaluate_policy(checkpoint)

if __name__ == '__main__':
    register(
        id='Env_1',
        # entry_point='core.envs.circuit_env:CircuitEnv',
        entry_point='env.env_v1:CircuitEnv_v1',
        max_episode_steps=20000,
    )
    env = gym.make('Env_1')
    obs, info = env.reset()
    # args = ConfigSingleton().get_config()
    # try:
    #     ray.init(num_gpus=1, local_mode=args.local_mode)
    #     train()
    # except Exception as e:
    #     pprint(e)
