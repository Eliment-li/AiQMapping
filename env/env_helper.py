from gymnasium import register
import gymnasium as gym
#测试 env 是否能正常工作
def check_env(env_id,entry_point):
    register(
        id=env_id,
        #entry_point='env.env_v1:CircuitEnv_v1',
        entry_point=entry_point,
        max_episode_steps=1000,
    )

    # Create the env to do inference in.
    env = gym.make(env_id)
    obs, info = env.reset()
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(obs, reward, terminated, truncated, info)


check_env('Env_1','env.env_v1:CircuitEnv_v1')
