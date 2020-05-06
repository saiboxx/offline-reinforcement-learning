import os
import yaml
import time
import numpy as np
from typing import Tuple
from src.mlagents.environment import UnityEnvironment
from src.mlagents.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from src.agents import RandomAgent


def main():
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    run(cfg)


def run(cfg: dict):
    print("Loading environment {}.".format(cfg["EXECUTABLE"]))
    worker_id = np.random.randint(2000)
    env, config_channel = load_environment(cfg["EXECUTABLE"],
                                            cfg["NO_GRAPHICS"],
                                            worker_id)

    config_channel.set_configuration_parameters(time_scale=cfg["TIME_SCALE"])
    env.reset()

    group_name = env.get_agent_groups()[0]
    group_spec = env.get_agent_group_spec(group_name)
    action_space = group_spec.action_shape
    observation_space = group_spec.observation_shapes[0][0]
    step_result = env.get_step_result(group_name)
    state = step_result.obs[0]
    num_agents = len(state)

    print("Creating Agent.")
    agent = RandomAgent(observation_space, action_space)

    print("Starting training with {} steps.".format(cfg["STEPS"]))
    reward_cur_episode = np.zeros(num_agents)
    reward_last_episode = np.zeros(num_agents)
    rolling_reward_mean_episode = []
    start_time_episode = time.time()
    episode = 1

    start_time = time.time()
    for steps in range(1, cfg["STEPS"] + 1):
        action = agent.act(state)
        env.set_actions(group_name, action)
        env.step()
        step_result = env.get_step_result(group_name)
        new_state = step_result.obs[0]
        reward = step_result.reward
        done = step_result.done

        mean_step_reward = np.mean(reward)
        reward_cur_episode += reward

        if steps % cfg["VERBOSE_STEPS"] == 0:
            elapsed_time = time.time() - start_time
            print("Ep. {0:>4} with {1:>7} steps total; {2:8.2f} last ep. rewards; {3:+.3f} step reward; {4}h elapsed" \
                  .format(episode, steps, reward_mean_episode, mean_step_reward, format_timedelta(elapsed_time)))

        for i, d in enumerate(done):
            if d:
                reward_last_episode[i] = reward_cur_episode[i]
                if steps >= cfg["STEPS"] * 0.9:
                    rolling_reward_mean_episode.append(reward_cur_episode[i])
                reward_cur_episode[i] = 0

        if done[0]:
            reward_mean_episode = reward_last_episode.mean()
            duration_last_episode = time.time() - start_time_episode
            start_time_episode = time.time()
            episode += 1

        state = new_state

    print("Closing environment.")
    env.close()
    max_reward_mean_episode = np.mean(rolling_reward_mean_episode)
    return max_reward_mean_episode


def load_environment(env_name: str, no_graphics: bool, worker_id: int) \
        -> Tuple[UnityEnvironment, EngineConfigurationChannel]:
    """
    Loads a Unity environment with a given key name.
    """
    env_path = os.path.join("executables", env_name)
    files_in_dir = os.listdir(env_path)
    env_file = [os.path.join(env_path, f) for f in files_in_dir
                if os.path.isfile(os.path.join(env_path, f))][0]
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_file,
                           no_graphics=no_graphics,
                           worker_id=worker_id,
                           side_channels=[engine_configuration_channel])
    return env, engine_configuration_channel


def format_timedelta(timedelta):
    total_seconds = int(timedelta)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:0>2d}:{:0>2d}:{:0>2d}'.format(hours, minutes, seconds)


if __name__ == '__main__':
    main()