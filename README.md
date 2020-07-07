# Offline Reinforcement Learning

This repository focuses on exploring algorithms in the domain of offline RL. Currently implemented are:
- DQN
- Ensemble-DQN
- QR-DQN
- Random Ensemble Mixture (REM) DQN
- LSPI

The original goal was to teach an agent how to play Atari games without interacting with the environment. As I had limited
time and resource constraints I changed scope to create a proof of concept on the classic Lunar Lander environment.
Subsequently the code in `atari_archive` is probably usable but deprecated in favor of the lunar lander section. I recommend
to be cautious.

This project occurred as part of the statistics seminar "Reinforcement Learning" at LMU Munich. The accompanying presentation with
elaborated information is available in the `presentation` folder.

## How to use the repository

Clone the project:
```
git clone https://github.com/saiboxx/offline-reinforcement-learning.git
```
I recommend to create an own python virtual environment and activate it:
```
cd offline-reinforcement-learning
python -m venv .venv
source .venv/bin/activate
```
To install necessary packages run:
```
make requirements
```

In total the project offers three functionalities:
- `make lunar-generate`: Employ a Random policy or an DQN Agent to collect and sample a dataset from the environment.
- `make lunar-train`: Use the previously collected data to train an agent in an offline manner.
- `make lunar-inference`: Use the saved offline model to launch a inference run in multiple environments.

## Parameters

Most parameters are collected in the `config.yml` file, which will be loaded to memory. The parameters behave as following:

- RENDER: Boolean whether to render the env or not.
- STEPS: Number of steps, where data will be collected.
- VERBOSE_STEPS: Ever n steps a status will be printed.
- WARM_UP_STEPS: Number of steps where the buffer will be filled with random actions on start up
- SUMMARY_PATH: Directory of logs


- GEN_DATA_PATH: Directory where collected data will be saved.

- AGENT: Agent to train (DQN, ENSEMBLE, REM, ...)
- TRAIN_DATA_PATH: Path pointing to training data
- EPOCHS: Epochs to train
- BATCH_SIZE: Chosen batch size
- EVAL_EPISODES: Number of environment episodes to run for evaluation after each epoch.
- EVAL_RENDER: Boolean for rendering the evaluation episodes.

- LEARNING_RATE: Learning rate
- GAMMA: Discount factor
- NUM_HEADS: Number of heads for multi-head architectures
- TARGET_UPDATE_INTERVAL: Interval where the target net will be updated with the policy parameters.
- SUMMARY_CHECKPOINT: Interval where data will be logged

- NUM_ENVS: Number of environments to spawn in inference mode
- INF_AGENT: Agent for inference mode
- INF_MODEL: Path to model file