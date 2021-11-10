All plotted results are under the root README.

Example of running a command: `python -m rlf --cmd ddpg/pendulum --cd 0 --cfg ./tests/config.yaml --seed "31,41" --sess-id 0 --cuda False`

# Expert Datasets
All under `tests/expert_demonstrations`
* `halfcheetah_50ep.pt` 50 episodes (1k transitions each), 6678 average reward

# Directory Structure
- `dev` are under development and for my personal use `--cfg ./tests/config_dev.yaml`. 
- `test_cmds` are confirmed to work. To run commands specify `--cfg ./tests/config.yaml`. 

# Sanity Checking Algorithms
- Discrete: CartPole-v0 and get above 195.0 reward for over 100 episodes.
