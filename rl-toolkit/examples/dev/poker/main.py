import sys
sys.path.insert(0, './')
from rlf import QLearning
from rlf import DQN
from tests.test_run_settings import TestRunSettings
from rlf import run_policy
import rlf.envs.neuron_poker


class DqnRunSettings(TestRunSettings):
    def get_policy(self):
        return DQN()

    def get_algo(self):
        return QLearning()

    def get_add_args(self, parser):
        super().get_add_args(parser)

if __name__ == "__main__":
    run_policy(DqnRunSettings())
