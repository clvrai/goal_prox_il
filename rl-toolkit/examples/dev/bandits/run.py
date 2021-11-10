import sys
sys.path.insert(0, './')
from rlf import run_policy
from rlf.policies.tabular.action_value_policy import ActionValuePolicy
from rlf.algos.tabular.bandit_algos import SimpleBanditAlgo
from tests.test_run_settings import TestRunSettings
import gym_bandits

class BanditRunSettings(TestRunSettings):
    def get_policy(self):
        return ActionValuePolicy()

    def get_algo(self):
        return SimpleBanditAlgo()

run_policy(BanditRunSettings())
