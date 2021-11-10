import sys
sys.path.insert(0, './')

from rlf.algos.hier.option_critic import OptionCritic
from rlf import run_policy
from tests.test_run_settings import TestRunSettings
from rlf.policies.options_policy import OptionsPolicy

class OptionCriticRunSettings(TestRunSettings):
    def get_policy(self):
        return OptionsPolicy()

    def get_algo(self):
        return OptionCritic()

if __name__ == "__main__":
    run_policy(OptionCriticRunSettings())
