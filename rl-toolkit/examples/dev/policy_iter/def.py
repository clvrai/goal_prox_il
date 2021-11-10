import sys
sys.path.insert(0, './')
from rlf import run_policy
from rlf.policies.tabular.tabular_policy import TabularPolicy
from rlf.algos.tabular.policy_iteration import PolicyIteration
from rlf.algos.tabular.value_iteration import ValueIteration
from tests.test_run_settings import TestRunSettings
from rlf.args import str2bool
from rlf.rl.loggers.plt_logger import PltLogger

class PolicyIterRunSettings(TestRunSettings):
    def get_policy(self):
        return TabularPolicy()

    def get_algo(self):
        if self.base_args.value_iter:
            return ValueIteration()
        else:
            return PolicyIteration()

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--value-iter', default=False, type=str2bool)

    def get_logger(self):
        return PltLogger(['eval_train_r'], '# Updates', ['Reward'], ['Frozen Lake'])



run_policy(PolicyIterRunSettings())
