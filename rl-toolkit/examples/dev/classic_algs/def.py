import sys
sys.path.insert(0, './')
from rlf.algos.tabular.td_methods import TabularTdMethods
from rlf.algos.tabular.mc_methods import TabularMcMethods
from rlf import run_policy
from tests.test_run_settings import TestRunSettings
from rlf.policies.tabular.q_table import QTable
from rlf.rl.loggers.plt_logger import PltLogger
from rlf.rl.loggers.wb_logger import WbLogger

class ClassicAlgRunSettings(TestRunSettings):
    def get_policy(self):
        return QTable()

    def get_algo(self):
        if self.base_args.alg_type == 'td':
            return TabularTdMethods()
        elif self.base_args.alg_type == 'mc':
            return TabularMcMethods()
        else:
            raise ValueError(f"Unrecognized option {self.base_args.alg_type}")

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--alg-type', type=str, default='td')
        parser.add_argument('--wb', action='store_true', default=False)

    def get_logger(self):
        if self.base_args.wb:
            return WbLogger()
        else:
            return PltLogger(['avg_r'], '# Updates', ['Reward'], ['Cliff Walker'])

if __name__ == "__main__":
    run_policy(ClassicAlgRunSettings())
