from rlf.run_settings import RunSettings
from rlf.rl.loggers.wb_logger import WbLogger
from rlf.rl.loggers.tb_logger import TbLogger
from rlf.rl.loggers.plt_logger import PltLogger
from rlf.rl.loggers.base_logger import BaseLogger
import os.path as osp
from rlf.args import str2bool

class TestRunSettings(RunSettings):
    def get_logger(self):
        if self.base_args.tb:
            return TbLogger('./data/tb')
        elif self.base_args.plt:
            return PltLogger(["avg_r"], "Steps", "Reward", "Reward Curve")
        elif self.base_args.no_wb:
            return BaseLogger()
        else:
            return WbLogger()

    def get_config_file(self):
        # Path to testing config
        config_dir = osp.dirname(osp.realpath(__file__))
        return osp.join(config_dir, 'config.yaml')

    def get_add_args(self, parser):
        parser.add_argument('--no-wb', default=False, action='store_true')
        parser.add_argument('--tb', default=False, action='store_true')
        parser.add_argument('--plt', default=False, action='store_true')
        parser.add_argument('--env-name')

