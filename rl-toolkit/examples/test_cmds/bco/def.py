import sys
sys.path.insert(0, './')
from rlf import BehavioralCloningFromObs
from rlf import BasicPolicy
from rlf import run_policy
from tests.test_run_settings import TestRunSettings
from rlf.policies.actor_critic.dist_actor_critic import DistActorCritic
from rlf.rl.model import MLPBase

class BcoRunSettings(TestRunSettings):
    def get_policy(self):
        return BasicPolicy(
                is_stoch=self.base_args.stoch_policy,
                get_base_net_fn=lambda i_shape, recurrent: MLPBase(
                    i_shape[0], False, (400, 300))
                )

    def get_algo(self):
        return BehavioralCloningFromObs()

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--stoch-policy', default=False, action='store_true')

if __name__ == "__main__":
    run_policy(BcoRunSettings())
