import sys
sys.path.insert(0, './')
from rlf import run_policy
from tests.test_run_settings import TestRunSettings
from rlf.policies.actor_critic.dist_actor_q import DistActorQ, get_sac_actor, get_sac_critic
from rlf.algos.il.sqil import SQIL
from functools import partial

class SqilRunSettings(TestRunSettings):
    def get_policy(self):
        return DistActorQ(
				get_critic_fn=partial(get_sac_critic, hidden_dim=256),
                get_actor_fn=partial(get_sac_actor, hidden_dim=256))

    def get_algo(self):
        return SQIL()

if __name__ == "__main__":
    run_policy(SqilRunSettings())
