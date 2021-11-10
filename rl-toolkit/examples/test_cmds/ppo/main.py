import sys
sys.path.insert(0, './')
from rlf import PPO
from rlf import run_policy
from tests.test_run_settings import TestRunSettings
from rlf.policies.actor_critic.dist_actor_critic import DistActorCritic

class PPORunSettings(TestRunSettings):
    def get_policy(self):
        return DistActorCritic()

    def get_algo(self):
        return PPO()

if __name__ == "__main__":
    run_policy(PPORunSettings())
