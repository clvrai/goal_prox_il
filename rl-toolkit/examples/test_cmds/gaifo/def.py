import sys
sys.path.insert(0, './')
from rlf import run_policy
from tests.test_run_settings import TestRunSettings
from rlf.policies.actor_critic.dist_actor_critic import DistActorCritic
from rlf.rl.model import MLPBase
from rlf import GAIFO
from rlf.rl.model import ConcatLayer
import torch.nn as nn

def get_discrim(in_shape, ac_dim):
    return nn.Sequential(
        ConcatLayer(-1),
        nn.Linear(2*in_shape[0], 400), nn.Tanh(),
        nn.Linear(400, 300), nn.Tanh(),
        nn.Linear(300, 1))


class GaifoRunSettings(TestRunSettings):
    def get_policy(self):
        return DistActorCritic(
                get_actor_fn=lambda _, i_shape: MLPBase(i_shape[0], False, (400, 300)),
                get_critic_fn=lambda _, i_shape, a_shape: MLPBase(i_shape[0], False, (400, 300)))

    def get_algo(self):
        return GAIFO(get_discrim=get_discrim)

if __name__ == "__main__":
    run_policy(GaifoRunSettings())
