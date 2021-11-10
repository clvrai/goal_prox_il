import sys
sys.path.insert(0, './')
from rlf import DDPG
from rlf import run_policy
from tests.test_run_settings import TestRunSettings
from rlf import RegActorCritic
from rlf.rl.model import MLPBase, TwoLayerMlpWithAction
import torch.nn as nn
import torch.nn.functional as F

def reg_init(m):
    return m

def get_actor_head(hidden_dim, action_dim):
    return nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh())

class DDPGRunSettings(TestRunSettings):
    def get_policy(self):
        if 'Pendulum' in self.base_args.env_name:
            hidden_size = 128
        else:
            hidden_size = 256
        return RegActorCritic(
                get_actor_fn=lambda _, i_shape: MLPBase(
                    i_shape[0], False, (hidden_size, hidden_size),
                    weight_init=reg_init,
                    get_activation=lambda: nn.ReLU()),
                get_actor_head_fn=get_actor_head,
                get_critic_fn=lambda _, i_shape, a_space: TwoLayerMlpWithAction(
                    i_shape[0], (hidden_size, hidden_size), a_space.shape[0],
                    weight_init=reg_init,
                    get_activation=lambda: nn.ReLU()),
                get_critic_head_fn = lambda hidden_dim: nn.Linear(hidden_dim, 1)
                )

    def get_algo(self):
        return DDPG()

if __name__ == "__main__":
    run_policy(DDPGRunSettings())
