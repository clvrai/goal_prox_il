import sys
sys.path.insert(0, './')
from rlf.rl.model import MLPBase, TwoLayerMlpWithAction
from rlf.algos.il.bc import BehavioralCloning
from rlf.algos.il.gail import GAIL
from rlf.algos.il.bco import BehavioralCloningFromObs
from rlf.algos.off_policy.ddpg import DDPG
from rlf.algos.on_policy.sarsa import SARSA
from rlf.algos.on_policy.reinforce import REINFORCE
from rlf.algos.on_policy.a2c import A2C
from rlf.algos.off_policy.q_learning import QLearning
from rlf.algos.on_policy.ppo import PPO
from rlf import BaseAlgo
from rlf.algos.il.gaifo import GAIFO
from rlf.policies.basic_policy import BasicPolicy
from rlf.policies.random_policy import RandomPolicy
from rlf.policies.dqn import DQN
from rlf.policies.actor_critic.reg_actor_critic import RegActorCritic
from rlf.policies.actor_critic.dist_actor_critic import DistActorCritic
from rlf.run_settings import RunSettings
from rlf import run_policy
from rlf.rl.loggers.wb_logger import WbLogger
from rlf.rl.loggers.base_logger import BaseLogger

DIST_ACTOR_CRITIC = ('ppo', 'a2c', 'reinforce', 'gail', 'gaifo')
REG_ACTOR_CRITIC = ('ddpg',)
NO_CRITIC = ('q_learn', 'sarsa')
BASIC_POLICY = ('bc','bco')
RND_POLICY = ('rnd',)


class DefaultRunSettings(RunSettings):
    def get_policy(self):
        alg = self.base_args.alg
        if alg in DIST_ACTOR_CRITIC:
            return DistActorCritic()
        elif alg in REG_ACTOR_CRITIC:
            # DDPG is hard to train, make some changes to the base actor
            if alg == 'ddpg' and self.base_args.env_name == 'Hopper-v3':
                return RegActorCritic(
                    get_actor_head_fn=lambda _, i_shape: MLPBase(
                        i_shape[0], False, (128, 128)),
                    get_critic_head_fn=lambda _, i_shape, a_space: TwoLayerMlpWithAction(
                        i_shape[0], (128, 128), a_space.shape[0])
                )

            if alg == 'ddpg' and self.base_args.env_name == 'MountainCarContinuous-v0':
                return RegActorCritic(
                    get_actor_head_fn=lambda _, i_shape: MLPBase(
                        i_shape[0], False, (400, 300)),
                    get_critic_head_fn=lambda _, i_shape, a_space: TwoLayerMlpWithAction(
                        i_shape[0], (400, 300), a_space.shape[0])
                )
            else:
                return RegActorCritic()
        elif alg in NO_CRITIC:
            return DQN()
        elif alg in BASIC_POLICY:
            return BasicPolicy()
        elif alg in RND_POLICY:
            return RandomPolicy()
        else:
            raise ValueError('Unrecognized alg for policy architecture')

    def get_algo(self):
        alg = self.base_args.alg
        if alg == 'ppo':
            return PPO()
        elif alg == 'a2c':
            return A2C()
        elif alg == 'reinforce':
            return REINFORCE()
        elif alg == 'q_learn':
            return QLearning()
        elif alg == 'sarsa':
            return SARSA()
        elif alg == 'ddpg':
            return DDPG()
        elif alg == 'bc':
            return BehavioralCloning()
        elif alg == 'gail':
            return GAIL()
        elif alg == 'gaifo':
            return GAIFO()
        elif alg == 'rnd':
            return BaseAlgo()
        elif alg == 'bco':
            return BehavioralCloningFromObs()
        else:
            raise ValueError('Unrecognized alg for optimizer')

    def get_logger(self):
        if self.base_args.no_wb:
            return BaseLogger()
        else:
            return WbLogger()

    def get_config_file(self):
        return './tests/config.yaml'

    def get_add_args(self, parser):
        parser.add_argument('--alg')
        parser.add_argument('--no-wb', default=False, action='store_true')
        parser.add_argument('--env-name')


run_policy(DefaultRunSettings())
