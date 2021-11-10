import sys
sys.path.insert(0, './')
import os
import os.path as osp

import torch
import numpy as np
import argparse
import string
import random
import datetime

from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
import pybullet_envs  # noqa: F401  # pylint: disable=unused-import
import pybulletgym
from garage.torch import set_gpu_mode
from garage.sampler import LocalSampler
from garage.sampler import VecWorker
from garage.sampler import DefaultWorker
from garage.sampler import MultiprocessingSampler
from garage.envs import GymEnv, normalize

from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.replay_buffer import PathBuffer
from garage.torch.optimizers import OptimizerWrapper
from torch import nn
from torch.nn import functional as F
from rlf.garage.auto_arg import convert_to_args, convert_kwargs
from rlf.args import str2bool
from rlf.exp_mgr import config_mgr

from dowel import logger
from rlf.garage.wb_logger import WbOutput
from rlf.garage.std_logger import StdLogger

def setup_def_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n-epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--env-name', type=str, required=True)
    parser.add_argument('--prefix', type=str, default='debug')
    parser.add_argument('--env-norm', type=str2bool, default=False)
    parser.add_argument('--cuda', type=str2bool, default=False)
    parser.add_argument('--no-wb', action='store_true', default=False)
    parser.add_argument('--alg', type=str, required=True)
    return parser


def ppo_args(parser):
    convert_to_args(PPO, parser)
    parser.add_argument('--policy-lr', type=float, default=3e-4)
    parser.add_argument('--vf-lr', type=float, default=3e-4)
    parser.add_argument('--n-minibatches', type=float, default=10)
    parser.add_argument('--minibatch-size', type=float, default=None)


def ppo_setup(env, trainer, args):
    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[args.hidden_dim]*args.depth,
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=[args.hidden_dim]*args.depth,
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    algo = PPO(env_spec=env.spec,
            policy=policy,
            value_function=value_function,
            policy_optimizer=OptimizerWrapper(
                (torch.optim.Adam, dict(lr=args.policy_lr)),
                policy,
                max_optimization_epochs=args.n_minibatches,
                minibatch_size=args.minibatch_size),
            vf_optimizer=OptimizerWrapper(
                (torch.optim.Adam, dict(lr=args.vf_lr)),
                value_function,
                max_optimization_epochs=args.n_minibatches,
                minibatch_size=args.minibatch_size),
            **convert_kwargs(args, PPO))
    trainer.setup(algo, env, sampler_cls=LocalSampler, worker_class=VecWorker,
            worker_args={'n_envs': 8})
    return algo


def sac_args(parser):
    convert_to_args(SAC, parser)
    parser.add_argument('--buffer-size', type=float, default=1e6)
    parser.add_argument('--gradient-steps-per-itr', type=int, default=1000)


def sac_setup(env, trainer, args):
    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[args.hidden_dim]*args.depth,
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[args.hidden_dim]*args.depth,
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[args.hidden_dim]*args.depth,
                                 hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(args.buffer_size))

    sac = SAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              **convert_kwargs(args, SAC))

    trainer.setup(algo=sac, env=env, sampler_cls=LocalSampler)
    return sac


USE_FNS = {
        'ppo': (ppo_args, ppo_setup),
        'sac': (sac_args, sac_setup),
        }

def get_env_id(args):
    upper_case = [c for c in args.env_name if c.isupper()]
    if len(upper_case) == 0:
        return ''.join([word[0] for word in args.env_name.split(".")])
    else:
        return ''.join(upper_case)

def create_prefix(args):
    assert args.prefix is not None and args.prefix != '', 'Must specify a prefix'
    d = datetime.datetime.today()
    date_id = '%i%i' % (d.month, d.day)
    env_id = get_env_id(args)
    rnd_id = ''.join(random.sample(
        string.ascii_uppercase + string.digits, k=2))
    before = ('%s-%s-%s-%s-' %
              (date_id, env_id, args.seed, rnd_id))

    if args.prefix != 'debug' and args.prefix != 'NONE':
        prefix = before + args.prefix
        print('Assigning full prefix %s' % prefix)
    else:
        prefix = args.prefix
    return prefix

def setup_launcher():
    config_dir = osp.dirname(osp.realpath(__file__))
    config_path = osp.join(config_dir, 'config.yaml')
    config_mgr.init(config_path)
    parser = setup_def_parser()

    # First parse the regular args
    base_args, _ = parser.parse_known_args()
    get_args, get_algo = USE_FNS[base_args.alg]
    use_prefix = create_prefix(base_args)

    @wrap_experiment(archive_launch_repo=False, snapshot_mode='none', name=use_prefix)
    def alg_train(ctxt=None):
        get_args(parser)
        args = parser.parse_args()
        args.prefix = use_prefix

        set_seed(args.seed)
        env = GymEnv(args.env_name)
        if args.env_norm:
            env = normalize(env)

        trainer = Trainer(ctxt)

        logger.remove_all()
        logger.add_output(StdLogger(args.log_interval))

        if not args.no_wb:
            wb_logger = WbOutput(args.log_interval, base_args)
            logger.add_output(wb_logger)

        algo = get_algo(env, trainer, args)

        if args.cuda:
            set_gpu_mode(True)
            algo.to()
        else:
            set_gpu_mode(False)

        trainer.train(n_epochs=args.n_epochs, batch_size=args.batch_size)

    return alg_train

launcher = setup_launcher()
launcher()
