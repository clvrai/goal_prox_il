import sys
sys.path.insert(0, './')
from rlf import PPO
from rlf import run_policy
from tests.test_run_settings import TestRunSettings
from rlf.policies.actor_critic.dist_actor_critic import DistActorCritic
from rlf.envs.env_interface import EnvInterfaceWrapper, register_env_interface
from rlf.envs.minigrid_interface import MinigridInterface
from rlf.algos.nested_algo import NestedAlgo
from abc import ABC, abstractmethod
import gym
from enum import IntEnum
import rlf.rl.utils as rutils
from rlf import BaseNetPolicy
from rlf.rl.multi_runner import MultiRunner


class ForwardStorage:
    Data = ['TEST!']
    @staticmethod
    def sample():
        return ForwardStorage.Data[0]

class PlayMode(IntEnum):
    REAL = 0
    BACK = 1


class DoublePlaybackWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._starting = None
        self._play_mode = PlayMode.REAL
        #self.observation_space = rutils.combine_spaces(
        #        self.observation_space,
        #        "mode", gym.spaces.Discrete(2))

    def step(self, a):
        obs, reward, done, info = super().step(a)
        #info['mode'] = self._play_mode
        print('Sampling!', ForwardStorage.sample())
        if done:
            if self._play_mode == PlayMode.REAL:
                self._play_mode = PlayMode.BACK
            elif self._play_mode == PlayMode.BACK:
                self._play_mode = PlayMode.REAL
        return self._mod_obs(obs), reward, done, info

    def _mod_obs(self, obs):
        #obs = rutils.combine_obs(obs, "mode", self._play_mode)
        return obs

    @abstractmethod
    def _set_state(self, obs):
        pass

    def _get_cur_state(self, obs):
        return obs

    def reset(self):
        obs = super().reset()
        #if self._play_mode == PlayMode.BACK:
        #    self._set_state(self._starting)
        self._starting = self._get_cur_state(obs)
        return self._mod_obs(obs)

class MinigridPlaybackWrapper(DoublePlaybackWrapper):
    def _set_state(self, obs):
        # Extract the agent and goal positions
        grid, start, direction = obs
        self.env.env.grid = grid
        self.env.env.agent_pos = start
        self.env.env.agent_dir = direction

    def _get_cur_state(self, obs):
        return (self.env.grid, self.env.agent_pos, self.env.agent_dir)


class DoublePlaybackEnvInterface(EnvInterfaceWrapper):
    def __init__(self, args):
        env = super().__init__(args, MinigridInterface)

    def create_from_id(self, env_id):
        env = super().create_from_id(env_id)
        env = MinigridPlaybackWrapper(env)
        return env

register_env_interface("^MiniGrid", DoublePlaybackEnvInterface)

#class DoublePlayUpdater(NestedAlgo):
#    def __init__(self, updater1, updater2):
#        super().__init__([updater1, updater2], 0)
#
#    def get_storage_buffer(self, policy, envs, args):
#        return SwitchingNestedStorage(
#                self.modules[0].get_storage_buffer(policy, envs, args),
#                self.modules[1].get_storage_buffer(policy, envs, args))
#
#
#class SwitchingNestedStorage(NestedStorage):
#    def __init__(self, main_storage, back_storage):
#        super().__init__({
#            'main': main_storage,
#            'back': back_storage
#            }, 'main')
#
#
#class DoublePolicy(BasePolicy):
#    def __init__(self, policy1, policy2):
#        super().__init__()
#        self.policy1 = policy1
#        self.policy2 = policy2
#        self.policy1.set_arg_prefix('A')
#        self.policy2.set_arg_prefix('B')
#
#    def init(self, obs_space, action_space, args):
#        super().init(obs_space, action_space, args)
#        self.policy1.init(obs_space, action_space, args)
#        self.policy2.init(obs_space, action_space, args)
#
#    def get_storage_hidden_states(self):
#        return self.policy1.get_storage_hidden_states()
#
#
#    def get_action(self, state, add_state, hxs, masks, step_info):
#        pass
#
#    def get_add_args(self, parser):
#        self.policy1.get_add_args(parser)
#        self.policy2.get_add_args(parser)
#
#    def load_from_checkpoint(self, checkpointer):
#        self.policy1.load_from_checkpoint(checkpointer)
#        self.policy2.load_from_checkpoint(checkpointer)
#
#    def save_to_checkpoint(self, checkpointer):
#        self.policy1.save_to_checkpoint(checkpointer)
#        self.policy2.save_to_checkpoint(checkpointer)
#
#    def load_resume(self, checkpointer):
#        self.policy1.load_resume(checkpointer)
#        self.policy2.load_resume(checkpointer)
#
#    def set_env_ref(self, envs):
#        self.policy1.set_env_ref(envs)
#        self.policy2.set_env_ref(envs)
#
#    def to(self, device):
#        self.policy1 = self.policy1.to(envs)
#        self.policy2 = self.policy2.to(envs)
#        return self
#
#    def eval(self):
#        self.policy1.eval()
#        self.policy2.eval()
#
#    def train(self):
#        self.policy1.train()
#        self.policy2.train()


class SimpleRunSettings(TestRunSettings):
    def __init__(self, is_forward, use_log=None, use_args=None):
        super().__init__()
        self.is_forward = is_forward
        self.use_log = use_log
        self.use_args = use_args

    def _get_env_interface(self, args, task_id=None):
        if not self.is_forward:
            ei = DoublePlaybackEnvInterface(args)
            ei.setup(args, task_id)
            return ei
        return super()._get_env_interface(args)

    def get_policy(self):
        return DistActorCritic()

    def _sys_setup(self, add_args, ray_create):
        if not self.is_forward:
            return self.use_args, self.use_log
        return _sys_setup(add_args, ray_create)

    def get_algo(self):
        return PPO()

    def get_add_args(self, parser):
        super().get_add_args(parser)

frunsts = SimpleRunSettings(True)
frunner = frunsts.create_runner()

brunsts = SimpleRunSettings(False, frunner.log, frunner.args)
brunner = brunsts.create_runner()

class DualRunner(MultiRunner):
    def training_iter(self, update_iter):
        self.forward_runner.training_iter(update_iter)

if __name__ == "__main__":
    run_policy(PPORunSettings())


