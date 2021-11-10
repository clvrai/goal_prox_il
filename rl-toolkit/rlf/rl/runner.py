import contextlib
from typing import Any, Dict, Optional

import numpy as np
import torch
from rlf.algos.base_net_algo import BaseNetAlgo
from rlf.baselines.vec_env import VecEnvWrapper
from rlf.policies.base_policy import get_step_info
from rlf.rl import utils
from rlf.rl.envs import get_vec_normalize, make_vec_envs
from rlf.rl.evaluation import full_eval, train_eval


class Runner:
    """
    Trains a policy
    """

    def __init__(
        self, envs, storage, policy, log, env_interface, checkpointer, args, updater
    ):
        self.envs = envs
        self.storage = storage
        self.policy = policy
        self.log = log
        self.env_interface = env_interface
        self.checkpointer = checkpointer
        self.args = args
        self.updater = updater
        self.train_eval_envs = None

        if self.policy.requires_inference_grads():
            self.train_ctx = contextlib.nullcontext
        else:
            self.train_ctx = torch.no_grad

    def training_iter(self, update_iter: int) -> Dict[str, Any]:
        self.log.start_interval_log()
        self.updater.pre_update(update_iter)

        for step in self.updater.get_steps_generator(update_iter):
            # Sample actions
            obs = self.storage.get_obs(step)

            step_info = get_step_info(update_iter, step, self.episode_count, self.args)

            with self.train_ctx():
                ac_info = self.policy.get_action(
                    utils.get_def_obs(obs, self.args.policy_ob_key),
                    utils.get_other_obs(obs),
                    self.storage.get_hidden_state(step),
                    self.storage.get_masks(step),
                    step_info,
                )
                if self.args.clip_actions:
                    ac_info.clip_action(*self.ac_tensor)

            next_obs, reward, done, infos = self.envs.step(ac_info.take_action)

            reward += ac_info.add_reward

            step_log_vals = utils.agg_ep_log_stats(infos, ac_info.extra)

            self.episode_count += sum([int(d) for d in done])
            self.log.collect_step_info(step_log_vals)

            self.storage.insert(obs, next_obs, reward, done, infos, ac_info)
        updater_log_vals = self.updater.update(self.storage)

        self.storage.after_update()

        return updater_log_vals

    @property
    def should_start_with_eval(self) -> bool:
        """
        If true, will evaluate the policy before the main training loop begins.
        """
        return False

    def setup(self) -> None:
        """
        Runs before any evaluation or training.
        """
        self.episode_count = 0
        self.alg_env_settings = self.updater.get_env_settings(self.args)
        self.updater.first_train(self.log, self._eval_policy, self.env_interface)
        if self.args.clip_actions:
            self.ac_tensor = utils.ac_space_to_tensor(self.policy.action_space)

    def _eval_policy(self, policy, total_num_steps, args) -> Optional[VecEnvWrapper]:
        return train_eval(
            self.envs,
            self.alg_env_settings,
            policy,
            args,
            self.log,
            total_num_steps,
            self.env_interface,
            self.train_eval_envs,
        )

    def log_vals(self, updater_log_vals, update_iter):
        total_num_steps = self.updater.get_completed_update_steps(update_iter + 1)
        return self.log.interval_log(
            update_iter,
            total_num_steps,
            self.episode_count,
            updater_log_vals,
            self.args,
        )

    def save(self, update_iter: int) -> None:
        if (
            (self.episode_count > 0) or (self.args.num_steps == 0)
        ) and self.checkpointer.should_save():
            vec_norm = get_vec_normalize(self.envs)
            if vec_norm is not None:
                self.checkpointer.save_key("ob_rms", vec_norm.ob_rms_dict)
            self.checkpointer.save_key("step", update_iter)

            self.policy.save_to_checkpoint(self.checkpointer)
            self.updater.save(self.checkpointer)

            self.checkpointer.flush(num_updates=update_iter)
            if self.args.sync:
                self.log.backup(self.args, update_iter + 1)

    def eval(self, update_iter):
        if (
            (self.episode_count > 0)
            or (self.args.num_steps <= 1)
            or self.should_start_with_eval
        ):
            total_num_steps = self.updater.get_completed_update_steps(update_iter + 1)
            self.train_eval_envs = self._eval_policy(
                self.policy, total_num_steps, self.args
            )

    def close(self):
        self.log.close()
        if self.train_eval_envs is not None:
            self.train_eval_envs.close()
        self.envs.close()

    def resume(self):
        self.updater.load_resume(self.checkpointer)
        self.policy.load_resume(self.checkpointer)
        return self.checkpointer.get_key("step")

    def should_load_from_checkpoint(self):
        return self.checkpointer.should_load()

    def full_eval(self, create_traj_saver_fn):
        alg_env_settings = self.updater.get_env_settings(self.args)

        tmp_env = make_vec_envs(
            self.args.env_name,
            self.args.seed,
            1,
            self.args.gamma,
            self.args.device,
            False,
            self.env_interface,
            self.args,
            alg_env_settings,
            set_eval=False,
        )
        vec_norm = None
        if self.checkpointer.has_load_key("ob_rms"):
            ob_rms_dict = self.checkpointer.get_key("ob_rms")
            vec_norm = get_vec_normalize(tmp_env)
            if vec_norm is not None:
                vec_norm.ob_rms_dict = ob_rms_dict

        return full_eval(
            self.envs,
            self.policy,
            self.log,
            self.checkpointer,
            self.env_interface,
            self.args,
            alg_env_settings,
            create_traj_saver_fn,
            vec_norm,
        )

    def load_from_checkpoint(self):
        self.policy.load_state_dict(self.checkpointer.get_key("policy"))

        if self.checkpointer.has_load_key("ob_rms"):
            ob_rms_dict = self.checkpointer.get_key("ob_rms")
            vec_norm = get_vec_normalize(self.envs)
            if vec_norm is not None:
                vec_norm.ob_rms_dict = ob_rms_dict
        self.updater.load(self.checkpointer)
