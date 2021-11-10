import operator
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from functools import *

import numpy as np
import rlf.rl.utils as rutils
import torch
from rlf.algos.il.base_il import BaseILAlgo
from rlf.storage import RolloutStorage


class BaseIRLAlgo(BaseILAlgo):
    def __init__(self):
        super().__init__()
        self.traj_log_stats = defaultdict(list)

    def init(self, policy, args):
        super().init(policy, args)
        self.ep_log_vals = defaultdict(lambda: deque(maxlen=args.log_smooth_len))
        self.culm_log_vals = defaultdict(
            lambda: [0.0 for _ in range(args.num_processes)]
        )

    @abstractmethod
    def _get_reward(self, state, next_state, action, mask, add_info):
        pass

    def _update_reward_func(self, storage):
        return {}

    def _infer_rollout_storage_reward(self, storage, log_vals):
        add_info = {k: storage.get_add_info(k) for k in storage.get_extract_info_keys()}
        for k in storage.ob_keys:
            if k is not None:
                add_info[k] = storage.obs[k]

        for step in range(self.args.num_steps):
            mask = storage.masks[step]
            state = self._trans_agent_state(storage.get_obs(step))
            next_state = self._trans_agent_state(storage.get_obs(step + 1))
            action = storage.actions[step]
            add_inputs = {k: v[(step + 1) - 1] for k, v in add_info.items()}

            rewards, ep_log_vals = self._get_reward(
                state, next_state, action, mask, add_inputs
            )

            ep_log_vals["reward"] = rewards
            storage.rewards[step] = rewards

            for i in range(self.args.num_processes):
                for k in ep_log_vals:
                    self.culm_log_vals[k][i] += ep_log_vals[k][i].item()

                if storage.masks[step, i] == 0.0:
                    for k in ep_log_vals:
                        self.ep_log_vals[k].append(self.culm_log_vals[k][i])
                        self.culm_log_vals[k][i] = 0.0

        for k, vals in self.ep_log_vals.items():
            log_vals[f"culm_irl_{k}"] = np.mean(vals)

    def update(self, storage):
        super().update(storage)
        is_rollout_storage = isinstance(storage, RolloutStorage)
        if is_rollout_storage:
            # CLEAR ALL REWARDS so no environment rewards can leak to the IRL method.
            for step in range(self.args.num_steps):
                storage.rewards[step] = 0

        log_vals = self._update_reward_func(storage)

        if is_rollout_storage:
            self._infer_rollout_storage_reward(storage, log_vals)
        else:

            def get_reward(states, actions, next_states, mask):
                return self._get_reward(states, next_states, actions, mask, {})[0]

            storage.set_modify_reward_fn(get_reward)

        return log_vals

    def on_traj_finished(self, trajs):
        pass
