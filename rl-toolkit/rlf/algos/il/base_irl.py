from rlf.algos.il.base_il import BaseILAlgo
import torch
from functools import *
import operator
import numpy as np
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import rlf.rl.utils as rutils

class BaseIRLAlgo(BaseILAlgo):
    def __init__(self):
        super().__init__()
        self.traj_log_stats = defaultdict(list)

    def init(self, policy, args):
        super().init(policy, args)
        self.ep_log_vals = defaultdict(lambda:
                deque(maxlen=args.log_smooth_len))
        self.culm_log_vals = defaultdict(lambda:
                [0.0 for _ in range(args.num_processes)])

    @abstractmethod
    def _get_reward(self, step, storage, add_info):
        pass

    def _update_reward_func(self, storage):
        return {}

    def update(self, storage):
        super().update(storage)
        # CLEAR ALL REWARDS so no environment rewards can leak to the IRL method.
        for step in range(self.args.num_steps):
            storage.rewards[step] = 0

        log_vals = self._update_reward_func(storage)
        add_info = {k: storage.get_add_info(k) for k in storage.get_extract_info_keys()}
        for k in storage.ob_keys:
            if k is not None:
                add_info[k] = storage.obs[k]

        for step in range(self.args.num_steps):
            rewards, ep_log_vals  = self._get_reward(step, storage, add_info)

            ep_log_vals['reward'] = rewards
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

        return log_vals

    def on_traj_finished(self, trajs):
        pass
