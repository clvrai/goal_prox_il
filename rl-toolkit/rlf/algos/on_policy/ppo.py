import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from rlf.algos.on_policy.on_policy_base import OnPolicy

class PPO(OnPolicy):
    def update(self, rollouts):
        self._compute_returns(rollouts)
        advantages = rollouts.compute_advantages()

        use_clipped_value_loss = True

        log_vals = defaultdict(lambda: 0)

        for e in range(self._arg('num_epochs')):
            data_generator = rollouts.get_generator(advantages,
                    self._arg('num_mini_batch'))

            for sample in data_generator:
                # Get all the data from our batch sample
                ac_eval = self.policy.evaluate_actions(sample['state'],
                        sample['other_state'],
                        sample['hxs'], sample['mask'],
                        sample['action'])

                ratio = torch.exp(ac_eval['log_prob'] - sample['prev_log_prob'])
                surr1 = ratio * sample['adv']
                surr2 = torch.clamp(ratio,
                        1.0 - self._arg('clip_param'),
                        1.0 + self._arg('clip_param')) * sample['adv']
                action_loss = -torch.min(surr1, surr2).mean(0)

                if use_clipped_value_loss:
                    value_pred_clipped = sample['value'] + (ac_eval['value'] - sample['value']).clamp(
                                    -self._arg('clip_param'),
                                    self._arg('clip_param'))
                    value_losses = (ac_eval['value'] - sample['return']).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - sample['return']).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (sample['return'] - ac_eval['value']).pow(2).mean()

                loss = (value_loss * self._arg('value_loss_coef') + action_loss -
                     ac_eval['ent'].mean() * self._arg('entropy_coef'))

                self._standard_step(loss)

                log_vals['value_loss'] += value_loss.sum().item()
                log_vals['action_loss'] += action_loss.sum().item()
                log_vals['dist_entropy'] += ac_eval['ent'].mean().item()

        num_updates = self._arg('num_epochs') * self._arg('num_mini_batch')
        for k in log_vals:
            log_vals[k] /= num_updates

        return log_vals

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument(f"--{self.arg_prefix}clip-param",
            type=float,
            default=0.2,
            help='ppo clip parameter')

        parser.add_argument(f"--{self.arg_prefix}entropy-coef",
            type=float,
            default=0.01,
            help='entropy term coefficient (old default: 0.01)')

        parser.add_argument(f"--{self.arg_prefix}value-loss-coef",
            type=float,
            default=0.5,
            help='value loss coefficient')

