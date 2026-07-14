"""PPO training agent with reference-scoped exploration for Dexplore."""

import time
import psutil
import subprocess

import numpy as np
import torch
from torch import nn

from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from isaacgym.torch_utils import *

import learning.common_agent as common_agent


class DexploreAgent(common_agent.CommonAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)

        if self._normalize_input:
            self._input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)
        self.resume_from = config['resume_from']
        self.done_indices = []

    def train(self):
        if self.resume_from != 'None':
            try:
                self.restore(self.resume_from)
            except Exception:
                print('Failed to restore from checkpoint')
        super().train()

    def init_tensors(self):
        super().init_tensors()
        self._build_rand_action_probs()
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['rand_action_mask'] = torch.zeros(batch_shape, dtype=torch.float32, device=self.ppo_device)
        self.tensor_list += ['amp_obs', 'rand_action_mask']

    def set_eval(self):
        super().set_eval()
        if self._normalize_input:
            self._input_mean_std.eval()

    def set_train(self):
        super().set_train()
        if self._normalize_input:
            self._input_mean_std.train()

    def get_stats_weights(self):
        state = super().get_stats_weights()
        if self._normalize_input:
            state['amp_input_mean_std'] = self._input_mean_std.state_dict()
        return state

    def set_stats_weights(self, weights):
        super().set_stats_weights(weights)
        if self._normalize_input:
            self._input_mean_std.load_state_dict(weights['amp_input_mean_std'])

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if self._normalize_input:
            self._input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
        self.set_full_state_weights(checkpoint)

    def play_steps(self):
        # Collect rollout experience with epsilon-greedy reference-scoped exploration.
        self.set_eval()
        update_list = self.update_list

        for n in range(self.horizon_length):
            self.obs = self.env_reset(self.done_indices)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs, self._rand_action_probs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])

            # Sanitize invalid observations
            invalid_obs = ~torch.isfinite(self.obs['obs'])
            invalid_batches = torch.any(invalid_obs, dim=1)
            if torch.any(invalid_obs):
                print("invalid observation")
                print(torch.where(invalid_obs))
                self.obs['obs'][invalid_batches] = 0
                self.dones[invalid_batches] = True
                infos['terminate'][invalid_batches] = True

            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            self.experience_buffer.update_data('rand_action_mask', n, res_dict['rand_action_mask'])

            terminated = infos['terminate'].float().unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            self.done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[self.done_indices])
            self.game_lengths.update(self.current_lengths[self.done_indices])
            self.algo_observer.process_infos(infos, self.done_indices)

            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            self.done_indices = self.done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        return batch_dict

    def get_action_values(self, obs_dict, rand_action_probs):
        # Forward pass with stochastic/deterministic action selection based on rand_action_probs.
        processed_obs = self._preproc_obs(obs_dict['obs'])

        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': processed_obs,
            'rnn_states': self.rnn_states,
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs_dict['states']
                input_dict = {
                    'is_train': False,
                    'states': states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value

        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)

        rand_action_mask = torch.bernoulli(rand_action_probs)
        det_action_mask = rand_action_mask == 0.0
        res_dict['actions'][det_action_mask] = res_dict['mus'][det_action_mask]
        res_dict['rand_action_mask'] = rand_action_mask

        return res_dict

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict['rand_action_mask'] = batch_dict['rand_action_mask']

    def train_epoch(self):
        play_time_start = time.time()

        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None

        if self.is_rnn:
            frames_mask_ratio = rnn_masks.sum().item() / rnn_masks.nelement()
            print(frames_mask_ratio)

        for _ in range(self.mini_epochs_num):
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])

                if self.schedule_type == 'legacy':
                    if self.multi_gpu:
                        curr_train_info['kl'] = self.hvd.average_value(curr_train_info['kl'], 'ep_kls')
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, curr_train_info['kl'].item())
                    self.update_lr(self.last_lr)

                if train_info is None:
                    train_info = {k: [v] for k, v in curr_train_info.items()}
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)

            av_kls = torch_ext.mean_list(train_info['kl'])

            if self.schedule_type == 'standard':
                if self.multi_gpu:
                    av_kls = self.hvd.average_value(av_kls, 'ep_kls')
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

        if self.schedule_type == 'standard_epoch':
            if self.multi_gpu:
                av_kls = self.hvd.average_value(torch_ext.mean_list(train_info['kl']), 'ep_kls')
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
            self.update_lr(self.last_lr)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        train_info['play_time'] = play_time
        train_info['update_time'] = update_time
        train_info['total_time'] = total_time
        self._record_train_batch_info(batch_dict, train_info)

        return train_info

    def calc_gradients(self, input_dict):
        # Compute PPO gradients with rand_action_mask weighting.
        self.set_train()

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = self._preproc_obs(input_dict['obs'])

        rand_action_mask = input_dict['rand_action_mask']
        rand_action_sum = torch.sum(rand_action_mask)

        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']
            a_clipped = a_info['actor_clipped'].float()

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            b_loss = self.bound_loss(mu)

            c_loss = torch.mean(c_loss)
            a_loss = torch.sum(rand_action_mask * a_loss) / rand_action_sum
            entropy = torch.sum(rand_action_mask * entropy) / rand_action_sum
            b_loss = torch.sum(rand_action_mask * b_loss) / rand_action_sum
            a_clip_frac = torch.sum(rand_action_mask * a_clipped) / rand_action_sum

            loss = a_loss + self.critic_coef * c_loss + self.bounds_loss_coef * b_loss

            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac
            c_info['critic_loss'] = c_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        if not torch.isfinite(loss):
            print(f"[Agent] Skipping backward: loss is {loss.item()}")
            return
        self.scaler.scale(loss).backward()
        if self.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            if self.multi_gpu:
                self.optimizer.synchronize()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()

        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr,
            'lr_mul': lr_mul,
            'b_loss': b_loss,
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)

    def _load_config_params(self, config):
        super()._load_config_params(config)
        self._enable_eps_greedy = bool(config['enable_eps_greedy'])
        self._amp_observation_space = self.env_info['amp_observation_space']
        self._normalize_input = config.get('normalize_input', True)

    def _build_net_config(self):
        config = super()._build_net_config()
        config['amp_input_shape'] = self._amp_observation_space.shape
        return config

    def _build_rand_action_probs(self):
        # Build per-env epsilon-greedy probabilities (exponential decay across env IDs).
        num_envs = self.vec_env.env.task.num_envs
        env_ids = to_torch(np.arange(num_envs), dtype=torch.float32, device=self.ppo_device)

        self._rand_action_probs = 1.0 - torch.exp(10 * (env_ids / (num_envs - 1.0) - 1.0))
        self._rand_action_probs[0] = 1.0
        self._rand_action_probs[-1] = 0.0

        if not self._enable_eps_greedy:
            self._rand_action_probs[:] = 1.0

    def _calc_advs(self, batch_dict):
        # Compute advantages with rand_action_mask normalization.
        returns = batch_dict['returns']
        values = batch_dict['values']
        rand_action_mask = batch_dict['rand_action_mask']

        advantages = returns - values
        advantages = torch.sum(advantages, axis=1)
        if self.normalize_advantage:
            advantages = torch_ext.normalization_with_masks(advantages, rand_action_mask)

        return advantages

    # --- System monitoring ---

    def get_cpu_usage(self):
        return psutil.cpu_percent(interval=1)

    def get_cpu_memory_usage(self):
        return psutil.virtual_memory().percent

    def get_gpu_usage(self):
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'], stdout=subprocess.PIPE)
        return int(result.stdout.decode().strip().split()[0])

    def get_gpu_memory_usage(self):
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
        return int(result.stdout.decode().strip().split()[0])

    def _log_train_info(self, train_info, frame):
        self.writer.add_scalar('performance/update_time', train_info['update_time'], frame)
        self.writer.add_scalar('performance/play_time', train_info['play_time'], frame)
        self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(train_info['actor_loss']).item(), frame)
        self.writer.add_scalar('losses/c_loss', torch_ext.mean_list(train_info['critic_loss']).item(), frame)
        self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(train_info['b_loss']).item(), frame)
        self.writer.add_scalar('losses/entropy', torch_ext.mean_list(train_info['entropy']).item(), frame)
        self.writer.add_scalar('info/last_lr', train_info['last_lr'][-1] * train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/lr_mul', train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/e_clip', self.e_clip * train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/clip_frac', torch_ext.mean_list(train_info['actor_clip_frac']).item(), frame)
        self.writer.add_scalar('info/kl', torch_ext.mean_list(train_info['kl']).item(), frame)
        self.writer.add_scalar('usage/cpu', self.get_cpu_usage(), frame)
        self.writer.add_scalar('usage/gpu', self.get_gpu_usage(), frame)
        self.writer.add_scalar('usage/cpu_memory', self.get_cpu_memory_usage(), frame)
        self.writer.add_scalar('usage/gpu_memory', self.get_gpu_memory_usage(), frame)
