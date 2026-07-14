import math
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from isaacgym.torch_utils import *

from learning.dexplore_agent import DexploreAgent as DexploreAgentBase

# Object state offset in teacher obs (1442D): humanoid_obs=646, task_obs starts at 646
# task_obs = local_tar_pos(3) + local_tar_rot_obs(6) + ...
OBJ_STATE_START = 646
OBJ_STATE_END = 655  # pos(3) + rot(6) = 9D


class DexploreAgent(DexploreAgentBase):
    """Distillation agent that extends DexploreAgent with DAgger and VAE support."""

    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        self.expert_loss_coef = 1.0
        num_envs = self.vec_env.env.task.num_envs
        task = self.vec_env.env.task
        self.vae_latent_dim = int(getattr(task, 'vae_latent_dim', 64))
        self.vae_noise = torch.zeros(num_envs, self.vae_latent_dim, dtype=torch.float, device=self.device)
        self.prior_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.aux_loss_coef = task.aux_loss_coef
        self.kld_floor = task.kld_floor
        self.kld_max = task.kld_max
        self.kld_ramp_start = task.kld_ramp_start
        self.kld_ramp_epochs = task.kld_ramp_epochs
        self.dagger_start_epoch = task.dagger_start_epoch
        self.dagger_decay_epochs = task.dagger_decay_epochs
        self.zero_vae_noise = getattr(task, 'zero_vae_noise', False)
        # Inspire hand: 12 real action dims (6 wrist + 6 finger); dims [7,9,11,13,16,17] are mimic slaves
        # and should not contribute to e_loss (they're teacher output noise on dead channels).
        self.mask_mimic_dims = getattr(task, 'mask_mimic_dims', False)
        self.real_action_dims = torch.tensor([0,1,2,3,4,5,6,8,10,12,14,15], dtype=torch.long, device=self.device)
        # Contact-weighted e_loss: upweight frames where fingers are in contact
        # (critical grasping moments). w = 1 + contact_loss_alpha * contact.sum(-1)
        self.contact_loss_alpha = float(getattr(task, 'contact_loss_alpha', 0.0))

    def init_tensors(self):
        super().init_tensors()
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['expert_mask'] = torch.zeros(batch_shape, dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['expert'] = torch.zeros((*batch_shape, 18), dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['vae_noise'] = torch.zeros((*batch_shape, self.vae_latent_dim), dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['student_obs'] = torch.zeros((*batch_shape, 2102), dtype=torch.float32, device=self.ppo_device)
        self.tensor_list += ['expert', 'expert_mask', 'vae_noise', 'student_obs']

    def play_steps(self):
        self.set_eval()
        update_list = self.update_list
        beta_t = min(1, max(1 - ((self.epoch_num - self.dagger_start_epoch) / max(1, self.dagger_decay_epochs)), 0))

        # Update wrist delta masking schedule (cosine annealing: 1.0 → 0.0 over epochs 500–5000)
        progress = min(max(0, self.epoch_num - 100) / 2000, 1.0)
        keep_prob = 1.0 - (1 - math.cos(progress * math.pi)) / 2
        self.vec_env.env.task.distill_keep_prob = keep_prob

        for n in range(self.horizon_length):
            self.obs, self.expert = self.env_reset(self.done_indices)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('expert', n, torch.clamp(self.expert['mus'].to(self.ppo_device), -1.0, 1.0))
            self.experience_buffer.update_data('student_obs', n, self.expert['student_obs'].to(self.ppo_device))

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(
                    self.obs, self._rand_action_probs, beta_t,
                    self.expert['actions'].to(self.ppo_device),
                    self.expert['student_obs'].to(self.ppo_device),
                )

            if not self.zero_vae_noise:
                self.experience_buffer.update_data('vae_noise', n, res_dict['vae_noise'])
            self.experience_buffer.update_data('expert_mask', n, res_dict['expert_mask'])

            # Rollout-time student/teacher mu gap, identical formula to _supervise_loss.
            # Comparable to training e_loss. Logged once per play_steps (n == 0).
            if n == 0:
                with torch.no_grad():
                    tmu_roll = torch.clamp(self.expert['mus'].to(self.ppo_device), -1.0, 1.0)
                    smu_roll = res_dict['mus']
                    per_sample = ((smu_roll - tmu_roll) ** 2).sum(dim=-1)
                    roll_eL = per_sample.mean().item()
                    cos = torch.nn.functional.cosine_similarity(smu_roll, tmu_roll, dim=-1).mean().item()
                    print(
                        f"[ROLL e{self.epoch_num}] roll_eL={roll_eL:.4f} "
                        f"cos(smu,tmu)={cos:.4f} "
                        f"||smu||={smu_roll.norm(dim=-1).mean().item():.3f} "
                        f"||tmu||={tmu_roll.norm(dim=-1).mean().item():.3f} "
                        f"beta={beta_t:.3f}",
                        flush=True)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos, self.expert = self.env_step(res_dict['actions'])
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

            if self.vec_env.env.task.viewer:
                self._amp_debug(infos)

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

    def get_action_values(self, obs_dict, rand_action_probs, use_experts=0.0, expert=None, student_obs=None):
        processed_obs = self._preproc_obs(obs_dict['obs'])

        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': processed_obs,
            'rnn_states': self.rnn_states,
            'student_obs': student_obs.clone(),
            'vae_noise': self.vae_noise.clone(),
            'with_encoder': True,
            'with_vae': False,
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs_dict['states']
                input_dict = {'is_train': False, 'states': states}
                value = self.get_central_value(input_dict)
                res_dict['values'] = value

        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)

        rand_action_mask = torch.bernoulli(rand_action_probs)
        det_action_mask = rand_action_mask == 0.0
        res_dict['actions'][det_action_mask] = res_dict['mus'][det_action_mask]
        res_dict['rand_action_mask'] = rand_action_mask

        # DAgger: mix in expert actions
        num_envs = self.vec_env.env.task.num_envs
        expert_action_probs = torch.full((num_envs,), use_experts, dtype=torch.float32, device=self.ppo_device)
        expert_action_probs = torch.bernoulli(expert_action_probs)
        expert_mask = expert_action_probs == 1.0
        res_dict['actions'][expert_mask] = expert[expert_mask]
        res_dict['expert_mask'] = expert_action_probs
        res_dict['vae_noise'] = self.vae_noise.clone()

        return res_dict

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict['expert'] = batch_dict['expert']
        self.dataset.values_dict['expert_mask'] = batch_dict['expert_mask']
        self.dataset.values_dict['vae_noise'] = batch_dict['vae_noise']
        self.dataset.values_dict['student_obs'] = batch_dict['student_obs']

    def _supervise_loss(self, student, teacher):
        diff2 = (student - teacher) ** 2
        if self.mask_mimic_dims:
            diff2 = diff2.index_select(-1, self.real_action_dims)
        return {'expert_loss': diff2.sum(dim=-1)}

    def kl_loss(self, prior_outs, encoder_outs):
        return 0.5 * (
            prior_outs["logvar"]
            - encoder_outs["logvar"]
            + torch.exp(encoder_outs["logvar"]) / torch.exp(prior_outs["logvar"])
            + encoder_outs["mu"] ** 2 / torch.exp(prior_outs["logvar"])
            - 1
        )

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos, expert = self.vec_env.step(actions)

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos, expert
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos, expert

    def env_reset(self, env_ids=None):
        self.reset_vae_noise(env_ids)
        obs, expert = self.vec_env.reset(env_ids)
        obs = self.obs_to_tensors(obs)
        return obs, expert

    def reset_vae_noise(self, env_ids):
        num_envs = self.vec_env.env.task.num_envs
        vae_latent_dim = self.vae_latent_dim
        if env_ids is None:
            env_ids = torch.arange(num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.to(self.device)
        if self.zero_vae_noise:
            self.vae_noise[env_ids] = torch.zeros(env_ids.shape[0], vae_latent_dim, device=self.device)
        else:
            self.vae_noise[env_ids] = torch.randn(env_ids.shape[0], vae_latent_dim, device=self.device)

    def calc_gradients(self, input_dict):
        self.set_train()

        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        obs_batch = self._preproc_obs(input_dict['obs'])
        expert_mus = input_dict['expert']
        vae_noise = input_dict['vae_noise']
        student_obs = input_dict['student_obs']

        lr_mul = 1.0

        batch_dict = {
            'is_train': True,
            'prev_actions': input_dict['actions'],
            'obs': obs_batch,
            'student_obs': student_obs.clone(),
            'vae_noise': vae_noise.clone(),
            'with_encoder': True,
            'with_vae': True,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']
            state = res_dict['rnn_states']
            prior_outs, encoder_outs = state['prior_out'], state['encoder_out']

            vae_kld_loss = torch.mean(torch.sum(self.kl_loss(prior_outs, encoder_outs), dim=-1))
            ramp = min(max(0, self.epoch_num - self.kld_ramp_start) / max(1, self.kld_ramp_epochs), 1)
            kld_coeff = self.kld_floor + ramp * (self.kld_max - self.kld_floor)

            e_info = self._supervise_loss(mu, expert_mus)
            per_sample_e = e_info['expert_loss']  # (B,)
            if self.contact_loss_alpha > 0.0:
                # local_obs layout: dof_pos(15) + contact(5) + ..., so contact = student_obs[:, 15:20]
                contact = student_obs[:, 15:20].sum(dim=-1).clamp(min=0.0)
                w = 1.0 + self.contact_loss_alpha * contact
                e_loss = (per_sample_e * w).sum() / w.sum().clamp(min=1e-6)
            else:
                e_loss = per_sample_e.mean()

            # Auxiliary reconstruction loss: predict object pos(3) + rot(6) from VAE latent
            aux_pred = state['aux_pred']
            aux_target = obs_batch[:, OBJ_STATE_START:OBJ_STATE_END].detach()
            aux_loss = F.mse_loss(aux_pred, aux_target)

            loss = self.expert_loss_coef * e_loss + vae_kld_loss * kld_coeff + self.aux_loss_coef * aux_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        if not torch.isfinite(loss):
            print(f"[Agent] Skipping backward: loss is {loss.item()}")
            return
        self.scaler.scale(loss).backward()

        if not hasattr(self, '_dbg_step'):
            self._dbg_step = 0
        self._dbg_step += 1
        if self._dbg_step % 50 == 0:
            with torch.no_grad():
                pm = prior_outs['mu']; em = encoder_outs['mu']
                pv = prior_outs['logvar']; ev = encoder_outs['logvar']
                vae_noise_in = batch_dict['vae_noise']
                tmu = expert_mus
                smu = mu
                def n(x): return x.norm(dim=-1).mean().item()
                def s(x): return x.std(dim=0).mean().item()
                enc_head_w = self.model.a2c_network._mu_head_2[-1].weight
                trunk_first_w = self.model.a2c_network.actor_mlp[0].weight
                mu_head_w = self.model.a2c_network.mu.weight
                enc_grad = enc_head_w.grad.norm().item() if enc_head_w.grad is not None else -1
                trunk_grad = trunk_first_w.grad.norm().item() if trunk_first_w.grad is not None else -1
                mu_grad = mu_head_w.grad.norm().item() if mu_head_w.grad is not None else -1

                # Per-sample sum-of-squares error
                per_sample = ((smu - tmu) ** 2).sum(dim=-1)        # (B,)
                per_dim_mse = ((smu - tmu) ** 2).mean(dim=0)       # (18,)

                # Dead-baseline (always predict zero): best scalar baseline
                baseline_zero = (tmu ** 2).sum(dim=-1).mean().item()
                # Dead-baseline (predict mean of teacher over the batch)
                baseline_mean = ((tmu - tmu.mean(dim=0, keepdim=True)) ** 2).sum(dim=-1).mean().item()

                # Cosine similarity ignoring scale
                cos = torch.nn.functional.cosine_similarity(smu, tmu, dim=-1).mean().item()

                # Percentiles of per-sample loss
                qs = torch.quantile(per_sample, torch.tensor([0.25, 0.5, 0.75, 0.95, 0.99], device=per_sample.device))

                # Correlation of per-sample error with ||tmu|| (do big-action samples dominate?)
                tnorm = tmu.norm(dim=-1)
                corr = torch.corrcoef(torch.stack([per_sample, tnorm]))[0, 1].item()

                print(
                    f"[DBG e{self.epoch_num} step{self._dbg_step}] "
                    f"eL={e_loss.item():.4f} klL={vae_kld_loss.item():.4f} auxL={aux_loss.item():.4f} "
                    f"||smu||={n(smu):.3f} std(smu)={s(smu):.3f} "
                    f"||tmu||={n(tmu):.3f} std(tmu)={s(tmu):.3f} "
                    f"||pm||={n(pm):.3f} std(pm)={s(pm):.3f} "
                    f"||em||={n(em):.3f} std(em)={s(em):.3f} "
                    f"pv=[{pv.mean().item():.2f},{pv.std().item():.2f}] "
                    f"ev=[{ev.mean().item():.2f},{ev.std().item():.2f}] "
                    f"||noise||={n(vae_noise_in):.3f} "
                    f"grad(enc_head)={enc_grad:.2e} grad(trunk0)={trunk_grad:.2e} grad(mu_head)={mu_grad:.2e}",
                    flush=True)
                print(
                    f"[DBG2 e{self.epoch_num}] "
                    f"baseZero={baseline_zero:.4f} baseMean={baseline_mean:.4f} "
                    f"cos(smu,tmu)={cos:.4f} corr(err,|tmu|)={corr:.3f} "
                    f"pctile[25,50,75,95,99]=[{qs[0].item():.3f},{qs[1].item():.3f},{qs[2].item():.3f},{qs[3].item():.3f},{qs[4].item():.3f}]",
                    flush=True)
                print(
                    f"[DBG3 e{self.epoch_num}] per_dim_mse="
                    + ",".join(f"{v:.3f}" for v in per_dim_mse.tolist()),
                    flush=True)
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
            'kl_loss': vae_kld_loss * kld_coeff,
            'aux_loss': aux_loss.detach(),
        }
        self.train_result.update(e_info)

    def _eval_critic(self, obs_dict):
        self.model.eval()
        obs = obs_dict['obs']
        processed_obs = self._preproc_obs(obs)
        value = self.model.a2c_network.eval_critic(processed_obs)
        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value

    def _build_net_config(self):
        config = super()._build_net_config()
        config['input_shape'] = (1442,)
        return config

    def _log_train_info(self, train_info, frame):
        self.writer.add_scalar('performance/update_time', train_info['update_time'], frame)
        self.writer.add_scalar('performance/play_time', train_info['play_time'], frame)
        self.writer.add_scalar('losses/e_loss', torch_ext.mean_list(train_info['expert_loss']).item(), frame)
        self.writer.add_scalar('losses/kl_loss', torch_ext.mean_list(train_info['kl_loss']).item(), frame)
        self.writer.add_scalar('losses/entropy', torch_ext.mean_list(train_info['entropy']).item(), frame)
        self.writer.add_scalar('info/last_lr', train_info['last_lr'][-1] * train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/lr_mul', train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/e_clip', self.e_clip * train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/kl', torch_ext.mean_list(train_info['kl']).item(), frame)
        self.writer.add_scalar('usage/cpu', self.get_cpu_usage(), frame)
        self.writer.add_scalar('usage/gpu', self.get_gpu_usage(), frame)
        self.writer.add_scalar('usage/cpu_memory', self.get_cpu_memory_usage(), frame)
        self.writer.add_scalar('usage/gpu_memory', self.get_gpu_memory_usage(), frame)
