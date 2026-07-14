"""Evaluate distilled student policy with encoder disabled.

True inference: runs rollouts with with_encoder=False, with_vae=False, vae_noise=0.
This is the only metric that reflects deployment performance, since training-time
mean_rewards always use with_encoder=True (encoder has access to teacher obs).

Usage:
    python dexplore/eval_distill.py \
        --task Dexplore_Distill --distill \
        --cfg_env dexplore/data/cfg/inspire_distill.yaml \
        --cfg_train dexplore/data/cfg/train/rlg/inspire_distill.yaml \
        --checkpoint checkpoint/dex-ZNlong/nn/dex-ZNlong_00001400.pth \
        --headless --num_envs 64 \
        --output eval_results/ZNlong_e1400.json
"""
import os
import sys
import json

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from isaacgym import gymapi  # noqa: must precede torch

import numpy as np
import torch

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task

from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from learning import dexplore_agent_distill
from learning import dexplore_models
from learning import dexplore_network_builder_student
from learning import dexplore_players


def parse_eval_args():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=str, default='eval_results.json')
    parser.add_argument('--n_episodes_per_env', type=int, default=2,
                        help='target # episodes per env (averaging over resets)')
    parser.add_argument('--with_encoder', action='store_true',
                        help='keep encoder on (default: off — pure student eval)')
    parser.add_argument('--stochastic', action='store_true',
                        help='sample vae_noise (default: zero) and action (default: mu)')
    eval_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return eval_args


class DistillEvalPlayer(dexplore_players.DexplorePlayerContinuous):
    """Rollout player for the distill student, encoder-disabled by default."""

    def __init__(self, config):
        config = dict(config)
        config['normalize_amp_input'] = False  # distill checkpoints don't save AMP stats
        super().__init__(config)
        self.episodes = []
        self.use_encoder = False
        self.stochastic = False
        self.n_episodes_per_env = 2
        self.output_file = 'eval_results.json'

    def restore(self, fn):
        """Load student weights + running_mean_std; skip AMP-specific keys."""
        if fn == 'Base':
            return
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    @torch.no_grad()
    def run(self):
        if hasattr(self.env.task, '_adaptive_kappa_enabled'):
            self.env.task._adaptive_kappa_enabled = False

        obs, expert = self.env.reset()
        device = self.device
        if isinstance(obs, dict):
            obs_tensor = obs['obs']
        else:
            obs_tensor = obs
        batch_size = obs_tensor.shape[0]
        target_episodes = batch_size * self.n_episodes_per_env

        cr = torch.zeros(batch_size, device=device)
        steps = torch.zeros(batch_size, device=device)
        done_indices = torch.empty(0, dtype=torch.long, device=device)

        print(f"[EVAL] starting: num_envs={batch_size}, target_episodes={target_episodes}, "
              f"with_encoder={self.use_encoder}, stochastic={self.stochastic}")

        step_count = 0
        while len(self.episodes) < target_episodes:
            # Reset envs that terminated in the previous step — matches the training loop.
            # Without this, task.reset_buf[done] stays at 1 and done envs never recover.
            if done_indices.numel() > 0:
                obs, expert = self.env.reset(done_indices)
                if isinstance(obs, dict):
                    obs_tensor = obs['obs']
                else:
                    obs_tensor = obs

            student_obs = expert['student_obs'].to(device)
            latent_dim = int(getattr(self.env.task, 'vae_latent_dim', 64))
            if self.stochastic:
                vae_noise = torch.randn(batch_size, latent_dim, device=device)
            else:
                vae_noise = torch.zeros(batch_size, latent_dim, device=device)

            input_dict = {
                'is_train': False,
                'obs': obs_tensor,
                'student_obs': student_obs,
                'vae_noise': vae_noise,
                'with_encoder': self.use_encoder,
                'with_vae': False,
            }
            mu, sigma = self.model.a2c_network.act(input_dict)
            if self.stochastic:
                action = mu + sigma * torch.randn_like(mu)
            else:
                action = mu
            action = torch.clamp(action, -1.0, 1.0)

            next_obs, reward, done, info, expert = self.env.step(action)
            if isinstance(next_obs, dict):
                obs_tensor = next_obs['obs']
            else:
                obs_tensor = next_obs
            reward = reward.to(device) if torch.is_tensor(reward) else torch.as_tensor(reward, device=device)
            done = done.to(device) if torch.is_tensor(done) else torch.as_tensor(done, device=device)

            cr += reward
            steps += 1
            step_count += 1

            done_idx = done.nonzero(as_tuple=False).squeeze(-1)
            if done_idx.numel() > 0:
                terminate = None
                if hasattr(self.env.task, '_terminate_buf'):
                    terminate = self.env.task._terminate_buf
                for i in done_idx.tolist():
                    early = bool(terminate[i].item() > 0) if terminate is not None else False
                    self.episodes.append({
                        'reward': float(cr[i].item()),
                        'steps': int(steps[i].item()),
                        'survived': (not early),
                    })
                cr = cr * (1.0 - done.float())
                steps = steps * (1.0 - done.float())

            done_indices = done_idx

            if step_count % 50 == 0:
                print(f"[EVAL] step={step_count} episodes_done={len(self.episodes)}/{target_episodes}",
                      flush=True)

            if step_count > 20000:
                print("[EVAL] hit 20000-step cap, stopping")
                break

        self._summarize()

    def _summarize(self):
        if not self.episodes:
            print("[EVAL] no episodes completed")
            return
        n = len(self.episodes)
        rewards = np.array([e['reward'] for e in self.episodes])
        lengths = np.array([e['steps'] for e in self.episodes])
        survived = np.array([e['survived'] for e in self.episodes], dtype=np.float32)

        summary = {
            'num_episodes': n,
            'with_encoder': self.use_encoder,
            'stochastic': self.stochastic,
            'success_rate': round(float(survived.mean()), 4),
            'mean_reward': round(float(rewards.mean()), 2),
            'std_reward': round(float(rewards.std()), 2),
            'median_reward': round(float(np.median(rewards)), 2),
            'mean_steps': round(float(lengths.mean()), 1),
            'median_steps': round(float(np.median(lengths)), 1),
        }

        print('=' * 60)
        print(f"DISTILL EVAL ({n} episodes)")
        print(f"  with_encoder = {summary['with_encoder']}")
        print(f"  stochastic   = {summary['stochastic']}")
        print(f"  Success Rate: {summary['success_rate']:.1%}")
        print(f"  Mean Reward:  {summary['mean_reward']:.2f} +/- {summary['std_reward']:.2f}")
        print(f"  Median Rwd:   {summary['median_reward']:.2f}")
        print(f"  Mean Steps:   {summary['mean_steps']:.1f}  (median {summary['median_steps']:.1f})")
        print('=' * 60)

        out_dir = os.path.dirname(self.output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump({'summary': summary, 'per_episode': self.episodes}, f, indent=2)
        print(f"Saved: {self.output_file}")


# --- env setup mirrors run.py ---
args = None
cfg = None
cfg_train = None


def create_rlgpu_env(**kwargs):
    from run import RLGPUEnv as _  # noqa: ensure registered
    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params, distill=args.distill)
    print(f'num_envs: {env.num_envs}, num_actions: {env.num_actions}, num_obs: {env.num_obs}')
    return env


from run import RLGPUEnv, RLGPUAlgoObserver  # noqa
vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    'vecenv_type': 'RLGPU',
})


def main():
    global args, cfg, cfg_train

    eval_args = parse_eval_args()
    set_np_formatting()
    args = get_args()

    args.play = True
    args.train = False
    args.test = True

    cfg, cfg_train, _ = load_cfg(args)
    cfg_train['params']['seed'] = set_seed(
        cfg_train['params'].get("seed", -1),
        cfg_train['params'].get("torch_deterministic", False),
    )

    if args.motion_file:
        cfg['env']['motion_file'] = args.motion_file
    cfg_train['params']['config']['train_dir'] = args.output_path

    algo_observer = RLGPUAlgoObserver()
    runner = Runner(algo_observer)

    def _make_player(**kwargs):
        p = DistillEvalPlayer(**kwargs)
        p.output_file = eval_args.output
        p.use_encoder = eval_args.with_encoder
        p.stochastic = eval_args.stochastic
        p.n_episodes_per_env = eval_args.n_episodes_per_env
        return p

    runner.algo_factory.register_builder(
        'dexplore', lambda **kwargs: dexplore_agent_distill.DexploreAgent(**kwargs))
    runner.player_factory.register_builder(
        'dexplore', lambda **kwargs: _make_player(**kwargs))
    runner.model_builder.model_factory.register_builder(
        'dexplore', lambda network, **kwargs: dexplore_models.ModelDexploreContinuous(network))
    runner.model_builder.network_factory.register_builder(
        'dexplore', lambda **kwargs: dexplore_network_builder_student.DexploreBuilder())

    runner.load(cfg_train)
    runner.reset()
    runner.run(vars(args))


if __name__ == '__main__':
    main()
