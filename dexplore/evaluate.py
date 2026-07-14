"""
Evaluate tracking success rate using the standard rl_games player pipeline.

This extends the DexplorePlayerContinuous to collect per-episode metrics
(reward, steps, survival, tracking errors) during inference.

Usage:
    python dexplore/evaluate.py \
        --task Dexplore_Inspire \
        --cfg_env dexplore/data/cfg/inspire_slow_slow_energy_reset_contact_table_adjust_parameter.yaml \
        --cfg_train dexplore/data/cfg/train/rlg/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2.yaml \
        --checkpoint robot/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB.pth \
        --headless --num_envs 64 --output eval_results.json
"""
import os
import sys
import json
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from isaacgym import gymapi  # noqa: must import before torch

import numpy as np
import torch

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task

from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

from learning import dexplore_agent
from learning import dexplore_players
from learning import dexplore_models
from learning import dexplore_network_builder


def parse_eval_args():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=str, default='eval_results.json')
    eval_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return eval_args


class EvalPlayer(dexplore_players.DexplorePlayerContinuous):
    """Extended player that collects per-episode metrics."""

    def __init__(self, config):
        super().__init__(config)
        self.episode_results = []

    def run(self):
        # Disable adaptive termination during evaluation (use fixed thresholds)
        if hasattr(self.env.task, '_adaptive_kappa_enabled'):
            self.env.task._adaptive_kappa_enabled = False

        n_games = self.games_num
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn

        for _ in range(n_games):
            if games_played >= n_games:
                break

            obs_dict = self.env_reset()
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            cum_hand_err = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            cum_obj_err = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            done_indices = []

            for n in range(self.max_steps):
                obs_dict = self.env_reset(done_indices)

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obs_dict, masks, is_determenistic)
                else:
                    action = self.get_action(obs_dict, is_determenistic)
                obs_dict, r, done, info = self.env_step(self.env, action)
                cr += r
                steps += 1

                # Collect tracking metrics
                if hasattr(self.env.task, 'metric_1'):
                    cum_hand_err += self.env.task.metric_1
                if hasattr(self.env.task, 'metric_2'):
                    cum_obj_err += self.env.task.metric_2

                self._post_step(info)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                    for idx in done_indices:
                        i = idx.item()
                        ep_len = max(steps[i].item(), 1)
                        early_term = False
                        if hasattr(self.env.task, '_terminate_buf'):
                            early_term = self.env.task._terminate_buf[i].item() > 0

                        self.episode_results.append({
                            'reward': cr[i].item(),
                            'steps': int(steps[i].item()),
                            'survived': not early_term,
                            'mean_hand_error': cum_hand_err[i].item() / ep_len,
                            'mean_obj_error': cum_obj_err[i].item() / ep_len,
                        })

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    cum_hand_err = cum_hand_err * (1.0 - done.float())
                    cum_obj_err = cum_obj_err * (1.0 - done.float())

                    if batch_size // self.num_agents == 1 or games_played >= n_games:
                        break

                    done_indices = done_indices[:, 0]

        # Print and save results at end of run
        if self.episode_results:
            n_survived = sum(1 for r in self.episode_results if r['survived'])
            total = len(self.episode_results)
            success_rate = n_survived / total
            mean_reward = np.mean([r['reward'] for r in self.episode_results])
            std_reward = np.std([r['reward'] for r in self.episode_results])
            mean_steps = np.mean([r['steps'] for r in self.episode_results])
            mean_hand_err = np.mean([r['mean_hand_error'] for r in self.episode_results])
            mean_obj_err = np.mean([r['mean_obj_error'] for r in self.episode_results])

            print(f"\n{'=' * 60}")
            print(f"EVALUATION RESULTS ({total} episodes)")
            print(f"{'=' * 60}")
            print(f"  Success Rate:    {success_rate:.1%}")
            print(f"  Mean Reward:     {mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"  Mean Steps:      {mean_steps:.1f}")
            print(f"  Mean Hand Error: {mean_hand_err:.4f}")
            print(f"  Mean Obj Error:  {mean_obj_err:.4f}")
            print(f"{'=' * 60}")

            # Save to file if output path set
            output_file = getattr(self, 'output_file', 'eval_results.json')
            summary = {
                'num_episodes': total,
                'success_rate': round(success_rate, 4),
                'mean_reward': round(float(mean_reward), 2),
                'std_reward': round(float(std_reward), 2),
                'mean_steps': round(float(mean_steps), 1),
                'mean_hand_error': round(float(mean_hand_err), 4),
                'mean_obj_error': round(float(mean_obj_err), 4),
            }
            output = {'summary': summary, 'per_episode': self.episode_results}
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Results saved to {output_file}")


# Store output path globally so player can access it
_eval_output_file = 'eval_results.json'

# ---- Same run.py infrastructure ----
args = None
cfg = None
cfg_train = None


def create_rlgpu_env(**kwargs):
    from run import RLGPUEnv as _  # ensure registered
    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)
    print(f'num_envs: {env.num_envs}, num_actions: {env.num_actions}, num_obs: {env.num_obs}')
    return env


# Import and re-register with our create fn
from run import RLGPUEnv, RLGPUAlgoObserver
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

    # Force test mode
    args.play = True
    args.train = False
    args.test = True

    cfg, cfg_train, logdir = load_cfg(args)
    cfg_train['params']['seed'] = set_seed(
        cfg_train['params'].get("seed", -1),
        cfg_train['params'].get("torch_deterministic", False))

    if args.motion_file:
        cfg['env']['motion_file'] = args.motion_file

    cfg_train['params']['config']['train_dir'] = args.output_path

    # Build runner with EvalPlayer instead of standard player
    algo_observer = RLGPUAlgoObserver()
    runner = Runner(algo_observer)
    runner.algo_factory.register_builder('dexplore', lambda **kwargs: dexplore_agent.DexploreAgent(**kwargs))
    def _make_eval_player(**kwargs):
        p = EvalPlayer(**kwargs)
        p.output_file = eval_args.output
        return p
    runner.player_factory.register_builder('dexplore', lambda **kwargs: _make_eval_player(**kwargs))
    runner.model_builder.model_factory.register_builder('dexplore', lambda network, **kwargs: dexplore_models.ModelDexploreContinuous(network))
    runner.model_builder.network_factory.register_builder('dexplore', lambda **kwargs: dexplore_network_builder.DexploreBuilder())

    runner.load(cfg_train)
    runner.reset()

    # The EvalPlayer.run() prints results and saves to eval_args.output internally
    runner.run(vars(args))


if __name__ == '__main__':
    main()
