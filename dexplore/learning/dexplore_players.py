"""Inference player for Dexplore trained policies."""
import time
import torch

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd

import learning.common_player as common_player


class DexplorePlayerContinuous(common_player.CommonPlayer):
    def __init__(self, config):
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        super().__init__(config)

    def run(self):
        """Run inference loop: rollout policy and collect episode statistics."""
        # Disable adaptive termination during inference (use fixed thresholds)
        if hasattr(self.env.task, '_adaptive_kappa_enabled'):
            self.env.task._adaptive_kappa_enabled = False

        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
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
            batch_size = self.get_batch_size(obs_dict['obs'], 1)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            print_game_res = False
            done_indices = []

            # Dataset playback mode (visualization only, no policy)
            if self.env.task.play_dataset:
                while True:
                    for t in range(self.env.task.max_episode_length.max()):
                        self.env.task.play_dataset_step(t)
            else:
                # Policy rollout mode
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

                    self._post_step(info)

                    if render:
                        self.env.render(mode='human')
                        time.sleep(self.render_sleep)

                    all_done_indices = done.nonzero(as_tuple=False)
                    done_indices = all_done_indices[::self.num_agents]
                    done_count = len(done_indices)
                    games_played += done_count

                    if done_count > 0:
                        if self.is_rnn:
                            for s in self.states:
                                s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        game_res = 0.0
                        if isinstance(info, dict):
                            if 'battle_won' in info:
                                print_game_res = True
                                game_res = info.get('battle_won', 0.5)
                            if 'scores' in info:
                                print_game_res = True
                                game_res = info.get('scores', 0.5)
                        if self.print_stats:
                            if print_game_res:
                                print('reward:', cur_rewards / done_count, 'steps:', cur_steps / done_count, 'w:', game_res)
                            else:
                                print('reward:', cur_rewards / done_count, 'steps:', cur_steps / done_count)

                        sum_game_res += game_res
                        if batch_size // self.num_agents == 1 or games_played >= n_games:
                            break

                    done_indices = done_indices[:, 0]

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life)

    def restore(self, fn):
        if fn != 'Base':
            super().restore(fn)
            if self._normalize_amp_input:
                checkpoint = torch_ext.load_checkpoint(fn)
                self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])

    def _build_net(self, config):
        super()._build_net(config)
        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config['amp_input_shape']).to(self.device)
            self._amp_input_mean_std.eval()

    def _post_step(self, info):
        super()._post_step(info)
        if self.env.task.viewer:
            self._amp_debug(info)

    def _build_net_config(self):
        config = super()._build_net_config()
        if hasattr(self, 'env'):
            config['amp_input_shape'] = self.env.amp_observation_space.shape
        else:
            config['amp_input_shape'] = self.env_info['amp_observation_space']
        return config

    def _amp_debug(self, info):
        pass
