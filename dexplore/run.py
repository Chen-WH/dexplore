"""Main entry point for Dexplore training and inference."""

import os

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task

from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

import numpy as np
import torch

from learning import dexplore_agent
from learning import dexplore_agent_distill
from learning import dexplore_players
from learning import dexplore_models
from learning import dexplore_network_builder
from learning import dexplore_network_builder_student

args = None
cfg = None
cfg_train = None


def create_rlgpu_env(**kwargs):
    """Create a GPU-accelerated RL environment, configuring multi-GPU via Horovod if enabled."""
    use_horovod = cfg_train['params']['config'].get('multi_gpu', False)
    if use_horovod:
        import horovod.torch as hvd
        rank = hvd.rank()
        print("Horovod rank: ", rank)

        cfg_train['params']['seed'] = cfg_train['params']['seed'] + rank
        args.device = 'cuda'
        args.device_id = rank
        args.rl_device = 'cuda:' + str(rank)
        cfg['rank'] = rank
        cfg['rl_device'] = 'cuda:' + str(rank)

    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params, distill=args.distill)

    print(f'num_envs: {env.num_envs}')
    print(f'num_actions: {env.num_actions}')
    print(f'num_obs: {env.num_obs}')
    print(f'num_states: {env.num_states}')

    return env


class RLGPUAlgoObserver(AlgoObserver):
    """Tracks and logs consecutive success metrics to TensorBoard during training."""

    def __init__(self, use_successes=True):
        self.use_successes = use_successes

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if not self.use_successes and 'consecutive_successes' in infos:
                cons_successes = infos['consecutive_successes'].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and 'successes' in infos:
                successes = infos['successes'].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))

    def after_clear_stats(self):
        self.mean_scores.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.consecutive_successes.current_size > 0:
            mean_con_successes = self.consecutive_successes.get_mean()
            self.writer.add_scalar('successes/consecutive_successes/mean', mean_con_successes, frame)
            self.writer.add_scalar('successes/consecutive_successes/iter', mean_con_successes, epoch_num)
            self.writer.add_scalar('successes/consecutive_successes/time', mean_con_successes, total_time)


class RLGPUEnv(vecenv.IVecEnv):
    """Vectorized environment wrapper that bridges IsaacGym tasks with rl_games."""

    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        self.use_global_obs = (self.env.num_states > 0)
        self.distill = getattr(args, 'distill', False)

        self.full_state = {}
        if self.distill:
            self.full_state["obs"], _ = self.reset()
        else:
            self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()

    def step(self, action):
        if self.distill:
            next_obs, reward, is_done, info, expert = self.env.step(action)
        else:
            next_obs, reward, is_done, info = self.env.step(action)

        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            obs_out = self.full_state
        else:
            obs_out = self.full_state["obs"]

        if self.distill:
            return obs_out, reward, is_done, info, expert
        return obs_out, reward, is_done, info

    def reset(self, env_ids=None):
        if self.distill:
            self.full_state["obs"], expert = self.env.reset(env_ids)
        else:
            self.full_state["obs"] = self.env.reset(env_ids)

        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            obs_out = self.full_state
        else:
            obs_out = self.full_state["obs"]

        if self.distill:
            return obs_out, expert
        return obs_out

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {
            'action_space': self.env.action_space,
            'observation_space': self.env.observation_space,
            'amp_observation_space': self.env.amp_observation_space,
        }
        if self.use_global_obs:
            info['state_space'] = self.env.state_space
        return info


vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    'vecenv_type': 'RLGPU',
})


def build_alg_runner(algo_observer, distill=False):
    """Construct an rl_games Runner and register Dexplore agent, player, model, and network."""
    runner = Runner(algo_observer)

    if distill:
        agent_cls = dexplore_agent_distill.DexploreAgent
        network_cls = dexplore_network_builder_student.DexploreBuilder
    else:
        agent_cls = dexplore_agent.DexploreAgent
        network_cls = dexplore_network_builder.DexploreBuilder

    runner.algo_factory.register_builder('dexplore', lambda **kwargs: agent_cls(**kwargs))
    runner.player_factory.register_builder('dexplore', lambda **kwargs: dexplore_players.DexplorePlayerContinuous(**kwargs))
    runner.model_builder.model_factory.register_builder('dexplore', lambda network, **kwargs: dexplore_models.ModelDexploreContinuous(network))
    runner.model_builder.network_factory.register_builder('dexplore', lambda **kwargs: network_cls())

    return runner


def main():
    global args, cfg, cfg_train

    # Load configuration and set up reproducibility
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)

    cfg_train['params']['seed'] = set_seed(
        cfg_train['params'].get("seed", -1),
        cfg_train['params'].get("torch_deterministic", False),
    )

    # Override config values with command-line arguments when provided
    if args.horovod:
        cfg_train['params']['config']['multi_gpu'] = args.horovod
    if args.horizon_length != -1:
        cfg_train['params']['config']['horizon_length'] = args.horizon_length
    if args.minibatch_size != -1:
        cfg_train['params']['config']['minibatch_size'] = args.minibatch_size
    if args.motion_file:
        cfg['env']['motion_file'] = args.motion_file
    if args.play_dataset:
        cfg['env']['playdataset'] = True
    if args.projtype:
        cfg['env']['projtype'] = args.projtype
    if args.cg1 != -1.:
        cfg['env']['rewardWeights']['cg1'] = args.cg1
    if args.cg2 != -1.:
        cfg['env']['rewardWeights']['cg2'] = args.cg2
    if args.ig != -1.:
        cfg['env']['rewardWeights']['ig'] = args.ig
    if args.op != -1.:
        cfg['env']['rewardWeights']['op'] = args.op
    if args.save_images:
        cfg['env']['saveImages'] = True
    if args.init_vel:
        cfg['env']['initVel'] = True
    if args.frames_scale != 0.:
        cfg['env']['dataFramesScale'] = args.frames_scale
    if args.ball_size != 0.:
        cfg['env']['ballSize'] = args.ball_size

    cfg_train['params']['config']['train_dir'] = args.output_path

    # Build the runner and launch training or inference
    algo_observer = RLGPUAlgoObserver()
    runner = build_alg_runner(algo_observer, distill=args.distill)
    runner.load(cfg_train)
    runner.reset()
    runner.run(vars(args))


if __name__ == '__main__':
    main()
