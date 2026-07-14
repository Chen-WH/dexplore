from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn

try:
    from torch_cluster import fps, knn
    HAS_FPS = True
except ImportError:
    fps = None
    HAS_FPS = False
    print("[PointNet2] torch_cluster not found - falling back to random sampling.")

from torch_geometric.nn import PointNetConv, fps, radius, global_max_pool, MLP
from torch_geometric.data import Batch


class SAModule(nn.Module):
    """Single-scale PointNet++ block (FPS -> radius-graph -> PointNetConv)."""

    def __init__(self, ratio: float, r: float, mlp_channels, k: int = 16):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.k = k
        self.conv = PointNetConv(MLP(mlp_channels), add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=self.k)
        edge_index = torch.stack([col, row], dim=0)

        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(nn.Module):
    """Global max-pool that returns one node per cloud."""

    def __init__(self, mlp_channels):
        super().__init__()
        self.nn = MLP(mlp_channels)

    def forward(self, x, pos, batch):
        x = torch.cat([x, pos], dim=1) if x is not None else pos
        x = self.nn(x)
        x = global_max_pool(x, batch)
        cloud_ids = torch.unique(batch, sorted=True)
        return x, cloud_ids


class PointNet2CUDA(nn.Module):
    """PointNet++ encoder: (B, 518, 4) + mask (B, 518) -> (B, 64)."""

    def __init__(self):
        super().__init__()
        self.sa1 = SAModule(ratio=0.125, r=0.20, mlp_channels=[4, 16, 32], k=16)
        self.sa2 = SAModule(ratio=0.25, r=0.40, mlp_channels=[32 + 3, 32, 64], k=16)
        self.global_sa = GlobalSAModule(mlp_channels=[64 + 3, 64])
        self.empty_token = nn.Parameter(torch.zeros(1, 64))

    @staticmethod
    def _to_batch(pts: torch.Tensor, mask: torch.Tensor) -> Batch:
        pos_l, feat_l, batch_l = [], [], []
        B, N, _ = pts.shape
        for b in range(B):
            idx = mask[b].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            pos_l.append(pts[b, idx, :3])
            feat_l.append(pts[b, idx, 3:])
            batch_l.append(torch.full((idx.size(0),), b, device=pts.device))
        if not pos_l:
            return None
        return Batch(pos=torch.cat(pos_l), x=torch.cat(feat_l), batch=torch.cat(batch_l))

    def forward(self, pts: torch.Tensor, mask: torch.Tensor):
        B = pts.size(0)
        data = self._to_batch(pts, mask)
        if data is None:
            return self.empty_token.repeat(B, 1)

        x, pos, batch = data.x, data.pos, data.batch
        x, pos, batch = self.sa1(x, pos, batch)
        x, pos, batch = self.sa2(x, pos, batch)
        x, cloud_ids = self.global_sa(x, pos, batch)

        latent = self.empty_token.repeat(B, 1)
        latent[cloud_ids] = x
        return latent


class DexploreBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)

            if self.is_continuous and not self.space_config['learn_sigma']:
                actions_num = kwargs.get('actions_num')
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                sigma_init(self.sigma)

            self.LOCAL_OBS_DIM = 30  # 15 dof_pos + 5 contact + 1 todo + 3 wrist_trans_delta + 6 wrist_rot_delta
            self.HAND_POS_DIM = 18  # root_pos(3) + 5 key body pos(15), table-relative
            self.latent_dim = int(params.get('vae_latent_dim', 64))
            PC_FEAT = 64  # PointNet2 output feature dim (independent of VAE latent)
            trunk_local = self.LOCAL_OBS_DIM + self.HAND_POS_DIM
            self.prior = PointNet2CUDA()
            self._mu_head_1 = nn.Sequential(nn.Linear(trunk_local + PC_FEAT, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.latent_dim))
            self._logvar_head_1 = nn.Sequential(nn.Linear(trunk_local + PC_FEAT, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.latent_dim))
            self._mu_head_2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.latent_dim))
            self._logvar_head_2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.latent_dim))
            self.encoder = nn.Sequential(nn.Linear(1442, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.Linear(512, 256))
            self.aux_decoder = nn.Sequential(nn.Linear(self.latent_dim, 128), nn.ReLU(), nn.Linear(128, 9))  # object pos(3) + rot(6)

            # Rebuild actor_mlp's first linear so it takes [z, local_obs, hand_pos] directly
            # instead of going through a 112->1442 upsample into a teacher-shaped actor_mlp.
            trunk_in = self.latent_dim + trunk_local
            first = self.actor_mlp[0]
            assert isinstance(first, nn.Linear)
            new_first = nn.Linear(trunk_in, first.out_features)
            self.actor_mlp[0] = new_first

        def forward(self, obs_dict):
            if obs_dict['with_vae']:
                actor_outputs, states = self.get_action_and_vae_outputs(obs_dict)
            else:
                actor_outputs = self.act(obs_dict)
                states = obs_dict.get('rnn_states', None)

            obs = obs_dict['obs']
            value = self.eval_critic(obs)
            output = actor_outputs + (value, states)
            return output

        def _prior(self, input_dict):
            student_obs = input_dict['student_obs']
            N_PTS = 518
            local_obs = student_obs[..., :self.LOCAL_OBS_DIM]
            xyz_flat = student_obs[..., self.LOCAL_OBS_DIM : self.LOCAL_OBS_DIM + N_PTS * 3]
            valid = student_obs[..., self.LOCAL_OBS_DIM + N_PTS * 3 :]  # (B, 518)
            xyz = xyz_flat.view(-1, N_PTS, 3)
            hand_pos = xyz[..., :6, :].reshape(*xyz.shape[:-2], self.HAND_POS_DIM)  # root + 5 key bodies
            pts = torch.cat([xyz, valid.unsqueeze(-1)], dim=-1)  # (B, 518, 4): xyz + semantic flag
            mask = valid > 0.2
            points_feature = self.prior(pts, mask)
            if torch.any(~torch.isfinite(points_feature)):
                raise RuntimeError("invalid points_feature")
            feature = torch.cat([local_obs, hand_pos, points_feature], dim=-1)
            mu = self._mu_head_1(feature)
            logvar = self._logvar_head_1(feature)
            if torch.any(~torch.isfinite(mu)):
                raise RuntimeError("invalid mu")
            if torch.any(~torch.isfinite(logvar)):
                raise RuntimeError("invalid logvar")
            return {'mu': mu, 'logvar': logvar}

        def _encoder(self, input_dict):
            obs = input_dict['obs']
            a_out = self.encoder(obs)
            mu = self._mu_head_2(a_out)
            logvar = self._logvar_head_2(a_out)
            if torch.any(~torch.isfinite(mu)):
                raise RuntimeError("invalid mu")
            if torch.any(~torch.isfinite(logvar)):
                raise RuntimeError("invalid logvar")
            return {'mu': mu, 'logvar': logvar}

        def _trunk(self, input_dict):
            student_obs = input_dict['student_obs']
            local_obs = student_obs[..., :self.LOCAL_OBS_DIM]
            hand_pos = student_obs[..., self.LOCAL_OBS_DIM : self.LOCAL_OBS_DIM + self.HAND_POS_DIM]
            a_out = torch.cat([input_dict["vae_latent"], local_obs, hand_pos], dim=-1)
            a_out = self.actor_mlp(a_out)
            mu = self.mu_act(self.mu(a_out))
            if torch.any(~torch.isfinite(mu)):
                raise RuntimeError("invalid mu")
            if self.space_config['fixed_sigma']:
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(a_out))
            if torch.any(~torch.isfinite(sigma)):
                raise RuntimeError("invalid sigma")
            return mu, sigma

        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)
            value = self.value_act(self.value(c_out))
            return value

        def reparameterization(self, mean, std, vae_noise):
            return mean + std * vae_noise

        def act(self, input_dict: dict):
            prior_out = self._prior(input_dict)
            if input_dict['with_encoder']:
                encoder_out = self._encoder(input_dict)
                mu = prior_out["mu"] + encoder_out["mu"]
                logvar = encoder_out["logvar"]
            else:
                mu = prior_out["mu"]
                logvar = prior_out["logvar"]

            z = self.reparameterization(mu, torch.exp(0.5 * logvar), input_dict["vae_noise"])
            input_dict["vae_latent"] = z
            return self._trunk(input_dict)

        def get_action_and_vae_outputs(self, input_dict: dict):
            """Get action and VAE outputs by sampling from the encoder.

            The encoder output's mu acts as a residual to the prior's mu, while
            its logvar is used directly.
            """
            prior_out = self._prior(input_dict)
            encoder_out = self._encoder(input_dict)

            mu = prior_out["mu"] + encoder_out["mu"]
            logvar = encoder_out["logvar"]

            if "vae_noise" not in input_dict:
                input_dict["vae_noise"] = torch.randn_like(mu)

            z = self.reparameterization(mu, torch.exp(0.5 * logvar), input_dict["vae_noise"])
            input_dict["vae_latent"] = z
            aux_pred = self.aux_decoder(z)
            action = self._trunk(input_dict)
            return action, {'prior_out': prior_out, 'encoder_out': encoder_out, 'aux_pred': aux_pred}

    def build(self, name, **kwargs):
        net = DexploreBuilder.Network(self.params, **kwargs)
        return net
