from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn


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

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)

            actor_outputs = self.eval_actor(obs)
            value = self.eval_critic(obs)

            output = actor_outputs + (value, states)
            return output

        def eval_actor(self, obs):
            a_out = self.actor_cnn(obs)
            a_out = a_out.contiguous().view(a_out.size(0), -1)
            a_out = self.actor_mlp(a_out)

            if self.is_discrete:
                return self.logits(a_out)
            if self.is_multi_discrete:
                return [logit(a_out) for logit in self.logits]
            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))
                return mu, sigma

        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)
            value = self.value_act(self.value(c_out))
            return value

    def build(self, name, **kwargs):
        net = DexploreBuilder.Network(self.params, **kwargs)
        return net
