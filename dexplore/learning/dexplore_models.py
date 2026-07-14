from rl_games.algos_torch.models import ModelA2CContinuousLogStd


class ModelDexploreContinuous(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)

    def build(self, config):
        net = self.network_builder.build('dexplore', **config)
        return ModelDexploreContinuous.Network(net)

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network):
            super().__init__(a2c_network)

        def forward(self, input_dict):
            return super().forward(input_dict)
