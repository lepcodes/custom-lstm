import torch.nn as nn

from custom_lstm.models.base_model import AblationModel


class MLP(AblationModel):
    def __init__(self, input_size, output_size, hidden_layers: list):
        super(MLP, self).__init__()

        layers = []
        current_input_size = input_size
        for layer_size in hidden_layers:
            linear_layer = nn.Linear(current_input_size, layer_size)
            nn.init.kaiming_normal_(linear_layer.weight, nonlinearity='leaky_relu')
            layers.append(linear_layer)
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            current_input_size = layer_size

        final_layer = nn.Linear(current_input_size, output_size)
        nn.init.constant_(final_layer.bias, 1.0)
        # init.constant_(final_layer.bias, 3.0)
        layers.append(final_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, sequence):
        return self.network(sequence), None

    def reset_state(self):
        pass
