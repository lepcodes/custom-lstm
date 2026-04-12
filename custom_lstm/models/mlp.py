import torch.nn as nn

from custom_lstm.models.base_model import AblationModel


class MLP(AblationModel):
    def __init__(self, input_size, output_size, hidden_layers: list):
        super(MLP, self).__init__()

        layers = []
        current_input_size = input_size
        for layer_size in hidden_layers:
            layers.append(nn.Linear(current_input_size, layer_size))
            layers.append(nn.ReLU())
            current_input_size = layer_size
        final_layer = nn.Linear(current_input_size, output_size)
        # init.constant_(final_layer.bias, 3.0)
        layers.append(final_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, sequence):
        return self.network(sequence), None

    def reset_state(self):
        pass
