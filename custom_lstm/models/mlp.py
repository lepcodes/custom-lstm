import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers: list):
        super(MLP, self).__init__()

        layers = []
        current_input_size = input_size
        for layer_size in hidden_layers:
            layers.append(nn.Linear(current_input_size, layer_size))
            layers.append(nn.ReLU())
            current_input_size = layer_size
        layers.append(nn.Linear(current_input_size, output_size))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
