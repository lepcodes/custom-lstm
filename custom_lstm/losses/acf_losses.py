import torch
from torch import nn


class EWACFLoss(nn.Module):
    def __init__(self, lambda_=0.5, lag=1, alpha=0.5):
        super(EWACFLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        self.lambda_ = lambda_
        self.lag = lag
        self.epsilon = 1e-8
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("variance", torch.zeros(1) + self.epsilon)
        self.register_buffer("covariance", torch.zeros(1))
        self.register_buffer("input_lag", torch.zeros(0))

    def forward(self, input, target, sequence, forget_gates):
        if self.input_lag.numel() == 0:
            full_sequence = sequence
            initial_padding = [torch.zeros(sequence.shape[0], sequence.shape[2], device=sequence.device)] * self.lag
        else:
            full_sequence = torch.cat((self.input_lag, sequence), dim=1)
            initial_padding = []

        self.input_lag = full_sequence[:, -self.lag :, :].detach()

        autocorrelation = [] + initial_padding
        for t in range(self.lag, full_sequence.shape[1]):
            x_t = full_sequence[:, t, :]
            x_t_lag = full_sequence[:, t - self.lag, :]

            self.mean = torch.mul(self.lambda_, self.mean) + torch.mul(1 - self.lambda_, x_t)
            self.variance = torch.mul(self.lambda_, self.variance) + torch.mul(1 - self.lambda_, (x_t - self.mean) ** 2)
            self.covariance = torch.mul(self.lambda_, self.covariance) + torch.mul(1 - self.lambda_, (x_t - self.mean) * (x_t_lag - self.mean))
            autocorrelation.append(self.covariance / torch.sqrt(self.variance * self.variance + self.epsilon))

        irrelevance = 1 - torch.abs(torch.stack(autocorrelation, dim=1))
        irrelevance = torch.mean(irrelevance, dim=2, keepdim=True)
        penalty = torch.mul(irrelevance, forget_gates)
        penalty = torch.mean(penalty)
        mse = self.mse_loss(input, target) + self.alpha * penalty
        return mse

    def reset_state(self):
        self.mean.zero_()
        self.variance.fill_(self.epsilon)
        self.covariance.zero_()
        self.input_lag = torch.tensor([], device=self.mean.device)
