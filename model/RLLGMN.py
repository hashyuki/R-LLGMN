from . RLLGMNCell import RLLGMNCell
import torch.nn as nn
import torch


class RLLGMN(nn.Module):
    def __init__(self, in_features: int, n_class: int, n_state: int, n_component: int) -> None:
        super().__init__()
        self.n_class = n_class
        self.n_state = n_state
        self.cell = RLLGMNCell(in_features, n_class, n_state, n_component)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        count = inputs.shape[1]  # depth
        batch_size = inputs.size()[0]
        hidden = torch.ones((batch_size, self.n_class, self.n_state))
        output = torch.ones((batch_size, self.n_class))
        if torch.cuda.is_available():
            hidden = hidden.cuda()
            output = output.cuda()
        for i in range(count):
            output, hidden = self.cell(inputs[:, i, :], hidden)
        return output