import torch
import torch.nn as nn
from typing import Tuple


class RLLGMNCell(nn.Module):
    """A Reccurent Log-Linearized Gaussian mixture Model layer
    Args:
        in_features: size of each first input sample
        n_class: size of each output sample
        n_state: number of states
        n_component: number of Gaussian components

    Shape:
        - Input : (sample(batch), in_features), (sample(batch), n_class, n_state)
        - Output : (sample(batch), n_class)

    Attributes:
        weight: shape (H, n_class, n_component),
                where H = 1 + in_features * (in_features + 3) / 2
        bias: None
    """

    def __init__(self, in_features: int, n_class: int, n_state: int, n_component: int) -> None:
        super(RLLGMNCell, self).__init__()
        self.in_features = in_features
        self.H = int(1 + self.in_features * (self.in_features + 3) / 2)
        self.weight = nn.Parameter(torch.Tensor(self.H, n_class, n_state, n_state, n_component))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """initialize weight
        """
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        with torch.no_grad():
            self.weight[:, -1, -1, -1, -1] = 0

    def nonlinear_trans(self, x: torch.Tensor) -> torch.Tensor:
        """Nonlinear transformation.
        Shape:
            Input: (sample(batch), in_future)
            Output: (batch_size, H), where H = 1 + dimension*(dimension + 3)/2
        """
        with torch.no_grad():
            outer_prod = torch.einsum('ni,nj->nij', x, x)
            ones_mat = torch.ones([x.size()[-1], x.size()[-1]],dtype=torch.bool)
            mask = torch.triu(ones_mat)
            quadratic_term = outer_prod[:, mask]
            bias_term = torch.ones([x.size()[0], 1])
            if torch.cuda.is_available():
                bias_term = bias_term.cuda()
            output = torch.cat([bias_term, x, quadratic_term], dim=1)
        return output

    def redundant_term_to_zero(self, I2: torch.Tensor) -> torch.Tensor:
        """
         Shape:
            Input: (sample(batch), n_class, n_component)
            Output: (sample(batch) n_class, n_component). I2 with redundant term replaced
        """
        with torch.no_grad():        
            I2[:, -1, -1, -1, -1] = 0.0
        return I2
     
    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        n: sample(batch), c: class, k, j : states, m: component
        """
        x_nonlinear = self.nonlinear_trans(inputs)
        I2 = torch.einsum('ni,ickjm->nckjm', x_nonlinear, self.weight)
        I2_ = self.redundant_term_to_zero(I2)
        O2 = torch.exp(I2_)
        I3 = torch.sum(O2, dim=4, keepdim=False)
        I4 = torch.einsum('ncj,ncjk->nck', hidden, I3)
        O4 = I4 / torch.sum(I4, dim=[1, 2], keepdim=True)
        O5 = torch.sum(O4, dim=2, keepdim=False)
        return O5, O4
