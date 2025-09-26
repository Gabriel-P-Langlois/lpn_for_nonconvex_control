"""Learned proximal network with softplus activation function and linear layers"""
import torch
import torch.nn as nn

class SmoothedReLU(nn.Module):
    def __init__(self, eps):
        super(SmoothedReLU,self).__init__()
        self.eps = eps

    def forward(self, x):
        eps = self.eps
        # Use torch operations for broadcasting and gradients
        out = torch.where(
            x > eps,
            x,
            torch.where(
                x < -eps,
                torch.zeros_like(x),
                (1 / (4 * eps)) * x**2 + (1 / 2) * x + (eps / 4)
            )
        )
        return out

class LPN(nn.Module):
    def __init__(self, in_dim, hidden, layers=1, beta=1,eps=1):
        super().__init__()

        self.hidden = hidden
        self.lin = nn.ModuleList(
            [
                nn.Linear(in_dim, hidden, bias=False),
                *[nn.Linear(hidden, hidden, bias=False) for _ in range(layers)],
                nn.Linear(hidden, 1, bias=False), 
            ]
        )

        self.res = nn.ModuleList(
            [*[nn.Linear(in_dim, hidden) for _ in range(layers)], nn.Linear(in_dim, 1)]
        )
        #self.act = nn.Softplus(beta=beta)
        #self.act = nn.ReLU() # Using ReLU activation for better performance in many case
        self.act = nn.Mish() # Other activations ReLU, Mish,Softmax() better performance in many case
        #self.act = SmoothedReLU(eps=eps) # Using a smoothed ReLU for better numerical stability

    
    def scalar(self, x):
        y = x.clone()
        y = self.act(self.lin[0](y))
        for core, res in zip(self.lin[1:-1], self.res[:-1]):
            y = self.act(core(y) + res(x))


        y = self.lin[-1](y) + self.res[-1](x)
        #y, _ = torch.max(y, dim=1, keepdim=True)
        #y = y + self.res[-1](x)
        return y

    def init_weights(self, mean, std):
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.normal_(mean, std).exp_()

    # this clips the weights to be non-negative to preserve convexity
    def wclip(self):
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.clamp_(0)

    # Note: We pass the input of the LPN model as the x variable below.
    def forward(self, x):
        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad = True
            y = self.scalar(x)
            grad = torch.autograd.grad(
                y.sum(), x, retain_graph=True, create_graph=True
            )[0]

        return grad
