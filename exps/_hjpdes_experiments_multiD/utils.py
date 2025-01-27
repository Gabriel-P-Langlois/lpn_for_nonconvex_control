import numpy as np
import torch

# Learned convex function
def cvx(x, model):
    """Evaluate the learned convex function at x.
    Inputs:
        x: (n, ), a vector of n points, numpy array
        model: an LPN model
    Outputs:
        y: (n, ), a vector of n values, numpy array

    The shape of x should match the input shape of the model.
    """
    device = next(model.parameters()).device
    x = torch.tensor(x).float().to(device)
    return model.scalar(x).squeeze(1).detach().cpu().numpy()



# Learned proximal operator
def prox(x, model):
    """Evaluate the learned proximal operator at x.
    Inputs:
        x: (n, ), a vector of n points, numpy array
        model: an LPN model
    Outputs:
        y: (n, ), a vector of n points, numpy array

    The shape of x should match the input shape of the model.
    """
    device = next(model.parameters()).device
    x = torch.tensor(x).float().to(device)
    return model(x).detach().cpu().numpy()



def invert_cvx_gd(x, model):
    """Invert the learned proximal operator at x by convex optimization with conjugate gradient.
    Inputs:
        x: (n, *), a vector of n points, numpy array
        model: LPN model.
    Outputs:
        y: (n, *), a vector of n points, numpy array

    The shape of x should match the input shape of the model, i.e., (n, c, w, h)
    """
    z = x.copy()
    device = next(model.parameters()).device
    z = torch.tensor(z).float().to(device)
    x = torch.zeros(z.shape).to(device)
    x.requires_grad_(True)

    optimizer = torch.optim.Adam([x], lr=1e-2)

    for i in range(2000):
        optimizer.zero_grad()
        loss = torch_f(x, model, z)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("loss", loss.item())

    print("final mse: ", (model(x) - z).pow(2).mean().item())
    x = x.detach().cpu().numpy()
    print("max, min:", x.max(), x.min())
    return x


def torch_f(x, net, z):
    """
    Compute the value of the objective function:
    psi(x) - <z, x>

    x: torch tensor, shape (batch, channel, img_size, img_size)
    net: ICNN
    z: torch tensor, shape (batch, channel, img_size, img_size)
    Return: torch tensor, shape (1,)
    """
    b = x.shape[0]
    v = net.scalar(x).squeeze() - torch.sum(z.reshape(b, -1) * x.reshape(b, -1), dim=1)
    v = v.sum()
    return v




# Learned prior
def prior(x, model):
    """Evaluate the learned prior function at x.
    Inputs:
        x: (n, ), a vector of n points, numpy array
        model: an LPN model
    Outputs:
        y: (n, ), a vector of n values, numpy array
    """
    # psi(y) = <y, f(y)> - 1/2 ||f(y)||^2 - phi(f(y))
    y = invert_mse(x, model)
    psi = cvx(y, model)
    q = 0.5 * (x**2)  # quadratic term
    out = y * x - q - psi

    # print(y.shape, x.shape, q.shape, psi.shape)


    return out


