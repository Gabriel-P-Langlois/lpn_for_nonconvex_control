"""tvpm -- recovering a denoiser's implicit regularizer from its outputs alone.

The TV posterior-mean example of work2.tex, Instantiation B: the anisotropic-TV
posterior-mean denoiser IS the proximal operator of some f_reg, so fitting one
convex network to (x_k, u_PM(x_k)) pairs recovers f_reg without ever seeing it.

Two stages, and they share nothing but the cached array between them:

    stage 1  dataset/sampler   x_k -> u_PM(x_k) by MCMC        (expensive, cached)
    stage 2  recover (nets in src/) fit a convex net to those pairs (expensive, cached)

then figures/ and denoise/ only READ the trained net.

The model's two degrees of freedom are (sigma, t); eps = sigma^2/t and lam = 2t
follow -- see sampler.from_sigma_t. Everything downstream is keyed on the pair,
so two (sigma, t) never overwrite each other's data, checkpoints, or figures.

Entry points, in the order a run uses them:

    dataset.ensure(...)        -> (train, val, eval) splits, sampling only what is missing
    recover.ensure_model(...)  -> (model, units, meta), training only if uncached
    recover.estimate_delta()   -> the sampler noise floor
    recover.score(...)         -> held-out prox residual and corr(J, TV)
    figures.figure_core(...)   -> what the recovered prior looks like
    denoise.run(...)           -> the prior in action, vs the true denoiser
"""
from . import dataset, denoise, figures, paths, quadrature, recover, sampler
from .sampler import from_sigma_t, params

__all__ = ["dataset", "denoise", "figures", "paths", "quadrature",
           "recover", "sampler", "from_sigma_t", "params"]
