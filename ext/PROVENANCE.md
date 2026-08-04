# ext/ — third-party code (not ours)

## ext/lpn/
Upstream "Learned Proximal Networks" reference implementation
(Fang, Buchanan, Sulam et al.), moved here verbatim from `numerics/lpn/`
on 2026-07-08 during the src/ consolidation. Do not edit in place; if a
patch is needed, record it here. Our own consolidated code lives in
`numerics/src/`; this tree is retained only for the imaging/LPN-baseline
experiments (see CLAUDE.md, WP6).

## ext/pirate/
Official code for "A Plug-and-Play Image Registration Network" (Hu, Gan,
Sun, An, Kamilov, ICLR 2024, arXiv:2310.04297). Shallow clone of
https://github.com/wustl-cig/PIRATE-code, commit
4901b719e6705c67fd6532b150cd0f2623a8d00c (2024-03-19), cloned 2026-07-30
for Experiment 2 of `__tasks/task_pnp_registration.tex` (the
PIRATE/PIRATE+ Jacobian diagnosis, `numerics/pnp_reg/`). MIT license
(upstream `LICENSE` retained). Do not edit in place.

Contents used by us: `pretrained_model/AWGN_denoiser/OASIS.pth.tar`
(DnCNN residual denoiser, epoch 400), `pretrained_model/PIRATEplus/OASIS.pth.tar`
(the DEQ-fine-tuned denoiser; same 12 DnCNN tensors under a
`PIRATE.dnn.` key prefix), `data/field.h5py` (one example registration
field, shape (1, 3, 80, 96, 112)), and `model/base.py` (the DnCNN class).
Conventions that differ from the paper's notation, verified against
`train_denoiser.py`: the network predicts the NOISE, so the denoiser is
D(z) = z - dnn(z); the release contains ONE denoiser checkpoint trained
at config sigma = 1, not ten; and the training noise is
`sigma**2 * randn`, i.e. the noise standard deviation is sigma squared
(equal to 1 at sigma = 1).
