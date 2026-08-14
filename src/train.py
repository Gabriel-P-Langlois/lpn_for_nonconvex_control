"""Training loop for the LPN potential (D2 convention, step-based budget).

Both networks (the psi network and the one-shot-recovery G network) use the SAME loss --
MSE -- fixing the Phase 1 finding 3 mismatch (psi was MAE, G drifted to MSE).
Training is mini-batched (batch 512), the measured ~50x speedup over the old
full-batch loop; the target is regressed against a held-out validation set so
the reported error is not the training loss (finding 5).

BUDGET IS IN OPTIMIZER STEPS, NOT EPOCHS (D2 amendment, 2026-07-09). The old
300-epoch budget left every network under-trained: 4x the steps on the SAME
data cuts psi's validation MSE by 4x at d=2 and 9x at d=4, while 4x the data at
a fixed step budget buys under 10%. Since N now grows with the dimension
(N = 15000*d), an epoch means something different at every d, so the
learning-rate decay is counted in STEPS. Epochs are not a control here and
are not reported.
"""
import numpy as np
import torch


def _to_tensor(a, device):
    return torch.tensor(np.asarray(a)).float().to(device)


def train_potential(
    model,
    inputs,
    targets,
    val_inputs=None,
    val_targets=None,
    batch_size=512,
    steps=250_000,
    lr=1e-3,
    # Fractions of the STEP budget, NOT absolute step counts, so the schedule is
    # invariant to S.
    #
    # We do NOT copy the paper's uniform 20% cadence (four decays x0.1 every 1e5
    # of its 5e5 full-batch steps). That cadence was designed for EXACT gradients;
    # ours are mini-batch estimates and need longer at the high learning rate.
    # Measured at d=16, quadratic_l1, psi's validation MSE against the number of
    # steps spent at lr=1e-3 -- which is what tracks, not the total budget:
    #     S=150k, (0.5,0.75)            ->  75k steps at 1e-3  ->  6.99e-2
    #     S=250k, (0.3,0.5,0.7,0.85)    ->  75k steps at 1e-3  ->  6.30e-2
    #     S=300k, (0.5,0.75)            -> 150k steps at 1e-3  ->  7.83e-3
    # Doubling the high-LR phase is worth 8x; adding 100k total steps while
    # halving it is worth 11%.
    lr_decay_at=(0.5, 0.75),
    eval_every=500,
    warmup_lr=1e-3,
    seed=1,
    verbose=True,
):
    """Fit model.scalar to `targets` with mini-batch MSE for a FIXED budget.

    NO EARLY STOPPING (removed 2026-07-09). Early stopping exists to prevent
    overfitting and to save compute. We do neither: training and validation MSE
    are both still falling when it fired, and the compute it saved is compute we
    want to spend. In practice it truncated the G network on NOISE -- at d=16 it
    stopped G at step 39500 of 300000, deep in the constant-LR phase, where the
    validation curve swings by more than an order of magnitude between
    checkpoints. Patience is gone as a knob; the schedule is now one sentence:
    `steps` steps of Adam at `lr`, decayed x0.1 at 50% and 75% of the budget.

    We keep the best-validation CHECKPOINT, which costs nothing, guards against a
    noise spike landing on the final step, and makes hist["best_val"] describe
    the model we actually return.
    """
    device = next(model.parameters()).device
    X = _to_tensor(inputs, device)
    Y = _to_tensor(targets, device).reshape(-1, 1)
    n = X.shape[0]
    has_val = val_inputs is not None and val_targets is not None
    if has_val:
        Xv = _to_tensor(val_inputs, device)
        Yv = _to_tensor(val_targets, device).reshape(-1, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=warmup_lr)
    _one_step(model, optimizer, X[:batch_size], Y[:batch_size])  # warm-up step
    for g in optimizer.param_groups:
        g["lr"] = lr

    decay_steps = {int(f * steps) for f in lr_decay_at}
    gen = torch.Generator(device="cpu").manual_seed(seed)

    hist = {"train": [], "val": [], "steps": [], "best_val": None}
    best_val, best_state = np.inf, None
    step, running, seen = 0, 0.0, 0
    perm = torch.randperm(n, generator=gen).to(device)
    cursor = 0

    while step < steps:
        if cursor + batch_size > n:  # reshuffle when the epoch is exhausted
            perm = torch.randperm(n, generator=gen).to(device)
            cursor = 0
        idx = perm[cursor : cursor + batch_size]
        cursor += batch_size

        if step in decay_steps and step > 0:
            for g in optimizer.param_groups:
                g["lr"] *= 0.1

        running += _one_step(model, optimizer, X[idx], Y[idx]) * idx.shape[0]
        seen += idx.shape[0]
        step += 1

        if has_val and step % eval_every == 0:
            with torch.no_grad():
                val_mse = (model.scalar(Xv) - Yv).pow(2).mean().item()
            hist["train"].append(running / seen)
            hist["val"].append(val_mse)
            hist["steps"].append(step)
            running, seen = 0.0, 0

            if val_mse < best_val - 1e-12:
                best_val = val_mse
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
            if verbose and step % max(eval_every, steps // 10) == 0:
                print(f"  step {step:7d}  train {hist['train'][-1]:.4e}  "
                      f"val {val_mse:.4e}")

    if has_val and best_state is not None:
        model.load_state_dict(best_state)
        hist["best_val"] = float(best_val)
    return hist


def _one_step(model, optimizer, xb, yb):
    """One MSE step with post-step weight clipping to preserve convexity."""
    optimizer.zero_grad()
    loss = (model.scalar(xb) - yb).pow(2).mean()
    loss.backward()
    optimizer.step()
    model.wclip()
    return loss.item()
