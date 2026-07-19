"""Convolutional input-convex neural network (conv-ICNN) for the TV example.

WHY THIS EXISTS. The fully-connected ICNN (`src/network.py:LPN`) recovers the TV
denoiser's f_reg to a ~10.7% held-out prox residual and stalls there: width
(256/512/1024) and depth (2/3 layers) do not move it, while the three levers that
DID work (input standardization, high-LR steps, activation sharpness beta) all
MATCHED the net to the problem's scales rather than adding capacity. That pattern
says the floor is the statistical cost of fitting a SHIFT-INVARIANT, LOCAL
function with a dense net that knows neither fact. f_reg ~ sum over lattice sites
of phi(local neighbourhood): a conv-ICNN builds locality (small kernels) and
shift-invariance (weight sharing) in, so it learns phi once instead of learning
it separately at all 64 sites.

NOT CIRCULAR. Handing the net the difference directions y_i - y_j would assume
f_reg is a sum of pairwise functions of those differences -- i.e. assume TV. A
conv-ICNN assumes only LOCALITY and SHIFT-INVARIANCE, which are properties of any
image prior and are in fact KNOWN here: the denoiser is anisotropic TV plus a box,
local and (in its interior) shift-invariant. A 3x3 conv can learn any local
shift-invariant convex functional -- joint 9-pixel interactions, curvature
penalties, not only pairwise differences -- so it is a strictly weaker,
physically-justified prior, not the answer.

CONVEXITY -- the whole point, so state it exactly.

  Amos, Xu & Kolter, "Input Convex Neural Networks", ICML 2017, Prop. 1:
  a feed-forward net z_{i+1} = g_i(W_i^z z_i + W_i^y y + b_i), scalar output, is
  CONVEX in y provided (i) every W_i^z is entrywise NONNEGATIVE and (ii) every
  g_i is CONVEX and NON-DECREASING.

  Proof (induction): if z_i is convex in y then W_i^z z_i is a nonnegative
  combination of convex functions (convex), plus the affine W_i^y y + b_i
  (convex), and g_i convex non-decreasing composed with convex is convex. Base
  case is g_0 of an affine map. QED.

  A CONVOLUTION IS LINEAR, so it is a special W with Toeplitz structure and the
  proposition applies VERBATIM, read on the KERNEL entries. The convolutional
  instantiation follows Mukherjee et al., "Learned convex regularizers for
  inverse problems", 2020 (arXiv:2008.02839), who use exactly this as an image
  regularizer for deblurring and CT.

  Here that means:
    * in_conv  (from y):        UNCONSTRAINED  -- affine in y, convexity-neutral
    * feat convs (feature path): NONNEGATIVE kernels  -- the W^z of Prop. 1
    * skip convs (from y):       UNCONSTRAINED  -- affine in y
    * head (channels -> scalar): NONNEGATIVE          -- the last W^z
    * head_skip (from y):        UNCONSTRAINED  -- affine in y
    * spatial pooling:           SUM  -- a nonnegative linear map
    * activation Softplus(beta):  convex and strictly increasing for any beta > 0
      (Softplus' = sigmoid in (0,1) > 0, Softplus'' = sigmoid(1-sigmoid) > 0).
      A NON-monotone activation (e.g. Mish, dropped from this repo as B5) breaks
      condition (ii) and with it convexity.

  Biases are affine and may take any sign; only the listed weights are clipped.
  `wclip()` enforces the nonnegativity after every optimizer step, exactly as
  LPN.wclip() does for the dense net. Any FIXED padding (zero/reflect/replicate)
  is a linear map of y and preserves convexity; padding is chosen for FIDELITY to
  the non-wrapping TV boundary (see `pad`), NOT for convexity.

The guarantee holds only if the constraints hold in CODE, so it is a claim to be
TESTED, not trusted: `test_conv_icnn.py` checks the midpoint inequality, the
Hessian's positive-semidefiniteness, and that wclip actually clamps -- and checks
that convexity FAILS without wclip, so the test has teeth.

INTERFACE. Presents the same surface as LPN -- `scalar(x)` mapping a FLAT (N, HW)
batch to (N, 1), and `wclip()` -- so `recover.py`'s train_grad, net_grad,
net_value and score run UNCHANGED. It reshapes to (N, 1, H, W) internally.
"""
import torch
import torch.nn as nn


class ConvICNN(nn.Module):
    def __init__(self, hw=(8, 8), channels=32, blocks=2, beta=20.0, pad="reflect"):
        """Input-convex CNN over an H x W patch.

        hw       : (H, W) of the patch; the flat input is H*W long.
        channels : feature channels C. Needs C >= 2 to build |r| from +/-(y_i-y_j);
                   32 is generous and cheap (conv kernels are 9 params each).
        blocks   : hidden conv blocks after the input conv (the ICNN "depth").
        beta     : Softplus sharpness. f_reg's TV kink is sharp; beta=20 matched
                   the dense net best, so it is the default here too.
        pad      : padding for every conv. The sampler's TV does NOT wrap, so
                   'circular' is WRONG (it imposes periodic TV). 'reflect' (the
                   default) or 'replicate' invent no edge; 'zeros' fabricates a
                   step at the border. All are linear, hence convexity-safe;
                   this is a fidelity choice, watched at the patch boundary.
        """
        super().__init__()
        self.H, self.W = hw
        self.channels, self.blocks, self.beta, self.pad = channels, blocks, beta, pad
        C = channels
        k = dict(kernel_size=3, padding=1, padding_mode=pad)

        self.in_conv = nn.Conv2d(1, C, **k)                                  # affine in y
        self.feat = nn.ModuleList([nn.Conv2d(C, C, **k) for _ in range(blocks)])   # W^z >= 0
        self.skip = nn.ModuleList([nn.Conv2d(1, C, bias=False, **k)
                                   for _ in range(blocks)])                  # affine in y
        self.head = nn.Linear(C, 1, bias=False)                             # W^z >= 0
        self.head_skip = nn.Linear(self.H * self.W, 1)                      # affine in y
        self.act = nn.Softplus(beta=beta)

        # Nonnegative init for the constrained paths: Kaiming draws ~half the
        # weights negative and wclip would zero them at step 1 (harmless, but the
        # abs() start keeps more live capacity, matching LPN's observed behaviour).
        with torch.no_grad():
            for f in self.feat:
                f.weight.abs_()
            self.head.weight.abs_()

    def scalar(self, x):
        """Convex potential J(x); x is FLAT (N, H*W), returns (N, 1)."""
        y = x.view(-1, 1, self.H, self.W)
        z = self.act(self.in_conv(y))
        for feat, skip in zip(self.feat, self.skip):
            z = self.act(feat(z) + skip(y))          # feat.weight >= 0 keeps convexity
        # MEAN-pool, not sum. Both are nonnegative linear (convexity-safe) and
        # representationally equivalent -- the head absorbs the H*W factor -- but
        # mean starts J at O(1) instead of O(H*W): summing 64 sites inflated the
        # initial gradient ~1000x past the target and Adam could not recover.
        pooled = z.mean(dim=(2, 3))                   # (N, C)
        return self.head(pooled) + self.head_skip(x)  # nonneg head + affine skip

    def wclip(self):
        """Clamp the feature-path and head weights to >= 0 (convexity). Biases,
        in_conv, skip and head_skip are affine in y and left free."""
        with torch.no_grad():
            for feat in self.feat:
                feat.weight.data.clamp_(0)
            self.head.weight.data.clamp_(0)

    def forward(self, x, create_graph=False):
        """grad_x J(x), the learned proximal map; (N, H*W). Mirrors LPN.forward."""
        xin = x.detach().requires_grad_(True)
        g = torch.autograd.grad(self.scalar(xin).sum(), xin, create_graph=create_graph)[0]
        return g
