import sys, os; sys.path.insert(0, os.path.abspath("."))
import numpy as np, torch
from src.network import LPN
from src.recovery import conjugate_samples
torch.manual_seed(0)
m = LPN(in_dim=8, hidden=256, layers=2, beta=5)
x = np.random.default_rng(0).uniform(-4, 4, (7003, 8))   # not a multiple of chunk
y1, G1 = conjugate_samples(x, m, chunk=10**9)   # single pass
y2, G2 = conjugate_samples(x, m, chunk=1000)    # chunked, ragged last block
print("shapes", y1.shape, y2.shape, G1.shape, G2.shape)
print("max |dy| =", np.abs(y1-y2).max(), "  max |dG| =", np.abs(G1-G2).max())
assert y1.shape == y2.shape and G1.shape == G2.shape
assert np.array_equal(y1, y2) and np.array_equal(G1, G2), "chunking changed the result"
print("PASS: chunked conjugate_samples is bit-identical")
