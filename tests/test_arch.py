"""Architecture sanity tests for the PyTorch port."""
import torch

from gently_deabe.model import RCAN


def test_param_count_5x5():
    """Default RCAN (5 RG x 5 RB) should have 1,559,141 params (matches Keras)."""
    m = RCAN(num_residual_blocks=5, num_residual_groups=5)
    n = sum(p.numel() for p in m.parameters())
    assert n == 1_559_141, f"expected 1,559,141, got {n}"


def test_param_count_5x3():
    """Step-2 (Decon) variant: 5 RG x 3 RB."""
    m = RCAN(num_residual_blocks=3, num_residual_groups=5)
    n = sum(p.numel() for p in m.parameters())
    # 1 (head) + 5*(3 RCABs * (2*27680 + 132 + 160) + 27680) + 27680 + 865
    # = 896 + 5*(3*55652 + 27680) + 28545 = 896 + 5*194636 + 28545 = 1002621
    assert n == 1_002_621, f"expected 1,002,621, got {n}"


def test_forward_shape():
    """Forward pass preserves spatial shape and channel layout."""
    m = RCAN().eval()
    x = torch.rand(1, 1, 8, 32, 32)
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape


def test_forward_gpu_if_available():
    if not torch.cuda.is_available():
        return
    m = RCAN().cuda().eval()
    x = torch.rand(1, 1, 16, 64, 64, device="cuda")
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape
