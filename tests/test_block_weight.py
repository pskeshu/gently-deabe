"""Verify the linear-ramp blending mask matches the reference implementation."""
import numpy as np

from gently_deabe.inference import _build_block_weight


def test_block_weight_shape():
    bw = _build_block_weight((32, 128, 128), (4, 16, 16))
    assert bw.shape == (32, 128, 128)


def test_block_weight_centre_is_one():
    bw = _build_block_weight((32, 128, 128), (4, 16, 16))
    # The middle of the block should be 1 (no overlap there).
    assert bw[16, 64, 64] == 1.0


def test_block_weight_edges_taper():
    bw = _build_block_weight((32, 128, 128), (4, 16, 16))
    # The corners should be smaller than the centre due to the linear ramp.
    assert bw[0, 0, 0] < bw[16, 64, 64]


def test_block_weight_symmetry():
    """The mask is mirror-symmetric in each axis (linear ramp on both sides)."""
    bw = _build_block_weight((32, 32, 32), (4, 8, 8))
    # Symmetry under index reversal in each axis.
    np.testing.assert_allclose(bw, bw[::-1, :, :], rtol=1e-6)
    np.testing.assert_allclose(bw, bw[:, ::-1, :], rtol=1e-6)
    np.testing.assert_allclose(bw, bw[:, :, ::-1], rtol=1e-6)
