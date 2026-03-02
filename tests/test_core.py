import numpy as np
from pmf_from_gofr.core import PMFParams, normalize_g_tail, iterate_closure


def test_normalize_g_tail_scale_sets_tail_to_one():
    x = np.linspace(0.0, 40.0, 401)
    g = np.ones_like(x) * 1.2
    g2 = normalize_g_tail(x, g, tail_start=30.0, method="scale", target=1.0)
    tail = g2[x >= 30.0]
    assert np.isclose(np.median(tail), 1.0, atol=1e-12)


def test_iterate_closure_shapes():
    x = np.linspace(1.0, 40.0, 500)
    g = np.ones_like(x) * 1.1
    params = PMFParams(max_iter=200, conv_tol=1e-12)
    C, u, n = iterate_closure(x, g, params)
    assert C.shape == x.shape
    assert u.shape == x.shape
    assert n > 0

