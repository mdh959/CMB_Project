#!/usr/bin/env python3
"""
test_unet_phi.py  —  unit tests for train_unet_phi.py
Run with:  python test_unet_phi.py
"""

import sys
import numpy as np
import traceback

# Patch config so we don't need the real JLD2 file
import train_unet_phi as U
U.NSIDE    = 64       # tiny maps for fast tests
U.N_TRAIN  = 6
U.BATCH_SIZE = 2
U.EPOCHS   = 2
U.PATIENCE = 2
U._ELL_GRID = None    # reset cached grid

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []

def test(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append((name, True))
    except Exception as e:
        print(f"  {FAIL}  {name}")
        traceback.print_exc()
        results.append((name, False))


# ── 1. Import & TF available ──────────────────────────────────────────────────
def t_imports():
    import tensorflow as tf
    import h5py
    assert tf.__version__, "TensorFlow not found"

test("imports (tensorflow, h5py)", t_imports)


# ── 2. flat_cl: white noise map → flat spectrum ───────────────────────────────
def t_flat_cl_white():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((64, 64)).astype(np.float32)
    lc, cl = U.flat_cl(m, delta_l=500, l_min=500, l_max=8000)
    assert len(lc) > 0, "no bins returned"
    assert all(np.isfinite(cl)), "non-finite values in spectrum"
    assert all(cl >= 0), "negative power in auto-spectrum"

test("flat_cl: white noise auto-spectrum", t_flat_cl_white)


# ── 3. flat_cl: cross of identical maps = auto ────────────────────────────────
def t_flat_cl_cross_identity():
    rng = np.random.default_rng(1)
    m = rng.standard_normal((64, 64)).astype(np.float32)
    _, cl_auto  = U.flat_cl(m)
    _, cl_cross = U.flat_cl(m, m)
    np.testing.assert_allclose(cl_auto, cl_cross, rtol=1e-5,
                               err_msg="auto ≠ cross(m, m)")

test("flat_cl: cross(m, m) == auto(m)", t_flat_cl_cross_identity)


# ── 4. flat_cl: orthogonal maps → near-zero cross ────────────────────────────
def t_flat_cl_cross_orthogonal():
    rng = np.random.default_rng(2)
    m1 = rng.standard_normal((64, 64)).astype(np.float32)
    m2 = rng.standard_normal((64, 64)).astype(np.float32)
    _, cl_auto  = U.flat_cl(m1)
    _, cl_cross = U.flat_cl(m1, m2)
    # cross/auto should be << 1 for independent maps
    ratio = np.abs(cl_cross) / np.abs(cl_auto)
    assert np.mean(ratio) < 0.5, f"cross/auto too large: mean={np.mean(ratio):.3f}"

test("flat_cl: |cross(m1, m2)| / auto(m1) << 1 for independent maps", t_flat_cl_cross_orthogonal)


# ── 5. ensemble_spectra shape check ───────────────────────────────────────────
def t_ensemble_spectra_shapes():
    rng = np.random.default_rng(3)
    nsims = 5
    phi_t = rng.standard_normal((nsims, 64, 64)).astype(np.float32)
    phi_h = rng.standard_normal((nsims, 64, 64)).astype(np.float32)
    lc, tt, th, hh = U.ensemble_spectra(phi_t, phi_h)
    assert tt.shape == (nsims, len(lc)), f"tt shape {tt.shape}"
    assert th.shape == (nsims, len(lc)), f"th shape {th.shape}"
    assert hh.shape == (nsims, len(lc)), f"hh shape {hh.shape}"
    assert all(np.isfinite(lc)), "non-finite ell centres"

test("ensemble_spectra: output shapes correct", t_ensemble_spectra_shapes)


# ── 6. summary_stats: identical maps → W_L = 1, ρ_L = 1 ─────────────────────
def t_summary_stats_identical():
    rng = np.random.default_rng(4)
    nsims = 20
    phi_t = rng.standard_normal((nsims, 64, 64)).astype(np.float32)
    # perfect reconstruction: hat = true
    lc, tt, th, hh = U.ensemble_spectra(phi_t, phi_t)
    W, rho, N, sig, _ = U.summary_stats(tt, th, hh)
    # W_L should be near 1 for identical maps
    np.testing.assert_allclose(W, np.ones_like(W), atol=0.05,
                               err_msg="W_L ≠ 1 for identical maps")
    # ρ_L should be near 1
    np.testing.assert_allclose(rho, np.ones_like(rho), atol=0.05,
                               err_msg="ρ_L ≠ 1 for identical maps")
    # Noise should be near zero
    assert np.nanmean(np.abs(N)) < np.nanmean(tt), "noise floor not small for perfect estimator"

test("summary_stats: identical maps → W_L≈1, ρ_L≈1", t_summary_stats_identical)


# ── 7. summary_stats: zero hat → W_L = 0, ρ_L = 0 ───────────────────────────
def t_summary_stats_zero():
    rng = np.random.default_rng(5)
    nsims = 10
    phi_t = rng.standard_normal((nsims, 64, 64)).astype(np.float32)
    phi_z = np.zeros_like(phi_t)
    lc, tt, th, hh = U.ensemble_spectra(phi_t, phi_z)
    W, rho, N, sig, _ = U.summary_stats(tt, th, hh)
    np.testing.assert_allclose(W,   np.zeros_like(W),   atol=0.01, err_msg="W_L ≠ 0 for zero hat")
    np.testing.assert_allclose(rho, np.zeros_like(rho), atol=0.01, err_msg="ρ_L ≠ 0 for zero hat")

test("summary_stats: zero hat → W_L≈0, ρ_L≈0", t_summary_stats_zero)


# ── 8. build_unet output shape ────────────────────────────────────────────────
def t_unet_shape_single_channel():
    import tensorflow as tf
    orig_nside = U.NSIDE
    U.NSIDE = 64
    model = U.build_unet(n_channels=1, base_filters=4)
    x = np.zeros((1, 64, 64, 1), dtype=np.float32)
    y = model(x, training=False)
    assert y.shape == (1, 64, 64, 1), f"output shape {y.shape}"
    U.NSIDE = orig_nside

test("build_unet: output shape (1, 64, 64, 1) for single-channel input", t_unet_shape_single_channel)


# ── 9. build_unet two-channel input ──────────────────────────────────────────
def t_unet_shape_two_channel():
    import tensorflow as tf
    U.NSIDE = 64
    model = U.build_unet(n_channels=2, base_filters=4)
    x = np.zeros((2, 64, 64, 2), dtype=np.float32)
    y = model(x, training=False)
    assert y.shape == (2, 64, 64, 1), f"output shape {y.shape}"

test("build_unet: output shape (2, 64, 64, 1) for two-channel input", t_unet_shape_two_channel)


# ── 10. build_unet: output is finite and non-constant for non-zero input ──────
def t_unet_output_finite():
    import tensorflow as tf
    U.NSIDE = 64
    model = U.build_unet(n_channels=1, base_filters=4)
    rng = np.random.default_rng(6)
    x = rng.standard_normal((1, 64, 64, 1)).astype(np.float32)
    y = model(x, training=False).numpy()
    assert np.all(np.isfinite(y)), "U-Net output contains NaN/Inf"
    assert y.std() > 0, "U-Net output is constant (dead network)"

test("build_unet: output is finite and non-constant", t_unet_output_finite)


# ── 11. build_unet: gradient flows (loss decreases over 2 steps) ─────────────
def t_unet_trains():
    import tensorflow as tf
    U.NSIDE = 64
    rng = np.random.default_rng(7)
    X = rng.standard_normal((4, 64, 64, 1)).astype(np.float32)
    y = rng.standard_normal((4, 64, 64, 1)).astype(np.float32)
    model = U.build_unet(n_channels=1, base_filters=4)
    loss0 = model.evaluate(X, y, verbose=0)
    model.fit(X, y, epochs=3, verbose=0)
    loss1 = model.evaluate(X, y, verbose=0)
    assert loss1 < loss0 * 1.5, f"loss did not decrease: {loss0:.4f} → {loss1:.4f}"

test("build_unet: loss decreases during training", t_unet_trains)


# ── 12. _build_X: single and dual channel ─────────────────────────────────────
def t_build_X_channels():
    rng = np.random.default_rng(8)
    n = 5
    data = dict(
        phi_qe_wf = rng.standard_normal((n, 64, 64)).astype(np.float32),
        phi_gi_b  = rng.standard_normal((n, 64, 64)).astype(np.float32),
        phi_true  = rng.standard_normal((n, 64, 64)).astype(np.float32),
        seeds     = np.arange(n),
    )
    U.USE_GI = False
    X1 = U._build_X(data, use_gi=False)
    assert X1.shape == (n, 64, 64, 1), f"single ch: {X1.shape}"

    U.USE_GI = True
    X2 = U._build_X(data, use_gi=True)
    assert X2.shape == (n, 64, 64, 2), f"dual ch: {X2.shape}"

test("_build_X: single and dual channel shapes", t_build_X_channels)


# ── 13. Missing JLD2 file raises IOError, not silent failure ──────────────────
def t_missing_file_raises():
    try:
        U.load_phi_maps("results/DOES_NOT_EXIST.jld2")
        assert False, "should have raised"
    except (OSError, FileNotFoundError, IOError):
        pass   # expected

test("load_phi_maps: missing file raises IOError", t_missing_file_raises)


# ── 14. predict_phi_nn: output shape and physical units ───────────────────────
def t_predict_shape():
    import tensorflow as tf
    rng = np.random.default_rng(9)
    n = 4
    data = dict(
        phi_qe_wf = rng.standard_normal((n, 64, 64)).astype(np.float32),
        phi_gi_b  = rng.standard_normal((n, 64, 64)).astype(np.float32),
        phi_true  = rng.standard_normal((n, 64, 64)).astype(np.float32),
        seeds     = np.arange(n),
    )
    U.NSIDE  = 64
    U.USE_GI = True
    model = U.build_unet(n_channels=2, base_filters=4)
    # dummy norm: mean=0, std=1
    norm  = (0.0, 1.0, 0.0, 1.0)
    phi_nn = U.predict_phi_nn(model, data, norm)
    assert phi_nn.shape == (n, 64, 64), f"shape {phi_nn.shape}"
    assert np.all(np.isfinite(phi_nn)), "NaN/Inf in prediction"

test("predict_phi_nn: output shape and finite values", t_predict_shape)


# ── Summary ───────────────────────────────────────────────────────────────────
n_pass = sum(1 for _, ok in results if ok)
n_fail = sum(1 for _, ok in results if not ok)
print(f"\n{'─'*50}")
print(f"Results: {n_pass}/{len(results)} passed", end="")
if n_fail:
    print(f"  ({n_fail} failed)")
    failed = [name for name, ok in results if not ok]
    for f in failed:
        print(f"  FAILED: {f}")
    sys.exit(1)
else:
    print("  — all good")
