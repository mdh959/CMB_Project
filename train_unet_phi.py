#!/usr/bin/env python3
"""
train_unet_phi.py
=================
Train a U-Net to predict ϕ_true from ϕ_QE (and optionally ϕ_GI),
using UL (0.1 µK-arcmin) simulations stored in the JLD2 phi-maps file.

Architecture: U-Net (encoder-decoder with skip connections), 512×512 → 512×512.

Diagnostics plotted:
  1. ρ_L  cross-correlation coefficient  (QE, QE WF, U-Net)
  2. W_L  transfer function              (same)
  3. N_L  noise power                    (same)
  4. σ_L  per-sim scatter                (same)
  5. Visual: ϕ_true / ϕ_QE / ϕ_NN / residual maps for one test sim
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── Config ─────────────────────────────────────────────────────────────────────
JLD2_FILE   = "results/phi_maps_qe_gi_12000_ul.jld2"
MODEL_FILE  = "results/unet_phi_ul.keras"
NORM_FILE   = "results/unet_norm_ul.npz"
PLOT_FILE   = "results/unet_diagnostics.png"
TRAIN_CURVE = "results/unet_training_curve.png"

THETA_PIX_ARCMIN = 0.7438046267475303
NSIDE_FULL       = 512
NSIDE            = 256   # downsample from 512 for speed (2x per dim = 4x fewer pixels)
THETA_PIX_RAD    = THETA_PIX_ARCMIN * np.pi / (180.0 * 60.0)
THETA_PIX_RAD_DS = THETA_PIX_RAD * (NSIDE_FULL / NSIDE)  # effective pix size after downsample

DELTA_L    = 400
L_MIN      = 1000
L_MAX      = 12000   # Nyquist limit drops with lower resolution

FAST_TEST  = False    # set False for full training run
if FAST_TEST:
    N_TRAIN    = 50
    BATCH_SIZE = 4
    EPOCHS     = 5
    PATIENCE   = 5
    NSIDE      = 128
else:
    N_TRAIN    = 800     # sims for training; rest → validation + test
    BATCH_SIZE = 4
    EPOCHS     = 300
    PATIENCE   = 20
    NSIDE      = 256
USE_GI     = True    # set False to use only ϕ_QE_WF as input
RETRAIN    = True    # retrain from scratch


# ── Power spectrum on flat patch (2D FFT) ─────────────────────────────────────

def _ell_grid(n):
    """Return the ℓ magnitude for each pixel in a 2D FFT grid."""
    pix = THETA_PIX_RAD_DS if n == NSIDE else THETA_PIX_RAD
    kfreq = np.fft.fftfreq(n, d=pix) * 2.0 * np.pi
    kx, ky = np.meshgrid(kfreq, kfreq, indexing='ij')
    return np.sqrt(kx**2 + ky**2)


_ELL_GRID = None   # cached after first call

def flat_cl(map1, map2=None, delta_l=DELTA_L, l_min=L_MIN, l_max=L_MAX):
    """
    2D FFT cross- (or auto-) power spectrum on a flat patch.
    Returns (ell_centres, C_ell) with binwidth delta_l.
    """
    global _ELL_GRID
    n = map1.shape[0]
    if _ELL_GRID is None:
        _ELL_GRID = _ell_grid(n)
    ell = _ELL_GRID

    norm = THETA_PIX_RAD**2               # physical area per pixel
    f1 = np.fft.fft2(map1) * norm
    f2 = np.fft.fft2(map2) * norm if map2 is not None else f1
    power = np.real(f1 * np.conj(f2)) / (THETA_PIX_RAD * n)**2

    edges = np.arange(l_min, l_max + delta_l, delta_l)
    lc, cl = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (ell >= lo) & (ell < hi)
        if mask.sum() < 4:
            continue
        lc.append(0.5 * (lo + hi))
        cl.append(float(np.mean(power[mask])))
    return np.array(lc), np.array(cl)


def ensemble_spectra(phi_true_arr, phi_hat_arr):
    """
    Compute per-sim (C_tt, C_th, C_hh) for all sims.
    Returns (lc, cl_tt, cl_th, cl_hh) each (nsims, nbins).
    """
    nsims = len(phi_true_arr)
    lc_ref, _ = flat_cl(phi_true_arr[0])
    nb = len(lc_ref)
    cl_tt = np.zeros((nsims, nb))
    cl_th = np.zeros((nsims, nb))
    cl_hh = np.zeros((nsims, nb))
    for i in range(nsims):
        _, cl_tt[i] = flat_cl(phi_true_arr[i])
        _, cl_th[i] = flat_cl(phi_true_arr[i], phi_hat_arr[i])
        _, cl_hh[i] = flat_cl(phi_hat_arr[i])
        if (i + 1) % 100 == 0:
            print(f"  spectra {i+1}/{nsims}")
    return lc_ref, cl_tt, cl_th, cl_hh


def summary_stats(cl_tt, cl_th, cl_hh):
    """
    W_L  = mean_s[ C_th(s)/C_tt(s) ]           (paper eq. 3.1)
    ρ_L  = mean(C_th) / sqrt(mean(C_tt)*mean(C_hh))
    N_L  = mean(C_hh) / W_L^2 - mean(C_tt)     (debiased noise floor)
    σ_L  = std_s[ C_th(s)/mean(C_tt) ]         (per-sim scatter)
    """
    mean_tt  = np.mean(cl_tt, axis=0)
    mean_th  = np.mean(cl_th, axis=0)
    mean_hh  = np.mean(cl_hh, axis=0)
    W_L  = np.mean(cl_th / np.where(cl_tt > 0, cl_tt, np.nan), axis=0)
    rho_L = mean_th / np.sqrt(np.maximum(mean_tt * mean_hh, 1e-300))
    N_L   = mean_hh / np.where(W_L**2 > 1e-30, W_L**2, np.nan) - mean_tt
    sigma_L = np.nanstd(cl_th / np.where(mean_tt > 0, mean_tt, np.nan), axis=0)
    return W_L, rho_L, N_L, sigma_L, mean_tt


# ── Data loading ───────────────────────────────────────────────────────────────

def load_phi_maps(jld2_path):
    """
    Load phi maps from JLD2 (HDF5) file.
    JLD2 stores Julia arrays in column-major order → transpose for numpy.
    """
    print(f"Loading {jld2_path} ...")
    phi_true, phi_qe_wf, phi_gi_b, seeds = [], [], [], []

    with h5py.File(jld2_path, "r") as f:
        sim_keys = sorted(
            [k for k in f.keys() if k.startswith("sim_")],
            key=lambda x: int(x.split("_")[1])
        )
        print(f"  {len(sim_keys)} sims found")
        for key in sim_keys:
            g = f[key]
            # Julia arrays are column-major; .T gives correct C-order
            def _load_ds(arr):
                m = np.array(arr).T.astype(np.float32)
                if NSIDE < NSIDE_FULL:
                    from skimage.transform import resize
                    m = resize(m, (NSIDE, NSIDE), anti_aliasing=True).astype(np.float32)
                return m
            phi_true.append(_load_ds(g["ϕ_true"]))
            if "ϕ_qe_wf" in g: phi_qe_wf.append(_load_ds(g["ϕ_qe_wf"]))
            if "ϕ_gi_b"  in g: phi_gi_b.append( _load_ds(g["ϕ_gi_b"]))
            if "seed"    in g: seeds.append(int(np.array(g["seed"])))

    out = dict(
        phi_true  = np.array(phi_true),
        phi_qe_wf = np.array(phi_qe_wf) if phi_qe_wf else None,
        phi_gi_b  = np.array(phi_gi_b)  if phi_gi_b  else None,
        seeds     = np.array(seeds),
    )
    print(f"  phi_true shape: {out['phi_true'].shape}")
    return out


# ── U-Net architecture ─────────────────────────────────────────────────────────

def _conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def build_unet(n_channels=1, base_filters=32):
    """
    U-Net: 512×512×n_ch  →  512×512×1
    Four encoder stages (stride-2 pool), bottleneck, four decoder stages with
    skip connections.  ~7M parameters with base_filters=32.
    """
    inp = layers.Input(shape=(NSIDE, NSIDE, n_channels))

    # Encoder
    c1 = _conv_block(inp, base_filters * 1);  p1 = layers.MaxPooling2D()(c1)  # 256
    c2 = _conv_block(p1,  base_filters * 2);  p2 = layers.MaxPooling2D()(c2)  # 128
    c3 = _conv_block(p2,  base_filters * 4);  p3 = layers.MaxPooling2D()(c3)  # 64
    c4 = _conv_block(p3,  base_filters * 8);  p4 = layers.MaxPooling2D()(c4)  # 32

    # Bottleneck
    bn = _conv_block(p4,  base_filters * 16)                                   # 32

    # Decoder
    u4 = layers.UpSampling2D()(bn); u4 = layers.Concatenate()([u4, c4])
    d4 = _conv_block(u4, base_filters * 8)                                     # 64

    u3 = layers.UpSampling2D()(d4); u3 = layers.Concatenate()([u3, c3])
    d3 = _conv_block(u3, base_filters * 4)                                     # 128

    u2 = layers.UpSampling2D()(d3); u2 = layers.Concatenate()([u2, c2])
    d2 = _conv_block(u2, base_filters * 2)                                     # 256

    u1 = layers.UpSampling2D()(d2); u1 = layers.Concatenate()([u1, c1])
    d1 = _conv_block(u1, base_filters * 1)                                     # 512

    out = layers.Conv2D(1, 1, activation="linear")(d1)

    model = models.Model(inp, out, name="UNet_phi")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse")
    return model


# ── Training pipeline ──────────────────────────────────────────────────────────

def _build_X(data, use_gi):
    """Build input tensor: (N, NSIDE, NSIDE, n_ch)."""
    channels = [data["phi_qe_wf"][..., np.newaxis]]
    if use_gi and data["phi_gi_b"] is not None:
        channels.append(data["phi_gi_b"][..., np.newaxis])
    return np.concatenate(channels, axis=-1)


def run_training(data):
    X = _build_X(data, USE_GI)
    y = data["phi_true"][..., np.newaxis]

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    tr_idx, val_idx = idx[:N_TRAIN], idx[N_TRAIN:]

    X_tr, y_tr   = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Normalise using training statistics
    X_mean, X_std = float(X_tr.mean()), float(X_tr.std())
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    X_tr  = (X_tr  - X_mean) / X_std;  X_val = (X_val - X_mean) / X_std
    y_tr  = (y_tr  - y_mean) / y_std;  y_val = (y_val - y_mean) / y_std
    np.savez(NORM_FILE, X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)

    if os.path.exists(MODEL_FILE) and not RETRAIN:
        print(f"Loading existing model {MODEL_FILE}")
        model = tf.keras.models.load_model(MODEL_FILE)
        return model, val_idx, (X_mean, X_std, y_mean, y_std)

    model = build_unet(n_channels=X.shape[-1], base_filters=16)
    model.summary()

    cbs = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-7, verbose=1),
    ]
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=cbs, verbose=2,
    )
    model.save(MODEL_FILE)
    print(f"Model saved → {MODEL_FILE}")

    # Training curve
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(hist.history["loss"],     label="train")
    ax.semilogy(hist.history["val_loss"], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss")
    ax.legend(frameon=False); ax.set_title("U-Net training loss")
    fig.tight_layout(); fig.savefig(TRAIN_CURVE, dpi=150); plt.close(fig)
    print(f"Training curve → {TRAIN_CURVE}")

    return model, val_idx, (X_mean, X_std, y_mean, y_std)


def predict_phi_nn(model, data, norm):
    """Predict ϕ_NN for all sims, returned in physical units."""
    X_mean, X_std, y_mean, y_std = norm
    X = (_build_X(data, USE_GI) - X_mean) / X_std
    pred_norm = model.predict(X, batch_size=4, verbose=1)
    return (pred_norm[..., 0] * y_std + y_mean).astype(np.float32)


# ── Diagnostic plots ───────────────────────────────────────────────────────────

def make_diagnostics(data, phi_nn, test_sim_idx):
    phi_true = data["phi_true"]
    nsims    = len(phi_true)

    print("Computing ensemble power spectra (QE WF) …")
    lc, tt_wf, th_wf, hh_wf = ensemble_spectra(phi_true, data["phi_qe_wf"])
    print("Computing ensemble power spectra (GI Boryana) …")
    _,  tt_gi, th_gi, hh_gi = ensemble_spectra(phi_true, data["phi_gi_b"])
    print("Computing ensemble power spectra (U-Net) …")
    _,  tt_nn, th_nn, hh_nn = ensemble_spectra(phi_true, phi_nn)

    W_wf, rho_wf, N_wf, sig_wf, C_tt = summary_stats(tt_wf, th_wf, hh_wf)
    W_gi, rho_gi, N_gi, sig_gi, _    = summary_stats(tt_gi, th_gi, hh_gi)
    W_nn, rho_nn, N_nn, sig_nn, _    = summary_stats(tt_nn, th_nn, hh_nn)

    # κ = −∇²ϕ/2  → C_L^{κκ} = (L²/2)² C_L^{ϕϕ}
    msk  = (lc >= L_MIN) & (lc <= L_MAX)
    L    = lc[msk]
    kfac = (L**2 / 2.0)**2

    cols = {"QE WF": "#2CA02C", "GI Boryana": "#1F77B4", "U-Net": "#9467BD"}
    lw   = 2.0

    fig = plt.figure(figsize=(16, 13))
    gs  = fig.add_gridspec(3, 4, hspace=0.42, wspace=0.36)

    # ── ρ_L ──────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(L, rho_wf[msk], color=cols["QE WF"],     lw=lw, label="QE WF")
    ax.plot(L, rho_gi[msk], color=cols["GI Boryana"], lw=lw, label="GI Boryana")
    ax.plot(L, rho_nn[msk], color=cols["U-Net"],      lw=lw, label="U-Net")
    ax.axhline(1, color="gray", ls=":", lw=1)
    ax.set_xlim(L_MIN, L_MAX); ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("L"); ax.set_ylabel(r"$\rho_L$")
    ax.set_title(r"$\rho_L = \langle C_L^{\phi_{\rm true},\hat\phi}\rangle "
                 r"/ \sqrt{\langle C_L^{\phi\phi}\rangle\langle C_L^{\hat\phi\hat\phi}\rangle}$",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=9)

    # ── W_L ──────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2:])
    ax.plot(L, W_wf[msk], color=cols["QE WF"],     lw=lw, label="QE WF")
    ax.plot(L, W_gi[msk], color=cols["GI Boryana"], lw=lw, label="GI Boryana")
    ax.plot(L, W_nn[msk], color=cols["U-Net"],      lw=lw, label="U-Net")
    ax.axhline(1, color="gray", ls=":", lw=1); ax.axhline(0, color="k", ls="--", lw=0.5, alpha=0.4)
    ax.set_xlim(L_MIN, L_MAX)
    ax.set_xlabel("L"); ax.set_ylabel(r"$W_L$")
    ax.set_title(r"$W_L = \langle C_L^{\phi_{\rm true},\hat\phi} / "
                 r"C_L^{\phi_{\rm true},\phi_{\rm true}} \rangle$", fontsize=9)
    ax.legend(frameon=False, fontsize=9)

    # ── N_L noise power ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    for key, N, c in [("QE WF",     N_wf, cols["QE WF"]),
                      ("GI Boryana", N_gi, cols["GI Boryana"]),
                      ("U-Net",      N_nn, cols["U-Net"])]:
        valid = msk & np.isfinite(N) & (np.abs(N) > 0)
        ax.semilogy(lc[valid], np.abs(kfac) * np.abs(N[valid]),
                    color=c, lw=lw, label=key)
    ax.semilogy(L, kfac * C_tt[msk], color="k", ls="--", lw=1.2, alpha=0.7,
                label=r"$C_L^{\kappa\kappa}$ (signal)")
    ax.set_xlim(L_MIN, L_MAX)
    ax.set_xlabel("L"); ax.set_ylabel(r"$(L^2/2)^2 \, N_L^{\phi}$")
    ax.set_title(r"Noise power  $N_L = \langle C_L^{\hat\phi_{\rm deb},\hat\phi_{\rm deb}}\rangle - C_L^{\phi\phi}$",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=9)

    # ── σ_L ──────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2:])
    ax.semilogy(L, sig_wf[msk], color=cols["QE WF"],     lw=lw, label="QE WF")
    ax.semilogy(L, sig_gi[msk], color=cols["GI Boryana"], lw=lw, label="GI Boryana")
    ax.semilogy(L, sig_nn[msk], color=cols["U-Net"],      lw=lw, label="U-Net")
    ax.set_xlim(L_MIN, L_MAX)
    ax.set_xlabel("L")
    ax.set_ylabel(r"$\sigma\left[C_L^{\phi_{\rm true},\hat\phi} / C_L^{\phi\phi}\right]$")
    ax.set_title("Per-sim scatter of normalised cross-spectrum", fontsize=9)
    ax.legend(frameon=False, fontsize=9)

    # ── Visual maps: ϕ_true / ϕ_QE / ϕ_NN / residual ─────────────────────────
    s    = test_sim_idx
    vmax = float(np.percentile(np.abs(phi_true[s]), 97))
    kw   = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
    panels = [
        (r"$\phi_{\rm true}$",                     phi_true[s],                  kw),
        (r"$\hat\phi_{\rm QE\,WF}$",               data["phi_qe_wf"][s],         kw),
        (r"$\hat\phi_{\rm U-Net}$",                phi_nn[s],                    kw),
        (r"$\phi_{\rm true} - \hat\phi_{\rm NN}$", phi_true[s] - phi_nn[s],      None),
    ]
    for col, (title, m, kw_m) in enumerate(panels):
        ax = fig.add_subplot(gs[2, col])
        if kw_m is None:
            rv = float(np.percentile(np.abs(m), 97))
            im = ax.imshow(m, cmap="RdBu_r", vmin=-rv, vmax=rv, origin="lower")
        else:
            im = ax.imshow(m, **kw_m)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10); ax.axis("off")

    fig.suptitle(
        f"U-Net ϕ reconstruction  (0.1 µK-arcmin, {nsims} sims, test sim {test_sim_idx})",
        fontsize=11, y=0.995
    )
    fig.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnostics → {PLOT_FILE}")

    # ── Summary table ─────────────────────────────────────────────────────────
    check = [2000, 4000, 6000, 8000, 10000, 12000]
    print(f"\n{'L':>7}  {'ρ_WF':>8}  {'ρ_GI':>8}  {'ρ_NN':>8}  "
          f"{'W_WF':>8}  {'W_GI':>8}  {'W_NN':>8}")
    for Lc in check:
        i = int(np.argmin(np.abs(lc - Lc)))
        print(f"{round(lc[i]):>7}  {rho_wf[i]:8.4f}  {rho_gi[i]:8.4f}  {rho_nn[i]:8.4f}"
              f"  {W_wf[i]:8.4f}  {W_gi[i]:8.4f}  {W_nn[i]:8.4f}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    data = load_phi_maps(JLD2_FILE)

    model, val_idx, norm = run_training(data)

    phi_nn = predict_phi_nn(model, data, norm)

    test_sim = int(val_idx[0]) if len(val_idx) > 0 else 0
    make_diagnostics(data, phi_nn, test_sim_idx=test_sim)

    print("\nAll done.")
