#!/usr/bin/env python3
"""
lensit_full_comparison.py

QE + iterative MAP via LensIt.  Output files mirror run_qe_gi_wl12k.jl so results
can be compared in plot_qe_gi_sigma_12k.jl (pass as map_wl_file / map_phi_file).

JLD2 note: output .jld2 files are plain HDF5 written by h5py.  JLD2.jl reads
plain HDF5 datasets; groups with the sim_N/key structure match jldopen access.
φ-map RMS may differ from CMBLensing's internal units — check mj_rms threshold
in the plot script (default 1e-4) after first run.

Usage:
  conda run -n lensit python lensit_full_comparison.py
"""

import numpy as np
import os
import h5py

from lensit import get_ellmat
from lensit.ffs_covs.ell_mat import ffs_alm_pyFFTW
from lensit.ffs_covs import ffs_cov
from lensit.misc.misc_utils import gauss_beam
from lensit.sims import ffs_cmbs, ffs_maps
from lensit.ffs_iterators.ffs_iterator import ffs_iterator_pertMF
from lensit.qcinv import ffs_ninv_filt_ideal, chain_samples, opfilt_cinv

# ══════════════════════════════════════════════════════════════════════════════
NOISE_CASE = 'UL'
# ══════════════════════════════════════════════════════════════════════════════

# ── Fixed grid parameters ──────────────────────────────────────────────────
LD_res, HD_res = 9, 9
ellmin         = 100
Lmax_map       = 12000
θpix_arcmin    = 0.7438046267475303
θpix_rad       = θpix_arcmin * np.pi / (180. * 60.)
Δℓ             = 150

# ── Noise / beam ───────────────────────────────────────────────────────────
if NOISE_CASE == 'S4':
    NlevT      = 1.0
    beamFWHM   = 1.0
    N_iter     = 30     # quasi-Newton (BFGS) converges faster per step than Julia's gradient descent (40 steps)
    CG_tol     = 1e-6   # tighter tolerance avoids noisy BFGS gradient → displacement overshoot (was 1e-3)
    CG_maxiter = 500    # allow more iterations to reach the tighter tolerance
    ellmax     = Lmax_map
else:  # UL
    NlevT      = 0.1
    beamFWHM   = 0.3
    N_iter     = 50     # UL: more steps needed; Julia uses 70 gradient steps, BFGS needs fewer
    CG_tol     = 1e-6   # match S4 tolerance
    CG_maxiter = 500
    ellmax     = Lmax_map

nTpix = NlevT / θpix_arcmin

N_sims           = 100 if NOISE_CASE == 'S4' else 200
CHECKPOINT_EVERY = 10

# ── Output files (mirroring run_qe_gi_wl12k.jl naming) ───────────────────
suffix = '' if NOISE_CASE == 'S4' else '_ul'
out_dir         = f'results/lensit/{NOISE_CASE.lower()}'
checkpoint_file = f'{out_dir}/checkpoint.npz'                          # per-sim spectra
qe_wl_file      = f'{out_dir}/WL_lensit_qe{suffix}.jld2'              # mirrors WL_qe_gi_12000.jld2
qe_phi_file     = f'{out_dir}/phi_maps_lensit_qe{suffix}.jld2'        # mirrors phi_maps_qe_gi_12000.jld2
map_wl_file_out = f'{out_dir}/WL_lensit_map{suffix}.jld2'             # mirrors WL_map_12000.jld2
map_phi_file_out= f'{out_dir}/phi_maps_lensit_map{suffix}.jld2'       # mirrors phi_maps_map_12000.jld2
lib_dir_iso     = f'TEMP/lensit_{NOISE_CASE.lower()}_isocov'
lib_dir_sims    = f'TEMP/lensit_{NOISE_CASE.lower()}_sims'
iter_base_dir   = f'TEMP/lensit_{NOISE_CASE.lower()}_iter'

os.makedirs(out_dir, exist_ok=True)
os.makedirs(iter_base_dir, exist_ok=True)

print(f"\n{'='*60}")
print(f"  NOISE_CASE = {NOISE_CASE}")
print(f"  NlevT={NlevT} µK-arcmin   beamFWHM={beamFWHM}'   ellmax={ellmax}")
print(f"  N_iter={N_iter}   N_sims={N_sims}   Δℓ={Δℓ}")
print(f"{'='*60}\n")

# ── Helpers ────────────────────────────────────────────────────────────────
def cli(cl):
    ret = np.zeros_like(cl, dtype=float)
    ret[cl > 0] = 1. / cl[cl > 0]
    return ret

def safe_div(a, b):
    return np.where(np.abs(b) > 0, a / b, 0.)

# JLD2 magic: 512-byte user block that prevents JLD2.jl from auto-upgrading
# plain HDF5 files (which corrupts HDF5 groups → flat key-name string datasets).
_JLD2_MAGIC = b"HDF5-based Julia Data Format, version 0.1.1"
_JLD2_USERBLOCK = _JLD2_MAGIC + b'\x00' * (512 - len(_JLD2_MAGIC))

def _write_jld2_userblock(filename):
    with open(filename, 'r+b') as fh:
        fh.seek(0)
        fh.write(_JLD2_USERBLOCK)

def h5_write_sim(filename, sim_key, arrays):
    """Append per-sim datasets to HDF5 group sim_key (no-op if already present).
    Creates new files with a 512-byte JLD2 user block so JLD2.jl won't auto-upgrade."""
    is_new = not os.path.exists(filename)
    kwargs = {'userblock_size': 512} if is_new else {}
    with h5py.File(filename, 'a', **kwargs) as f:
        if sim_key in f:
            return
        grp = f.require_group(sim_key)
        for k, v in arrays.items():
            grp.create_dataset(k, data=np.asarray(v, dtype=np.float64)
                               if not isinstance(v, (int, np.integer))
                               else np.int64(v))
    if is_new:
        _write_jld2_userblock(filename)

def h5_save_wl(filename, data):
    """Overwrite W_L summary HDF5 (called at each checkpoint)."""
    with h5py.File(filename, 'w') as f:
        for k, v in data.items():
            if v is None:
                continue
            f.create_dataset(k, data=np.asarray(v, dtype=np.float64)
                             if not isinstance(v, (int, np.integer))
                             else np.int64(v))

# ── Cls ────────────────────────────────────────────────────────────────────
# Use Julia/CMBLensing CAMB Cls if available (generated by export_julia_cls.jl),
# otherwise fall back to lensit's built-in Planck 2015 fiducial Cls.
_julia_cls_path = 'results/lensit/julia_cls.npz'

def _pad(arr, n):
    out = np.zeros(n); m = min(len(arr), n); out[:m] = arr[:m]; return out

if os.path.exists(_julia_cls_path):
    print(f"Loading Julia/CMBLensing Cls from {_julia_cls_path}")
    _jcls = np.load(_julia_cls_path)
    cl_tt_unl = _pad(_jcls['cl_tt_unl'], ellmax + 1)
    cl_tt_len = _pad(_jcls['cl_tt_len'], ellmax + 1)
    cl_pp     = _pad(_jcls['cl_pp'],     ellmax + 1)
    # Julia exports up to ell=12000 so no power-law extrapolation needed
else:
    print("julia_cls.npz not found — using lensit Planck 2015 fiducial Cls")
    print("  Run 'julia export_julia_cls.jl' and clear TEMP/ to use Julia Cls")
    from lensit import get_fidcls as _get_fidcls
    _li_unl, _li_len = _get_fidcls(ellmax_sky=min(ellmax, 8000))
    cl_tt_unl = _pad(_li_unl['tt'], ellmax + 1)
    cl_tt_len = _pad(_li_len['tt'], ellmax + 1)
    cl_pp     = _pad(_li_unl['pp'], ellmax + 1)
    # Power-law extrapolate cl_pp beyond ell=8000 (where _get_fidcls is capped).
    # Zero prior at high L leaves BFGS unconstrained → displacement Jacobian goes negative.
    _ell_cap = min(8000, len(_li_unl['pp']) - 1)
    if ellmax > _ell_cap:
        _L1, _L2 = int(0.75 * _ell_cap), _ell_cap
        _slope = np.log(cl_pp[_L2] / cl_pp[_L1]) / np.log(_L2 / _L1)
        _Ls = np.arange(_ell_cap + 1, ellmax + 1, dtype=float)
        cl_pp[_ell_cap + 1:] = cl_pp[_ell_cap] * (_Ls / _ell_cap) ** _slope

cls_unl = {'tt': cl_tt_unl, 'ee': np.zeros(ellmax+1),
           'bb': np.zeros(ellmax+1), 'te': np.zeros(ellmax+1), 'pp': cl_pp}
cls_len = {'tt': cl_tt_len, 'ee': np.zeros(ellmax+1),
           'bb': np.zeros(ellmax+1), 'te': np.zeros(ellmax+1)}
cl_transf  = gauss_beam(beamFWHM / 60. * np.pi / 180., lmax=ellmax)
cls_noise  = {'t': (NlevT * np.pi / (180. * 60.))**2 * np.ones(ellmax + 1)}

# ── LensIt libraries ───────────────────────────────────────────────────────
print("Building LensIt libraries ...")
ellmat_obj = get_ellmat(LD_res, HD_res=HD_res)
lib_alm    = ffs_alm_pyFFTW(ellmat_obj,
                filt_func=lambda ell: (ell >= ellmin) & (ell <= ellmax))
isocov     = ffs_cov.ffs_diagcov_alm(
                lib_dir_iso, lib_alm, cls_unl, cls_len, cl_transf, cls_noise)
lib_qlm    = isocov.lib_skyalm

print("Computing N0 curves ...")
N0_unl = isocov.get_N0cls('T', lib_qlm, use_cls_len=False)[0]
N0_len = isocov.get_N0cls('T', lib_qlm, use_cls_len=True)[0]
cpp    = cl_pp[:lib_qlm.ellmax + 1]
H0_len = cli(N0_len[:lib_qlm.ellmax + 1])
H0_unl = cli(N0_unl[:lib_qlm.ellmax + 1])
print(f"  lib_qlm.ellmax = {lib_qlm.ellmax}")

# ── Simulation libraries ───────────────────────────────────────────────────
print("Building simulation libraries ...")
lib_lencmb = ffs_cmbs.sims_cmb_len(
    os.path.join(lib_dir_sims, 'lencmbs'), lib_alm, cls_unl, cache_lens=True)
lib_sims = ffs_maps.lib_noisemap(
    os.path.join(lib_dir_sims, 'maps'),
    lib_alm, lib_lencmb, cl_transf, nTpix, nTpix, nTpix)

# ── Iterator filter ────────────────────────────────────────────────────────
filt_ideal = ffs_ninv_filt_ideal.ffs_ninv_filt(
    lib_alm, lib_alm, cls_unl, cl_transf, NlevT, nlev_p=1e6)
chain_descr = chain_samples.get_isomgchain(
    filt_ideal.lib_skyalm.ellmax, filt_ideal.lib_datalm.shape,
    tol=CG_tol, iter_max=CG_maxiter)
opfilt_cinv._type = 'T'

# ── Binning ────────────────────────────────────────────────────────────────
edges   = np.arange(ellmin, ellmax + 1, Δℓ)
ell_lo, ell_hi = edges[:-1], edges[1:]
ell_mid = 0.5 * (ell_lo + ell_hi)
nbins   = len(ell_mid)

lx    = 2 * np.pi * np.fft.fftfreq(512, d=θpix_rad)
ell2d = np.sqrt(lx[None, :]**2 + lx[:, None]**2)

# Proper flat-sky power spectrum normalisation: C_L = (θpix/N)^2 × mean|FFT|^2
_cl_norm = (θpix_rad / 512) ** 2

def cross_cl_binned(m1, m2):
    cross = np.real(np.fft.fft2(m1) * np.conj(np.fft.fft2(m2))) * _cl_norm
    out   = np.zeros(nbins)
    for i, (lo, hi) in enumerate(zip(ell_lo, ell_hi)):
        mask   = (ell2d >= lo) & (ell2d < hi)
        out[i] = cross[mask].mean() if mask.any() else 0.
    return out

# κ conversion: C_L^κκ = (L^2/2)^2 × C_L^φφ
kfac = (ell_mid ** 2 / 2.) ** 2

# σ rescaling to f_sky=0.4 (matching Julia plot script convention)
f_sky_patch = (512 * θpix_rad) ** 2 / (4 * np.pi)
σ_scale     = np.sqrt(f_sky_patch / 0.4)

# Theory C_L^κκ at bin centres
_kfac_th    = np.where(ell_mid > 0, (ell_mid**2 / 2.)**2, 0.)
cl_kk_theory_binned = _kfac_th * np.interp(ell_mid, np.arange(len(cl_pp)), cl_pp)

# ── Resume checkpoint ──────────────────────────────────────────────────────
cl_tt_sims    = []   # per-sim C(φ_true, φ_true)
cl_tqe_sims   = []   # per-sim C(φ_true, φ_QE)
cl_tmap_sims  = []   # per-sim C(φ_true, φ_MAP)
cl_qeqe_sims  = []   # per-sim C(φ_QE,   φ_QE)
cl_mapmap_sims= []   # per-sim C(φ_MAP,  φ_MAP)
R_mj_sims     = []   # per-sim C_cross(true,MAP) / C_true  (for W_mj median, matching Julia)
sum_R_qe_raw  = np.zeros(nbins)   # for W_qe_raw mean (matching Julia)
nsims_done    = 0

if os.path.exists(checkpoint_file):
    d = np.load(checkpoint_file)
    cl_tt_sims    = list(d['cl_tt_sims'])
    cl_tqe_sims   = list(d['cl_tqe_sims'])
    cl_tmap_sims  = list(d['cl_tmap_sims'])
    cl_qeqe_sims  = list(d['cl_qeqe_sims'])
    cl_mapmap_sims= list(d['cl_mapmap_sims'])
    R_mj_sims     = list(d['R_mj_sims'])
    sum_R_qe_raw  = d['sum_R_qe_raw']
    nsims_done    = int(d['nsims_done'])
    print(f"Resumed checkpoint: {nsims_done}/{N_sims} sims done")

# ── Main loop ──────────────────────────────────────────────────────────────
for idx in range(nsims_done, N_sims):
    print(f"\n── Sim {idx+1}/{N_sims} (idx={idx}) ──", flush=True)
    sim_key = f'sim_{idx+1}'   # 1-indexed, matching Julia convention

    plm_true = lib_lencmb.get_sim_plm(idx)
    phi_true = lib_alm.alm2map(plm_true)

    T_obs   = lib_sims.get_sim_tmap(idx)
    datalms = np.array([lib_alm.map2alm(T_obs)])

    # QE: lensed filter + unlensed kernel (= weights=:unlensed in Julia)
    iblm_unl = isocov.get_iblms('T', datalms, use_cls_len=True)[0]
    plm_qe   = 0.5 * isocov.get_qlms('T', iblm_unl, lib_qlm, use_cls_len=False)[0]
    lib_qlm.almxfl(plm_qe, N0_unl[:lib_qlm.ellmax + 1], inplace=True)
    phi_qe   = lib_qlm.alm2map(plm_qe)

    # WF warm start for iterator: WF weight = N0_len*Cpp/(N0_len+Cpp) on unnorm QE
    iblm_len = isocov.get_iblms('T', datalms, use_cls_len=True)[0]
    plm_wf   = 0.5 * isocov.get_qlms('T', iblm_len, lib_qlm, use_cls_len=True)[0]
    wf_weight = np.where(cpp > 0, cli(H0_len + cli(cpp)), 0.)
    lib_qlm.almxfl(plm_wf, wf_weight, inplace=True)

    # Iterator — retry once from zero if displacement inversion fails (negative Jacobian)
    import shutil
    iter_dir = os.path.join(iter_base_dir, f'sim{idx:04d}')
    os.makedirs(iter_dir, exist_ok=True)
    for _attempt in range(2):
        try:
            _plm_start = plm_wf if _attempt == 0 else np.zeros_like(plm_wf)
            if _attempt == 1:
                print(f"  Retrying sim {idx+1} from zero initialisation (stale cache cleared)", flush=True)
                shutil.rmtree(iter_dir, ignore_errors=True)
                os.makedirs(iter_dir, exist_ok=True)
            itlib = ffs_iterator_pertMF(
                iter_dir, 'T', filt_ideal, datalms, lib_qlm,
                _plm_start, H0_unl, cpp,
                chain_descr=chain_descr, opfilt=opfilt_cinv, verbose=False)
            itlib.iterate(0, 'p')
            for it in range(1, N_iter + 1):
                itlib.iterate(it, 'p')
                if it % 5 == 0:
                    print(f"    it={it}/{N_iter}", flush=True)
            break  # success
        except AssertionError as _e:
            if _attempt == 0 and 'Negative value in det' in str(_e):
                print(f"  WARNING: displacement inversion failed (sim {idx+1}, attempt 1): {_e}", flush=True)
                continue
            raise  # re-raise on second failure or unrelated assertion
    phi_map = itlib.get_Phimap(N_iter, 'p')

    # Per-sim spectra (φ units, properly normalised)
    cl_tt    = cross_cl_binned(phi_true, phi_true)
    cl_tqe   = cross_cl_binned(phi_true, phi_qe)
    cl_tmap  = cross_cl_binned(phi_true, phi_map)
    cl_qeqe  = cross_cl_binned(phi_qe,   phi_qe)
    cl_mapmap= cross_cl_binned(phi_map,  phi_map)

    cl_tt_sims.append(cl_tt);    cl_tqe_sims.append(cl_tqe)
    cl_tmap_sims.append(cl_tmap); cl_qeqe_sims.append(cl_qeqe)
    cl_mapmap_sims.append(cl_mapmap)
    R_mj_sims.append(safe_div(cl_tmap, cl_tt))    # per-sim C_cross/C_true ratio
    sum_R_qe_raw += safe_div(cl_tqe, cl_tt)
    nsims_done += 1

    # Per-sim phi maps → HDF5 (JLD2-compatible, mirroring run_qe_gi_wl12k.jl)
    h5_write_sim(qe_phi_file, sim_key, {
        'ϕ_true':   phi_true,
        'ϕ_qe_raw': phi_qe,
        'seed':     idx + 1,
    })
    h5_write_sim(map_phi_file_out, sim_key, {
        'ϕ_true': phi_true,
        'ϕ_mj':   phi_map,
        'seed':   idx + 1,
    })

    # Checkpoint
    if nsims_done % CHECKPOINT_EVERY == 0 or nsims_done == N_sims:
        n = nsims_done
        A = np.array(cl_tt_sims); B = np.array(cl_tqe_sims); C = np.array(cl_tmap_sims)
        D = np.array(cl_qeqe_sims); E = np.array(cl_mapmap_sims)

        W_qe_raw = sum_R_qe_raw / n
        # W_mj: per-bin median across sims (matching Julia convention — robust against C_true→0)
        W_mj = np.array([np.median([r[i] for r in R_mj_sims]) for i in range(nbins)])

        # Per-sim spectra checkpoint
        np.savez(checkpoint_file,
                 cl_tt_sims=A, cl_tqe_sims=B, cl_tmap_sims=C,
                 cl_qeqe_sims=D, cl_mapmap_sims=E,
                 R_mj_sims=np.array(R_mj_sims),
                 sum_R_qe_raw=sum_R_qe_raw,
                 ell_mid=ell_mid, nsims_done=nsims_done,
                 N_iter=N_iter, NlevT=NlevT, beamFWHM=beamFWHM, ellmax=ellmax)

        # W_L files (JLD2-compatible HDF5, mirroring run_qe_gi_wl12k.jl)
        h5_save_wl(qe_wl_file, {
            'ℓ_template':      ell_mid,
            'W_qe_raw':        W_qe_raw,
            'sum_R_qe_raw':    sum_R_qe_raw,
            'nsims_completed': n,
        })
        h5_save_wl(map_wl_file_out, {
            'ℓ_template':    ell_mid,
            'W_mj':          W_mj,
            'R_mj_sims':     np.array(R_mj_sims),
            'nsims_map_done': n,
        })

        # Console summary
        dqe   = np.sqrt(np.maximum(A.mean(0) * D.mean(0), 0.))
        dmap  = np.sqrt(np.maximum(A.mean(0) * E.mean(0), 0.))
        rho_qe  = np.where(dqe  > 0, B.mean(0) / dqe,  0.)
        rho_map = np.where(dmap > 0, C.mean(0) / dmap, 0.)
        print(f"\n  ── Checkpoint @ {n} sims ──")
        print(f"  {'L':>7}  {'W_QE':>8}  {'W_MAP':>8}  {'rho_QE':>8}  {'rho_MAP':>8}")
        for lchk in [500, 1000, 2000, 3000, 5000, 7000, 9000, 11000]:
            if lchk > ellmax: continue
            i = np.argmin(np.abs(ell_mid - lchk))
            print(f"  {int(ell_mid[i]):>7d}  {W_qe_raw[i]:>8.4f}  {W_mj[i]:>8.4f}"
                  f"  {rho_qe[i]:>8.4f}  {rho_map[i]:>8.4f}")

    print(f"  sim {idx+1} done", flush=True)

print("\nAll sims done. Making final plots ...")

# ── Final 4-panel plot ─────────────────────────────────────────────────────
import matplotlib.pyplot as plt

A = np.array(cl_tt_sims); B = np.array(cl_tqe_sims); C = np.array(cl_tmap_sims)
D = np.array(cl_qeqe_sims); E = np.array(cl_mapmap_sims)
n = len(A)

kB = B * kfac;  kC = C * kfac    # κ-unit cross spectra

W_qe_raw = sum_R_qe_raw / n
W_mj     = np.array([np.median([r[i] for r in R_mj_sims]) for i in range(nbins)])

dqe  = np.sqrt(np.maximum(A.mean(0) * D.mean(0), 0.))
dmap = np.sqrt(np.maximum(A.mean(0) * E.mean(0), 0.))
rho_qe  = np.where(dqe  > 0, B.mean(0) / dqe,  0.)
rho_map = np.where(dmap > 0, C.mean(0) / dmap, 0.)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'LensIt {NOISE_CASE} ({NlevT} µK-arcmin, {beamFWHM}\' beam)  —  {n} sims', fontsize=13)

ax = axes[0, 0]
ax.plot(ell_mid, kB.std(0) * σ_scale, 'b-o', ms=3, lw=1.5, label='QE')
ax.plot(ell_mid, kC.std(0) * σ_scale, 'r-s', ms=3, lw=1.5, label=f'MAP it={N_iter}')
ax.set_xlabel(r'$L$'); ax.set_ylabel(r'$\sigma(C_L^{\kappa\kappa,\,\rm cross})$')
ax.set_title(r'$\sigma_{\rm cross}$ (rescaled to $f_{\rm sky}=0.4$)')
ax.set_xlim(ellmin, ellmax); ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(ell_mid, cl_kk_theory_binned, 'k--', lw=1.5, label='Theory')
ax.plot(ell_mid, kB.mean(0), 'b-o', ms=3, lw=1.5, label='QE cross')
ax.plot(ell_mid, kC.mean(0), 'r-s', ms=3, lw=1.5, label=f'MAP cross it={N_iter}')
ax.set_xlabel(r'$L$'); ax.set_ylabel(r'$C_L^{\kappa\kappa}$')
ax.set_title(r'Mean cross power $\langle C_L^{\kappa_{\rm true}\hat\kappa}\rangle$')
ax.set_xlim(ellmin, ellmax); ax.set_yscale('log'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(ell_mid, W_qe_raw, 'b-o', ms=3, lw=1.5, label='QE')
ax.plot(ell_mid, W_mj,     'r-s', ms=3, lw=1.5, label=f'MAP it={N_iter}')
ax.axhline(1.0, color='k', lw=0.8, ls='--')
ax.set_xlabel(r'$L$'); ax.set_ylabel(r'$W_L$')
ax.set_title(r'Transfer function $W_L$')
ax.set_xlim(ellmin, ellmax); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(ell_mid, rho_qe,  'b-o', ms=3, lw=1.5, label='QE')
ax.plot(ell_mid, rho_map, 'r-s', ms=3, lw=1.5, label=f'MAP it={N_iter}')
ax.set_xlabel(r'$L$'); ax.set_ylabel(r'$\rho(L)$')
ax.set_title(r'Reconstruction correlation $\rho(L)$')
ax.set_xlim(ellmin, ellmax); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
outpng = f'{out_dir}/summary_{n}sims.png'
plt.savefig(outpng, dpi=150); plt.close()
print(f"Plot → {outpng}")
print(f"JLD2 files → {qe_phi_file}, {map_phi_file_out}, {qe_wl_file}, {map_wl_file_out}")
