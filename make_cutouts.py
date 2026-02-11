#!/usr/bin/env python3
"""
Extract flat-sky cutouts around massive halos from AbacusSummit full-sky maps.

Run ON CSD3:
    module load python/3.11.0-icl
    python3 make_cutouts.py
cd ~/CMB_Project
sbatch submit_cutouts.sh

Reads from /home/bth26/ and saves cutouts to results/cutouts/
"""

import numpy as np
import healpy as hp
import os

# ── Paths ──
DATA_DIR = "/home/bth26"
OUT_DIR = os.path.expanduser("~/CMB_Project/results/cutouts")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Parameters ──
NSIDE = 8192
CUTOUT_SIZE_DEG = 4.0       # side length of cutout in degrees
CUTOUT_NPIX = 512           # pixels per side
RESO_ARCMIN = CUTOUT_SIZE_DEG * 60.0 / CUTOUT_NPIX  # arcmin per pixel
N_SELECT = 200              # number of halos to select (top by kappa)

# ── Step 1: Load halo catalog ──
print("Loading halo catalog...")
halos = np.load(os.path.join(DATA_DIR, "halos_AbacusSummit_huge_c000_ph201_3.e12.npz"))
print(f"  Keys: {list(halos.keys())}")

ra = halos['ra']
dec = halos['dec']
redshift = halos['redshift']
print(f"  Total halos: {len(ra):,} (all > 3e12 Msun/h)")
print(f"  RA range: {ra.min():.1f} to {ra.max():.1f} deg")
print(f"  Dec range: {dec.min():.1f} to {dec.max():.1f} deg")
print(f"  Redshift range: {redshift.min():.3f} to {redshift.max():.3f}")

# ── Step 2: Load kappa map (needed for halo selection) ──
print("\nLoading kappa alm...")
kappa_alm = np.load(os.path.join(DATA_DIR, "kappa_alm_8192_ph201.npy"))
print(f"  lmax = {hp.Alm.getlmax(len(kappa_alm))}")
print("  Converting alm -> map...")
kappa_map = hp.alm2map(kappa_alm, nside=NSIDE)
del kappa_alm  # free memory

# ── Step 3: Select halos by kappa at halo position (mass proxy) ──
# No mass column available, so use kappa value at each halo's location
# as a proxy — high kappa = more matter along the line of sight
print("\nSelecting halos by kappa (mass proxy)...")
theta = np.radians(90.0 - dec)   # colatitude
phi = np.radians(ra)             # longitude
pix = hp.ang2pix(NSIDE, theta, phi)
kappa_at_halo = kappa_map[pix]

# Restrict to z < 0.8 and take the top N by kappa
mask = redshift < 0.8
candidates = np.where(mask)[0]
print(f"  Candidates (z < 0.8): {len(candidates):,}")

top_idx = candidates[np.argsort(kappa_at_halo[candidates])[::-1]]
sel = top_idx[:N_SELECT]

ra_sel = ra[sel]
dec_sel = dec[sel]
redshift_sel = redshift[sel]
kappa_sel = kappa_at_halo[sel]
print(f"  Selected top {len(sel)} halos by kappa")
print(f"  Kappa range of selected: {kappa_sel.min():.4f} to {kappa_sel.max():.4f}")
print(f"  Redshift range of selected: {redshift_sel.min():.3f} to {redshift_sel.max():.3f}")

# ── Step 4: Load CMB maps ──
print("\nLoading lensed CMB map...")
lensed_map = hp.read_map(os.path.join(DATA_DIR, "lensed_map_8192_hp_new_full.fits"), field=0)

print("Loading unlensed CMB map...")
unlensed_map = hp.read_map(os.path.join(DATA_DIR, "unlensed_map_8192_hp_new_full.fits"), field=0)

# ── Step 5: Extract cutouts via gnomonic projection ──
n = len(sel)
print(f"\nExtracting {n} cutouts ({CUTOUT_SIZE_DEG} deg x {CUTOUT_SIZE_DEG} deg, {CUTOUT_NPIX}x{CUTOUT_NPIX} pix)...")

kappa_cutouts = np.empty((n, CUTOUT_NPIX, CUTOUT_NPIX), dtype=np.float64)
lensed_cutouts = np.empty_like(kappa_cutouts)
unlensed_cutouts = np.empty_like(kappa_cutouts)

for i in range(n):
    if (i + 1) % 10 == 0 or i == 0:
        print(f"  Cutout {i+1}/{n}  (ra={ra_sel[i]:.2f}, dec={dec_sel[i]:.2f}, z={redshift_sel[i]:.3f}, kappa={kappa_sel[i]:.4f})")

    kappa_cutouts[i] = hp.gnomview(
        kappa_map, rot=(ra_sel[i], dec_sel[i]),
        xsize=CUTOUT_NPIX, ysize=CUTOUT_NPIX,
        reso=RESO_ARCMIN, return_projected_map=True, no_plot=True
    ).astype(np.float64)
    lensed_cutouts[i] = hp.gnomview(
        lensed_map, rot=(ra_sel[i], dec_sel[i]),
        xsize=CUTOUT_NPIX, ysize=CUTOUT_NPIX,
        reso=RESO_ARCMIN, return_projected_map=True, no_plot=True
    ).astype(np.float64)
    unlensed_cutouts[i] = hp.gnomview(
        unlensed_map, rot=(ra_sel[i], dec_sel[i]),
        xsize=CUTOUT_NPIX, ysize=CUTOUT_NPIX,
        reso=RESO_ARCMIN, return_projected_map=True, no_plot=True
    ).astype(np.float64)

# ── Step 6: Save ──
outfile = os.path.join(OUT_DIR, "cluster_cutouts.npz")
print(f"\nSaving to {outfile}...")
np.savez_compressed(outfile,
    ra=ra_sel,
    dec=dec_sel,
    redshift=redshift_sel,
    kappa_at_halo=kappa_sel,
    cutout_size_deg=np.float64(CUTOUT_SIZE_DEG),
    cutout_npix=np.int64(CUTOUT_NPIX),
    reso_arcmin=np.float64(RESO_ARCMIN),
    nside_source=np.int64(NSIDE),
    kappa=kappa_cutouts,
    lensed=lensed_cutouts,
    unlensed=unlensed_cutouts,
)

filesize_mb = os.path.getsize(outfile) / 1e6
print(f"Done! Saved {n} cutouts to {outfile} ({filesize_mb:.1f} MB)")
