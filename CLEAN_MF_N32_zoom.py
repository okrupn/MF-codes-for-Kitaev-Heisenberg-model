#!/usr/bin/env python3
"""
Zoomed (J_perp in [-0.1, 0.1], J_z in [0.1, 0.25], step 0.0025) winning-phase and total-energy maps
for the 32-spin cluster, read from checkpoint.pkl = {(J_perp, J_z): winner info}.
Saves CLEAN_MF_winning_phase_zoom_up.png and CLEAN_MF_total_energy_zoom_up.png.
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

PATTERN_COLORS = {
    'QSL':                  '#2a78d6',
    'QSL (bond-nematic!)':  '#eda100',
    'FM':                   '#4a3aa7',
    'AFM':                 '#e34948',
    'canted/other':         '#1baf7a',
}

# plot names: CLEAN_MF_<name>_up.png
PLOT_NAME_PREFIX = 'CLEAN_MF_'
PLOT_NAME_TAG = '_up'

NX = NY = 4
N_CELLS = NX * NY          # 16 unit cells
N_SPINS = 2 * N_CELLS      # 32 spins
# variational_energy is per unit cell -> total energy = variational_energy * N_CELLS (x16)

# checkpoint.pkl and the PNGs are next to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, 'checkpoint.pkl')
OUT_DIR = SCRIPT_DIR  # PNGs are written next to the script; change if you want elsewhere


def plot_name(stem):
    """'winning_phase_zoom' -> 'CLEAN_MF_winning_phase_zoom_up.png' (inside OUT_DIR)."""
    return os.path.join(OUT_DIR, '%s%s%s.png' % (PLOT_NAME_PREFIX, stem, PLOT_NAME_TAG))


with open(CHECKPOINT_PATH, 'rb') as f:
    results = pickle.load(f)

J_perp_vals = sorted(set(k[0] for k in results))
Jz_vals = sorted(set(k[1] for k in results))
nJp, nJz = len(J_perp_vals), len(Jz_vals)
print('grid:', nJp, 'x', nJz, '=', nJp * nJz, '  (have', len(results), ')')

jp_idx = {v: i for i, v in enumerate(J_perp_vals)}
jz_idx = {v: i for i, v in enumerate(Jz_vals)}

# grids: rows = J_perp (y axis), columns = J_z (x axis)
energy_total = np.full((nJp, nJz), np.nan)
pattern_grid = np.empty((nJp, nJz), dtype=object)
gap_grid = np.full((nJp, nJz), np.nan)
margin_grid = np.full((nJp, nJz), np.nan)
unconverged_grid = np.zeros((nJp, nJz), dtype=bool)

for (jp, jz), info in results.items():
    i, j = jp_idx[jp], jz_idx[jz]
    energy_total[i, j] = info['variational_energy'] * N_CELLS
    pattern_grid[i, j] = info['pattern']
    gap_grid[i, j] = info['gap']
    margin_grid[i, j] = info['energy_margin']
    unconverged_grid[i, j] = info['unconverged']

n_unconverged = int(unconverged_grid.sum())
print('unconverged points:', n_unconverged)

extent = [min(Jz_vals), max(Jz_vals), min(J_perp_vals), max(J_perp_vals)]

# ---------------------------------------------------------------- winning-phase map
patterns_present = sorted(set(pattern_grid.flatten()))
labels = [p for p in PATTERN_COLORS if p in patterns_present]
for p in patterns_present:
    if p not in labels:
        labels.append(p)  # safety net for any unexpected label
label_to_idx = {lab: i for i, lab in enumerate(labels)}
idx_grid = np.vectorize(lambda p: label_to_idx[p])(pattern_grid)
cmap = ListedColormap([PATTERN_COLORS.get(lab, '#888888') for lab in labels])
bounds = np.arange(len(labels) + 1) - 0.5
norm = BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.imshow(idx_grid, origin='lower', extent=extent, aspect='auto', cmap=cmap, norm=norm,
               interpolation='nearest')
ax.set_xlabel(r'$J_z/|K|$')
ax.set_ylabel(r'$J_\perp/|K|$')
ax.set_title('winning phase (zoom, step=0.0025, N=%d spins)' % N_SPINS)
assert im.get_array().shape == (nJp, nJz)  # sanity: rows=J_perp, cols=Jz, matches extent above
handles = [plt.Rectangle((0, 0), 1, 1, color=PATTERN_COLORS.get(lab, '#888888')) for lab in labels]
ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
fig.tight_layout()
out = plot_name('winning_phase_zoom')
fig.savefig(out, dpi=200)
print('saved', os.path.basename(out))

# ---------------------------------------------------------------- total energy map
fig2, ax2 = plt.subplots(figsize=(6, 5.5))
im2 = ax2.imshow(energy_total, origin='lower', extent=extent, aspect='auto', cmap='viridis',
                 interpolation='nearest')
cbar = fig2.colorbar(im2, ax=ax2)
cbar.set_label('total variational energy (%d spins)' % N_SPINS)
ax2.set_xlabel(r'$J_z/|K|$')
ax2.set_ylabel(r'$J_\perp/|K|$')
ax2.set_title('total ground-state energy (zoom, N=%d spins)' % N_SPINS)
fig2.tight_layout()
out = plot_name('total_energy_zoom')
fig2.savefig(out, dpi=200)
print('saved', os.path.basename(out))

# ---------------------------------------------------------------- quick diagnostics
# locate the smallest-gap points and the smallest-margin points, to describe the kink
flat = [(jp, jz, info) for (jp, jz), info in results.items()]
by_gap = sorted(flat, key=lambda t: t[2]['gap'])[:5]
by_margin = sorted(flat, key=lambda t: t[2]['energy_margin'])[:5]
print('\nsmallest-gap points (candidate Majorana gap-closing line):')
for jp, jz, info in by_gap:
    print('  Jperp=%+.4f Jz=%+.4f gap=%.5f pattern=%s' % (jp, jz, info['gap'], info['pattern']))
print('\nsmallest-energy-margin points (candidate branch-crossing / kink line):')
for jp, jz, info in by_margin:
    print('  Jperp=%+.4f Jz=%+.4f margin=%.2e pattern=%s' % (jp, jz, info['energy_margin'], info['pattern']))

n_bond_nematic = sum(1 for info in results.values() if info['pattern'] == 'QSL (bond-nematic!)')
n_afm = sum(1 for info in results.values() if info['pattern'] == 'AFM')
n_qsl = sum(1 for info in results.values() if info['pattern'] == 'QSL')
n_other = len(results) - n_bond_nematic - n_afm - n_qsl
print('\ncounts: QSL=%d  QSL(bond-nematic)=%d  AFM=%d  other=%d' % (n_qsl, n_bond_nematic, n_afm, n_other))#!/usr/bin/env python3
"""
Zoomed (J_perp in [-0.1, 0.1], J_z in [0.1, 0.25], step 0.0025) winning-phase and total-energy maps
for the 32-spin cluster, read from checkpoint.pkl = {(J_perp, J_z): winner info}.
Saves CLEAN_MF_winning_phase_zoom_up.png and CLEAN_MF_total_energy_zoom_up.png.
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

PATTERN_COLORS = {
    'QSL':                  '#2a78d6',
    'QSL (bond-nematic!)':  '#eda100',
    'FM':                   '#4a3aa7',
    'AFM':                 '#e34948',
    'canted/other':         '#1baf7a',
}

# plot names: CLEAN_MF_<name>_up.png
PLOT_NAME_PREFIX = 'CLEAN_MF_'
PLOT_NAME_TAG = '_up'

NX = NY = 4
N_CELLS = NX * NY          # 16 unit cells
N_SPINS = 2 * N_CELLS      # 32 spins
# variational_energy is per unit cell -> total energy = variational_energy * N_CELLS (x16)

# checkpoint.pkl and the PNGs are next to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, 'checkpoint.pkl')
OUT_DIR = SCRIPT_DIR  # PNGs are written next to the script; change if you want elsewhere


def plot_name(stem):
    """'winning_phase_zoom' -> 'CLEAN_MF_winning_phase_zoom_up.png' (inside OUT_DIR)."""
    return os.path.join(OUT_DIR, '%s%s%s.png' % (PLOT_NAME_PREFIX, stem, PLOT_NAME_TAG))


with open(CHECKPOINT_PATH, 'rb') as f:
    results = pickle.load(f)

J_perp_vals = sorted(set(k[0] for k in results))
Jz_vals = sorted(set(k[1] for k in results))
nJp, nJz = len(J_perp_vals), len(Jz_vals)
print('grid:', nJp, 'x', nJz, '=', nJp * nJz, '  (have', len(results), ')')

jp_idx = {v: i for i, v in enumerate(J_perp_vals)}
jz_idx = {v: i for i, v in enumerate(Jz_vals)}

# grids: rows = J_perp (y axis), columns = J_z (x axis)
energy_total = np.full((nJp, nJz), np.nan)
pattern_grid = np.empty((nJp, nJz), dtype=object)
gap_grid = np.full((nJp, nJz), np.nan)
margin_grid = np.full((nJp, nJz), np.nan)
unconverged_grid = np.zeros((nJp, nJz), dtype=bool)

for (jp, jz), info in results.items():
    i, j = jp_idx[jp], jz_idx[jz]
    energy_total[i, j] = info['variational_energy'] * N_CELLS
    pattern_grid[i, j] = info['pattern']
    gap_grid[i, j] = info['gap']
    margin_grid[i, j] = info['energy_margin']
    unconverged_grid[i, j] = info['unconverged']

n_unconverged = int(unconverged_grid.sum())
print('unconverged points:', n_unconverged)

extent = [min(Jz_vals), max(Jz_vals), min(J_perp_vals), max(J_perp_vals)]

# ---------------------------------------------------------------- winning-phase map
patterns_present = sorted(set(pattern_grid.flatten()))
labels = [p for p in PATTERN_COLORS if p in patterns_present]
for p in patterns_present:
    if p not in labels:
        labels.append(p)  # safety net for any unexpected label
label_to_idx = {lab: i for i, lab in enumerate(labels)}
idx_grid = np.vectorize(lambda p: label_to_idx[p])(pattern_grid)
cmap = ListedColormap([PATTERN_COLORS.get(lab, '#888888') for lab in labels])
bounds = np.arange(len(labels) + 1) - 0.5
norm = BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.imshow(idx_grid, origin='lower', extent=extent, aspect='auto', cmap=cmap, norm=norm,
               interpolation='nearest')
ax.set_xlabel(r'$J_z/|K|$')
ax.set_ylabel(r'$J_\perp/|K|$')
ax.set_title('winning phase (zoom, step=0.0025, N=%d spins)' % N_SPINS)
assert im.get_array().shape == (nJp, nJz)  # sanity: rows=J_perp, cols=Jz, matches extent above
handles = [plt.Rectangle((0, 0), 1, 1, color=PATTERN_COLORS.get(lab, '#888888')) for lab in labels]
ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
fig.tight_layout()
out = plot_name('winning_phase_zoom')
fig.savefig(out, dpi=200)
print('saved', os.path.basename(out))

# ---------------------------------------------------------------- total energy map
fig2, ax2 = plt.subplots(figsize=(6, 5.5))
im2 = ax2.imshow(energy_total, origin='lower', extent=extent, aspect='auto', cmap='viridis',
                 interpolation='nearest')
cbar = fig2.colorbar(im2, ax=ax2)
cbar.set_label('total variational energy (%d spins)' % N_SPINS)
ax2.set_xlabel(r'$J_z/|K|$')
ax2.set_ylabel(r'$J_\perp/|K|$')
ax2.set_title('total ground-state energy (zoom, N=%d spins)' % N_SPINS)
fig2.tight_layout()
out = plot_name('total_energy_zoom')
fig2.savefig(out, dpi=200)
print('saved', os.path.basename(out))

# ---------------------------------------------------------------- quick diagnostics
# locate the smallest-gap points and the smallest-margin points, to describe the kink
flat = [(jp, jz, info) for (jp, jz), info in results.items()]
by_gap = sorted(flat, key=lambda t: t[2]['gap'])[:5]
by_margin = sorted(flat, key=lambda t: t[2]['energy_margin'])[:5]
print('\nsmallest-gap points (candidate Majorana gap-closing line):')
for jp, jz, info in by_gap:
    print('  Jperp=%+.4f Jz=%+.4f gap=%.5f pattern=%s' % (jp, jz, info['gap'], info['pattern']))
print('\nsmallest-energy-margin points (candidate branch-crossing / kink line):')
for jp, jz, info in by_margin:
    print('  Jperp=%+.4f Jz=%+.4f margin=%.2e pattern=%s' % (jp, jz, info['energy_margin'], info['pattern']))

n_bond_nematic = sum(1 for info in results.values() if info['pattern'] == 'QSL (bond-nematic!)')
n_afm = sum(1 for info in results.values() if info['pattern'] == 'AFM')
n_qsl = sum(1 for info in results.values() if info['pattern'] == 'QSL')
n_other = len(results) - n_bond_nematic - n_afm - n_qsl
print('\ncounts: QSL=%d  QSL(bond-nematic)=%d  AFM=%d  other=%d' % (n_qsl, n_bond_nematic, n_afm, n_other))
