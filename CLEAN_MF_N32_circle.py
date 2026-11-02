"""
Circle scan of the zero-field Kitaev-Heisenberg(XXZ) model on the 32-spin cluster, using CLEAN_MF_N32_2d.py:
J_perp_j = R cos(pi j/20), J_z_j = R sin(pi j/20), R = 10 K, j = 0..39; multi-seed solve at every point.
Plots: paper_plots/CLEAN_MF_circle_R10_phase_up.png, CLEAN_MF_circle_R10_energy_plane_up.png;
checkpoint + text table in paper_plots/checkpoints/.
"""
import os
import sys
import importlib.util
import contextlib
import pickle
import multiprocessing as mp

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Mean-field module: MF_FILE = full path, or None to search this script's folder for the names below.
MF_FILE = None
MF_CANDIDATE_NAMES = ('CLEAN_MF_N32_2d.py', 'CLEAN_mean-field_N_32.py',
                      'KJGamma_MF_aniso_JperpJz_saving_N32.py')


def _find_mf_file():
    if MF_FILE is not None:
        if not os.path.isfile(MF_FILE):
            raise FileNotFoundError('MF_FILE = %r does not exist.' % MF_FILE)
        return MF_FILE
    here = os.path.dirname(os.path.abspath(__file__))
    for name in MF_CANDIDATE_NAMES:
        path = os.path.join(here, name)
        if os.path.isfile(path):
            return path
    py_files = sorted(f for f in os.listdir(here) if f.endswith('.py'))
    raise FileNotFoundError(
        'Mean-field module not found in %s.\nLooked for: %s\nPython files in this folder: %s\n'
        'Copy CLEAN_MF_N32_2d.py into this folder, or set MF_FILE at the top of this '
        'script to its full path.' % (here, ', '.join(MF_CANDIDATE_NAMES), ', '.join(py_files)))


_spec = importlib.util.spec_from_file_location('CLEAN_MF_N32_2d', _find_mf_file())
MF = importlib.util.module_from_spec(_spec)
sys.modules['CLEAN_MF_N32_2d'] = MF
_spec.loader.exec_module(MF)

# Configuration
K = 1.0                      # Kitaev coupling, Kx = Ky = Kz = K (this code's convention H = +K S S)
GAMMA = 0.0                  # off-diagonal Gamma
RADII = [10.0]               # circle radii in units of K (whiteboard: 10 K); e.g. [1, 2, 5, 10]
N_ANGLES = 40                # theta_j = 2*pi*j/N_ANGLES = pi*j/20  for N_ANGLES = 40
NX = NY = MF.Nx              # 4 x 4 unit cells = 32 spins
KX_SHIFT, KY_SHIFT = MF.KX_SHIFT, MF.KY_SHIFT   # (0, 0.5): physical sector of the 32-spin torus
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)
TAG = MF.PLOT_NAME_TAG       # '_up'
PREFIX = MF.PLOT_NAME_PREFIX  # 'CLEAN_MF_'
OUT_DIR = os.path.join(os.getcwd(), 'paper_plots')
CHECKPOINT_DIR = os.path.join(OUT_DIR, 'checkpoints')

# moment direction: |cos(angle to z)| > DIR_Z_TOL -> 'm || z', < DIR_XY_TOL -> 'm in xy', else 'tilted'
DIR_Z_TOL = 0.9
DIR_XY_TOL = 0.1
M_TOL = 1e-3
M0_STRONG = 0.9              # second, near-saturated seed magnitude (see solve_point)
MAX_REFINE_SWEEPS = 6        # neighbour-continuation sweeps (see refine_scan)


def circle_points(radii=RADII, n_angles=N_ANGLES, K=K):
    """(R, j, theta, J_perp, J_z) for every radius and angle index j = 0 .. n_angles-1."""
    pts = []
    for R in radii:
        for j in range(n_angles):
            theta = 2.0 * np.pi * j / n_angles
            pts.append((float(R), j, theta, R * K * np.cos(theta), R * K * np.sin(theta)))
    return pts


def moment_direction(mA, mB):
    """'-' (no order), 'm || z', 'm in xy' or 'tilted', from the larger sublattice moment."""
    m = mA if np.linalg.norm(mA) >= np.linalg.norm(mB) else mB
    norm = np.linalg.norm(m)
    if norm < M_TOL:
        return '-'
    cz = abs(m[2]) / norm
    if cz > DIR_Z_TOL:
        return 'm || z'
    if cz < DIR_XY_TOL:
        return 'm in xy'
    return 'tilted'


def _model(J_perp, Jz):
    MF.system_parameters.kx_shift, MF.system_parameters.ky_shift = KX_SHIFT, KY_SHIFT
    params = MF.Parameters(NX, NY, np.array([K, K, K]), np.array([J_perp, J_perp, Jz]), GAMMA)
    return MF.Majorana_rep(params)


def _pack(R, j, theta, J_perp, Jz, best, branches):
    mA, mB = np.asarray(best['m_A'], float), np.asarray(best['m_B'], float)
    nA, nB = np.linalg.norm(mA), np.linalg.norm(mB)
    n_cells = NX * NY
    return {
        'R': R, 'j': j, 'theta': theta, 'J_perp': J_perp, 'Jz': Jz,
        'pattern': best['pattern'], 'direction': moment_direction(mA, mB),
        'E_cell': best['variational_energy'], 'E_total': best['variational_energy'] * n_cells,
        'm_A': mA, 'm_B': mB, '|m_A|': nA, '|m_B|': nB,
        'cos_AB': float(mA @ mB / (nA * nB)) if nA > M_TOL and nB > M_TOL else np.nan,
        'gap': best['gap'], 'residual': best.get('residual', np.nan),
        'unconverged': best.get('unconverged', False),
        'chi00_xyz': best['chi00_xyz'],
        'seed': best.get('seed', ''),
        'MF_parameters': np.asarray(best['MF_parameters'], float),
        'branches': sorted((b['seed'], b['pattern'], b['variational_energy'] * n_cells)
                           for b in branches),
    }


def solve_point(args):
    """Worker: multi-seed solve at one point, with seed moments m0 = 0.3 and M0_STRONG; keep the best."""
    R, j, theta, J_perp, Jz = args
    model = _model(J_perp, Jz)
    best, branches = None, []
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        for m0 in (0.3, M0_STRONG):
            try:
                b, br = model.solve_best_branch(m0=m0, verbose=False)
            except RuntimeError:
                continue
            for x in br:
                x['seed'] = '%s(m0=%g)' % (x['seed'], m0)
            branches += br
            b['seed'] = '%s(m0=%g)' % (b['seed'], m0)
            # prefer a converged solution; among equally converged ones, the lower energy
            key = (bool(b.get('unconverged')), b['variational_energy'])
            if best is None or key < (bool(best.get('unconverged')), best['variational_energy']):
                best = b
    return _pack(R, j, theta, J_perp, Jz, best, branches)


def refine_point(args):
    """Worker: re-solve one point from its neighbours' solutions; return it only if converged and lower in energy."""
    p, starts = args
    model = _model(p['J_perp'], p['Jz'])
    best = None
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        for label, x0 in starts:
            model.MF_parameters = np.array(x0, float)
            try:
                model.Self_Consistent_Solution()
            except Exception:
                continue
            res = float(np.max(np.abs(model.MF_conditions(model.MF_parameters))))
            if res > 1e-6:
                continue
            info = model.classify_solution()
            info['residual'] = res
            info['seed'] = label
            info['MF_parameters'] = np.array(model.MF_parameters)
            if best is None or info['variational_energy'] < best['variational_energy']:
                best = info
    if best is not None and best['variational_energy'] * NX * NY < p['E_total'] - 1e-7:
        new = _pack(p['R'], p['j'], p['theta'], p['J_perp'], p['Jz'], best, [])
        new['branches'] = p['branches'] + [(best['seed'], best['pattern'], new['E_total'])]
        return new
    return None


def refine_scan(points, n_workers=N_WORKERS, max_sweeps=MAX_REFINE_SWEEPS, verbose=True):
    """Repeat neighbour warm-start sweeps until no point lowers its energy (or max_sweeps)."""
    by_key = {(p['R'], p['j']): p for p in points}
    radii = sorted({p['R'] for p in points})
    for sweep in range(max_sweeps):
        jobs = []
        for (R, j), p in by_key.items():
            starts = []
            for dj in (-1, 1):
                q = by_key.get((R, (j + dj) % N_ANGLES))
                if q is not None:
                    starts.append(('from j=%d' % q['j'], q['MF_parameters']))
            k = radii.index(R)
            for dk in (-1, 1):
                if 0 <= k + dk < len(radii):
                    q = by_key.get((radii[k + dk], j))
                    if q is not None:
                        starts.append(('from R=%g' % q['R'], q['MF_parameters']))
            jobs.append((p, starts))
        with mp.Pool(n_workers) as pool:
            new = [r for r in pool.map(refine_point, jobs) if r is not None]
        for r in new:
            if verbose:
                old = by_key[(r['R'], r['j'])]
                print('[refine sweep %d] R=%g j=%2d: %-20s E=%+.6f  ->  %-20s %-8s E=%+.6f (%s)'
                      % (sweep + 1, r['R'], r['j'], old['pattern'], old['E_total'], r['pattern'],
                         r['direction'], r['E_total'], r['seed']), flush=True)
            by_key[(r['R'], r['j'])] = r
        if not new:
            if verbose:
                print('[refine] converged after %d sweep(s): no point improved.' % (sweep + 1))
            break
    return sorted(by_key.values(), key=lambda p: (p['R'], p['j']))


def checkpoint_path(radii=RADII):
    rtxt = '-'.join('%g' % r for r in radii)
    return os.path.join(CHECKPOINT_DIR, 'circle_K%g_R%s_n%d_Gamma%g_Nx%dNy%d_BCx%gy%g%s.pkl'
                        % (K, rtxt, N_ANGLES, GAMMA, NX, NY, KX_SHIFT, KY_SHIFT, TAG))


def run_scan(radii=RADII, n_workers=N_WORKERS, verbose=True):
    """Solve every circle point (resuming from the checkpoint if present); returns a list of dicts."""
    path = checkpoint_path(radii)
    done = {}
    if os.path.exists(path):
        with open(path, 'rb') as f:
            done = {(p['R'], p['j']): p for p in pickle.load(f)}
        if verbose:
            print('[checkpoint] %d points loaded from %s' % (len(done), path))
    todo = [p for p in circle_points(radii) if (p[0], p[1]) not in done]
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if todo:
        with mp.Pool(n_workers) as pool:
            for res in pool.imap_unordered(solve_point, todo):
                done[(res['R'], res['j'])] = res
                with open(path + '.tmp', 'wb') as f:
                    pickle.dump(list(done.values()), f)
                os.replace(path + '.tmp', path)
                if verbose:
                    print('R=%-5g j=%2d  J_perp=%+8.4f  J_z=%+8.4f  ->  %-20s %-8s  E_tot=%+12.6f'
                          '  |mA|=%.3f |mB|=%.3f  res=%.1e'
                          % (res['R'], res['j'], res['J_perp'], res['Jz'], res['pattern'],
                             res['direction'], res['E_total'], res['|m_A|'], res['|m_B|'],
                             res['residual']), flush=True)
    return sorted(done.values(), key=lambda p: (p['R'], p['j']))


def export_text(points, text_path):
    cols = ['R', 'j', 'theta_deg', 'J_perp', 'Jz', 'pattern', 'direction', 'E_total', 'E_per_spin',
            '|m_A|', '|m_B|', 'cos_AB', 'gap', 'residual'] + list(MF.MF_PARAMETER_NAMES)
    with open(text_path, 'w') as f:
        f.write('# circle scan, K=%g, Gamma=%g, %d spins (Nx=Ny=%d), BC shift (%g, %g)\n'
                % (K, GAMMA, 2 * NX * NY, NX, KX_SHIFT, KY_SHIFT))
        f.write('\t'.join(cols) + '\n')
        for p in points:
            row = [repr(p['R']), str(p['j']), repr(np.degrees(p['theta'])), repr(p['J_perp']),
                   repr(p['Jz']), p['pattern'], p['direction'].replace(' ', ''), repr(p['E_total']),
                   repr(p['E_total'] / (2 * NX * NY)), repr(p['|m_A|']), repr(p['|m_B|']),
                   repr(p['cos_AB']), repr(p['gap']), repr(p['residual'])]
            row += [repr(float(x)) for x in p['MF_parameters']]
            f.write('\t'.join(row) + '\n')


# Plots
DIR_MARKERS = {'-': 'o', 'm || z': '^', 'm in xy': 's', 'tilted': 'D'}


def _sector_rings(radii):
    """Inner/outer radius of the ring drawn around each scanned radius (first ring starts at 0)."""
    radii = sorted(radii)
    edges = [0.0] + [0.5 * (a + b) for a, b in zip(radii[:-1], radii[1:])] + [radii[-1]]
    return {R: (edges[i], edges[i + 1]) for i, R in enumerate(radii)}


def _style_plane_axes(ax, Rmax):
    lim = 1.12 * Rmax
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.axhline(0, color='black', lw=0.8, zorder=1)
    ax.axvline(0, color='black', lw=0.8, zorder=1)
    ax.set_xlabel(r'$J^z / |K|$')
    ax.set_ylabel(r'$J^\perp / |K|$')
    # pure Kitaev point (J_perp = J_z = 0): small black dot + label
    ax.plot(0, 0, marker='o', ms=5, color='black', zorder=6)
    ax.annotate('pure Kitaev', (0, 0), xytext=(0.06 * Rmax, -0.12 * Rmax), fontsize=8)


def plot_phase_plane(points, save=True, base='circle'):
    """Winning phase in the (J_z, J_perp) plane: coloured wedges; marker shape = moment direction."""
    radii = sorted({p['R'] for p in points})
    rings = _sector_rings(radii)
    dth = np.degrees(2 * np.pi / N_ANGLES)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for p in points:
        r_in, r_out = rings[p['R']]
        # theta is measured from +J_perp towards +J_z -> plot angle 90 deg - theta
        phi = 90.0 - np.degrees(p['theta'])
        ax.add_patch(mpatches.Wedge((0, 0), r_out, phi - dth / 2, phi + dth / 2, width=r_out - r_in,
                                    facecolor=MF._pattern_color(p['pattern']), alpha=0.35,
                                    edgecolor='none', zorder=2))
        ax.scatter(p['Jz'], p['J_perp'], marker=DIR_MARKERS[p['direction']], s=46,
                   color=MF._pattern_color(p['pattern']), edgecolor='black', linewidth=0.6,
                   zorder=4)
    for R in radii:
        ax.add_patch(plt.Circle((0, 0), R, fill=False, ls=':', lw=0.8, color='gray', zorder=3))
    _style_plane_axes(ax, max(radii))
    patterns = [p['pattern'] for p in points]
    handles = [mpatches.Patch(facecolor=MF._pattern_color(q), edgecolor='black', lw=0.5, label=q)
               for q in MF.PATTERN_COLORS if q in patterns]
    dirs = [d for d in DIR_MARKERS if d in {p['direction'] for p in points}]
    handles += [Line2D([], [], marker=DIR_MARKERS[d], ls='', color='gray', mec='black',
                       label=('no order' if d == '-' else d)) for d in dirs]
    ax.legend(handles=handles, fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    ax.set_title(r'Winning phase, $J^\perp = R\cos(\pi j/20)$, $J^z = R\sin(\pi j/20)$'
                 '\n' r'R = %s|K|, $\Gamma$=%g, %d spins' % (', '.join('%g' % r for r in radii),
                                                             GAMMA, 2 * NX * NY), fontsize=10)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, '%s%s_phase%s.png' % (PREFIX, base, TAG))
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig, out


def plot_energy_plane(points, save=True, base='circle'):
    """E(J_z, J_perp) of the 32-spin cluster as colour in the (J_z, J_perp) plane."""
    radii = sorted({p['R'] for p in points})
    rings = _sector_rings(radii)
    dth = np.degrees(2 * np.pi / N_ANGLES)
    E = np.array([p['E_total'] for p in points])
    norm = matplotlib.colors.Normalize(E.min(), E.max())
    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    for p in points:
        r_in, r_out = rings[p['R']]
        phi = 90.0 - np.degrees(p['theta'])
        ax.add_patch(mpatches.Wedge((0, 0), r_out, phi - dth / 2, phi + dth / 2, width=r_out - r_in,
                                    facecolor=cmap(norm(p['E_total'])), edgecolor='white', lw=0.3,
                                    zorder=2))
    _style_plane_axes(ax, max(radii))
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, shrink=0.8, label=r'$E(J^z, J^\perp)$, total MF energy (%d spins)'
                 % (2 * NX * NY))
    ax.set_title(r'$E(J^z, J^\perp)$ on the circle(s) R = %s|K|' % ', '.join('%g' % r for r in radii),
                 fontsize=10)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, '%s%s_energy_plane%s.png' % (PREFIX, base, TAG))
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig, out


if __name__ == '__main__':
    MF.system_parameters.kx_shift, MF.system_parameters.ky_shift = KX_SHIFT, KY_SHIFT
    points = run_scan()
    points = refine_scan(points)
    with open(checkpoint_path(), 'wb') as f:
        pickle.dump(points, f)
    text_path = os.path.splitext(checkpoint_path())[0] + '.txt'
    export_text(points, text_path)
    print('wrote', text_path)

    n_bad = sum(1 for p in points if p['unconverged'])
    if n_bad:
        print('WARNING: %d point(s) did not converge (residual > 1e-6) -- see the text table.' % n_bad)

    base = 'circle_R' + '-'.join('%g' % r for r in RADII)
    for fn in (plot_phase_plane, plot_energy_plane):
        fig, out = fn(points, base=base)
        print('saved', out)
    plt.show()
