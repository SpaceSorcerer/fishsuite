"""Capture pre-change ortho line-profile golden (fishsuite ce546fa, unmodified ortho_profile.py)."""
import random, subprocess, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
np.random.seed(0); random.seed(0)
import fishsuite.core.ortho_profile as op
assert 'fishsuite_wt_ce546fa' in op.__file__, op.__file__
PX = 0.065
def synthetic():
    rng = np.random.default_rng(20261020)
    z, y, x = np.indices((12, 96, 96))
    blob = np.exp(-((z-6)**2/4 + (y-48)**2/30 + (x-48)**2/30))
    noise = rng.normal(0, 1, (3, 12, 96, 96))
    return np.stack([blob*100, blob*70, blob*20]) + noise*np.array([5., 4., 2.])[:, None, None, None]
stack = synthetic(); mask = np.zeros((96, 96), bool); mask[20:76, 20:76] = True
out = {}
for name, ends in (('horizontal', ((48., 30.), (48., 66.))), ('oblique', ((35.2, 31.7), (60.9, 64.3)))):
    d, v = op.line_profile(stack[:, 6], *ends, width_px=3, pixel_size_um=PX)
    out[f'{name}_distance'] = d; out[f'{name}_values'] = v
    fig = op.render_ortho_figure(stack, (6, 48, 48), 30, nucleus_mask=mask, pixel_size_um=PX, z_step_um=.2,
                                 display_levels=((0, 100), (0, 100)), line_endpoints=ends, profile_width_px=3)
    raw = next(a for a in fig.axes if a.get_label() == 'raw')
    tr = {l.get_label(): l for l in raw.lines}; assert {'MIAT', 'QKI'} <= set(tr), list(tr)
    out[f'{name}_raw_x'] = np.asarray(tr['MIAT'].get_xdata()); out[f'{name}_raw_y0'] = np.asarray(tr['MIAT'].get_ydata())
    out[f'{name}_raw_y1'] = np.asarray(tr['QKI'].get_ydata())
    matplotlib.pyplot.close(fig)
commit = subprocess.check_output(['git', '-C', r'E:\Claude\fishsuite_wt_ce546fa', 'rev-parse', 'HEAD'], text=True).strip()
np.savez(sys.argv[1], source_commit=np.array(commit), **out)
print(commit, {k: np.shape(v) for k, v in out.items()})

# Linked-set B2 raw traces + B4 CSV on the fake run of tests/test_linked_ortho_outlines_2026_09_30.py (0.065 um/px).
if len(sys.argv) > 2:
    import tempfile, pathlib, pandas as pd
    sys.path.insert(0, r'E:\Claude\fishsuite_wt_ce546fa\tests')
    from test_linked_ortho_outlines_2026_09_30 import _Run
    from fishsuite.figures import linked_set
    assert 'fishsuite_wt_ce546fa' in linked_set.__file__, linked_set.__file__
    run = _Run(pathlib.Path(tempfile.mkdtemp()))
    linked_set.read_stack = lambda r, k: run.stack
    linked_set.build_linked_set(run, "14", 1, 0.5)
    b2 = next(f for s, (f, _) in run.saver.figs.items() if s.startswith('B2_'))
    tr = {l.get_label(): l for l in next(a for a in b2.axes if a.get_label() == 'raw').lines}
    b4 = pd.read_csv(run.data / 'B4_line_profile_values_field14_nuc1.csv')
    np.savez(sys.argv[2], source_commit=np.array(commit), b2_raw_x=np.asarray(tr['MIAT-640'].get_xdata()),
             b2_raw_y0=np.asarray(tr['MIAT-640'].get_ydata()), b2_raw_y1=np.asarray(tr['QKI-561'].get_ydata()),
             **{f'b4_{c}': b4[c].to_numpy() for c in b4.columns})
    print('linked', b4.shape, list(tr))
