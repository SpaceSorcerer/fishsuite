"""Shared locked rendering inputs for ratio-module scalar estimands."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy import stats
from .endpoints import RATIO_ENDPOINTS
from .sensitivities import mixed_model


def context(run, colors):
    return SimpleNamespace(group_order=list(colors), reference=next(iter(colors)),
                           colors=colors, technical_layer='none', plot_style='replicate-simple',
                           run_path=str(run), run_name=Path(run).name, channel_labels={}, footer=lambda x:x)


def scalar_inputs(wells, nuclei, kind, colors):
    """Inputs use group/well_id/image plus T/A; preserve source well weighting."""
    frames, contrasts = [], []
    definitions = [e for e in RATIO_ENDPOINTS if e.name.startswith('ratio_'+kind+'_')]
    reference, test = list(colors)
    for col, definition in zip(('T', 'A'), definitions):
        w = wells.rename(columns={col:'well_mean_of_field_values'}).copy()
        w['endpoint'] = definition.name
        frames.append(w)
        a, b = [w.loc[w.group.eq(g), 'well_mean_of_field_values'].to_numpy(float) for g in (test, reference)]
        row = dict(endpoint=definition.name, test_group=test, reference_group=reference,
                   p_welch=stats.ttest_ind(a, b, equal_var=False).pvalue)
        if len(nuclei) and col in nuclei:
            row.update(mixed_model(nuclei.rename(columns={col:'value'}), test, reference))
        contrasts.append(row)
    return definitions, pd.concat(frames, ignore_index=True), pd.DataFrame(contrasts)
