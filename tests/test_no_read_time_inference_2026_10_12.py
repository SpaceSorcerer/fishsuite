"""Statistical tests and significance decisions are computed only by declared build-time functions (2026-10-12).

Any other call to welch / holm / multipletests / ttest_ind, or any assignment of a ``significant*`` value from a
comparison, anywhere in src/ or scripts/, fails this test. Presentation and reader paths therefore cannot
recompute inference from persisted values; they read stored results or refuse.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALLS = {'welch', 'holm', 'multipletests', 'ttest_ind'}
COMPARE_METHODS = {'lt', 'le', 'gt', 'ge'}

# file -> enclosing functions allowed to compute tests or significance; each is a build-time computation from
# measurements of the current run, or a validator that only asserts stored == recomputed and publishes nothing.
BUILD_TIME = {
    'src/fishsuite/report/stats.py': {'welch'},
    'src/fishsuite/report/aggregate.py': {'build_contrasts'},
    'src/fishsuite/report/sensitivities.py': {'scalar_sensitivities', 'outlier_sensitivity', 'add_headline_holm'},
    'src/fishsuite/report/a20_delivery.py': {'amend_rnase'},  # builds the A20 families from gated inputs
    'src/fishsuite/report/coloc_existing.py': {'simple_statistics'},
    'src/fishsuite/report/miat_qki.py': {'analyze', 'scalar_contrast', 'historical_registry'},  # last: assert-only
    'src/fishsuite/report/ratio_figures.py': {'scalar_inputs'},
    'src/fishsuite/report/dapi_delivery.py': {'_welch_power_monte_carlo', 'validate_report'},  # validators
    'src/fishsuite/core/native_by_condition.py': {'_load'},
    'src/fishsuite/core/_superplot.py': {'_fallback_superplot'},
    'src/fishsuite/core/_vendor/analysis/single_condition_plots.py': {'_superplot_stats'},
    'src/fishsuite/core/modes/if_intensity.py': {'_welch'},
    'src/fishsuite/core/pixel_pattern.py': {'_welch_table'},
    'src/fishsuite/core/singlecell.py': {'compute_singlecell'},
    'src/fishsuite/figures/stats.py': {'run_tests'},
    'scripts/coloc_standard_panel.py': {'welch', 'contrasts_table'},
}


def _hits(path, rel):
    tree = ast.parse(path.read_text(encoding='utf-8', errors='replace'))
    hits = []

    def walk(node, func):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = node.name if func is None else func  # outermost function owns nested helpers
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ''
            if name in CALLS:
                hits.append((rel, func or '<module>', name, node.lineno))
        if isinstance(node, ast.Assign) and any('significant' in ast.unparse(t) for t in node.targets):
            decides = any(isinstance(n, ast.Compare) or (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                                                         and n.func.attr in COMPARE_METHODS)
                          for n in ast.walk(node.value))
            if decides:
                hits.append((rel, func or '<module>', 'significance flag', node.lineno))
        for child in ast.iter_child_nodes(node):
            walk(child, func)

    walk(tree, None)
    return hits


def _all_hits():
    for base in ('src', 'scripts'):
        for path in sorted((ROOT / base).rglob('*.py')):
            if '__pycache__' not in path.parts:
                yield from _hits(path, path.relative_to(ROOT).as_posix())


def test_inference_is_computed_only_in_declared_build_time_functions():
    bad = [h for h in _all_hits() if h[1] not in BUILD_TIME.get(h[0], set())]
    assert not bad, 'statistics computed outside build-time functions:\n' + '\n'.join(
        f'{f}:{line} {func} -> {what}' for f, func, what, line in bad)


def test_scanner_detects_calls_and_flags(tmp_path):
    probe = tmp_path / 'probe.py'
    probe.write_text("def render(c):\n    r = welch(a, b)\n    c['significant_holm_0p05'] = c.p.lt(.05)\n"
                     "    c['significant_raw'] = c.p < .05\n    q = holm([1])\n", encoding='utf-8')
    found = {(func, what) for _, func, what, _ in _hits(probe, 'probe.py')}
    assert found == {('render', 'welch'), ('render', 'significance flag'), ('render', 'holm')}


def test_presentation_readers_named_in_the_review_compute_nothing():
    hits = list(_all_hits())
    assert not [h for h in hits if h[0] == 'src/fishsuite/report/figure_workbook.py']
    assert not [h for h in hits if h[0] == 'src/fishsuite/report/final_delivery.py']
