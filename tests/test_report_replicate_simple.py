"""Display-only report styles: artists, configuration and unchanged statistics."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from matplotlib.colors import to_rgba
from PIL import Image

from fishsuite.cli import cli
from fishsuite.report import figures as fig
from fishsuite.report.build import build_report, load_groups_file
from test_report_end_to_end import _synthetic_run


@pytest.fixture(scope="module")
def report_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("simple")
    run = _synthetic_run(root)
    result = build_report(run, out_dir=root / "report", make_figures=False,
                          coloc_panel=False,
                          groups=["WT=WT_1,WT_2,WT_3", "QKI-KO=KO_1,KO_2,KO_3"])
    return run, result


def context(run, **kwargs):
    return fig.FigureContext(run, {}, None, ["WT", "QKI-KO"], "WT", .05,
                             {}, {}, **kwargs)


@pytest.mark.parametrize("layer,collections", [("none", 2), ("fov", 4)])
def test_only_wells_are_primary_and_fovs_are_optional(report_data, layer, collections):
    run, r = report_data
    ctx = context(run, plot_style="replicate-simple", technical_layer=layer)
    canvas, ax = plt.subplots()
    foot = fig.draw_plot(ax, ctx, "rna1_spots_per_nucleus", r["well"], r["field"],
                         pd.DataFrame(), r["contrasts"], "Puncta", None)
    points = [a for a in ax.collections if hasattr(a, "get_sizes")]
    assert len(points) == collections
    primary = [a for a in points if str(a.get_gid()).startswith("well:")]
    assert len(primary) == 2
    for artist, group in zip(primary, ctx.group_order):
        expected = r["well"].query("endpoint == 'rna1_spots_per_nucleus' and group == @group")
        assert len(artist.get_offsets()) == 3
        np.testing.assert_allclose(artist.get_offsets()[:, 1], expected["well_mean_of_field_values"])
        np.testing.assert_allclose(artist.get_edgecolors()[0], to_rgba(ctx.colors[group]))
    assert "raw Welch p" in foot and "Holm" in foot and "Hedges g" in foot
    assert "MDE" in foot and "nuclei" in foot and "no effect" not in foot
    assert any("p=" in t.get_text() for t in ax.texts)
    plt.close(canvas)


@pytest.mark.parametrize("args,expected", [([], ("replicate-simple", "fov")),
    (["--plot-style", "superplot", "--technical-layer", "none"], ("superplot", "none"))])
def test_cli_overrides_yaml(tmp_path, monkeypatch, args, expected):
    import fishsuite.report.build as build
    profile = tmp_path / "groups.yaml"
    profile.write_text("groups: {WT: [WT_1]}\nplot_style: replicate-simple\ntechnical_layer: fov\n")
    seen = {}
    def capture(**kwargs):
        seen.update(kwargs)
        raise build._agg.ReportInputError("captured before build")
    monkeypatch.setattr(build, "build_report", capture)
    result = CliRunner().invoke(cli, ["report", "--run", str(tmp_path), "--groups-file", str(profile), *args])
    assert "captured before build" in result.output, result.output
    assert (seen["plot_style"], seen["technical_layer"]) == expected
    assert seen["style"] == "brian"
    assert load_groups_file(profile)["plot_style"] == "replicate-simple"


def test_defaults_and_validation(tmp_path):
    ctx = context(tmp_path)
    assert (ctx.plot_style, ctx.technical_layer) == ("superplot", "none")
    with pytest.raises(ValueError, match="plot_style"):
        context(tmp_path, plot_style="invalid")
    with pytest.raises(ValueError, match="technical_layer"):
        context(tmp_path, technical_layer="nuclei")


def test_png_svg_and_standalone_dispatch(report_data, tmp_path):
    run, r = report_data
    fig.set_style()
    ctx = context(run, plot_style="replicate-simple")
    manifest = []
    fig.superplot_standalone(ctx, "rna1_spots_per_nucleus", "Puncta", "Puncta per nucleus",
        r["well"], r["field"], pd.DataFrame(), r["contrasts"], None,
        tmp_path, "simple", manifest)
    with Image.open(tmp_path / "simple.png") as image:
        assert image.info["dpi"] == pytest.approx((600, 600), abs=.02)
    svg = (tmp_path / "simple.svg").read_text(encoding="utf-8")
    for label in ("<text", "Arial", "Filter:", "Welch", "p=", "#595959", "#d67ae5"):
        assert label in svg
    assert "replicate-simple" in manifest[0]["description"]


def test_statistics_identical_across_style_options(report_data, tmp_path):
    run, original = report_data
    rebuilt = build_report(run, out_dir=tmp_path / "replicate", plot_style="replicate-simple",
        technical_layer="fov", make_figures=False, coloc_panel=False,
        groups=["WT=WT_1,WT_2,WT_3", "QKI-KO=KO_1,KO_2,KO_3"])
    for key in ("well", "field", "contrasts", "sec"):
        pd.testing.assert_frame_equal(original[key], rebuilt[key], check_exact=True)


def test_missing_well_is_not_imputed_and_endpoint_counts_use_fields(report_data):
    run, r = report_data
    wells, fields = r["well"].copy(), r["field"].copy()
    endpoint = "rna1_spots_per_nucleus"
    wells.loc[(wells.endpoint == endpoint) & (wells.well_id == "WT_1"),
              "well_mean_of_field_values"] = np.nan
    fields.loc[fields.endpoint == endpoint, "n_nuclei_nonmissing"] = 1
    canvas, ax = plt.subplots()
    foot = fig.draw_plot(ax, context(run, plot_style="replicate-simple"), endpoint,
                        wells, fields, pd.DataFrame({"irrelevant_nucleus": [1e10]}),
                        r["contrasts"], "Puncta", None, clip_pct=1)
    primary = [a for a in ax.collections if str(a.get_gid()).startswith("well:")]
    assert [len(a.get_offsets()) for a in primary] == [2, 3]
    assert "WT: 2 wells, 9 FOVs, 9 defined nuclei" in foot
    plt.close(canvas)


def test_composite_uses_same_dispatch_and_palette(report_data, tmp_path, monkeypatch):
    run, r = report_data
    ctx = context(run, plot_style="replicate-simple", color_overrides={"WT": "#0072B2"})
    captured = []
    def capture(canvas, *args, **kwargs):
        for ax in canvas.axes:
            primary = [a for a in ax.collections if str(a.get_gid()).startswith("well:")]
            assert [len(a.get_offsets()) for a in primary] == [3, 3]
            np.testing.assert_allclose(primary[0].get_edgecolors()[0], to_rgba("#0072B2"))
        labels = "\n".join(t.get_text() for t in canvas.texts)
        assert "A" in labels and "B" in labels and "Filter:" in labels
        assert "MDE" in labels and "3 wells" in labels
        assert "Dots are nuclei" not in labels
        captured.append(True)
        plt.close(canvas)
        return {}
    monkeypatch.setattr(fig, "save", capture)
    specs = [dict(endpoint=e, ylabel=e) for e in
             ("rna1_spots_per_nucleus", "rna1_nuclear_spots_per_nucleus")]
    fig.composite_main(ctx, specs, [], [], r["well"], r["field"], pd.DataFrame(),
                       r["contrasts"], tmp_path, [])
    assert captured


@pytest.mark.parametrize("key,value", [("plot_style", "bad"), ("technical_layer", "nuclei")])
def test_bad_yaml_is_rejected_before_report(tmp_path, key, value):
    profile = tmp_path / "invalid.yaml"
    profile.write_text(f"groups: {{WT: [WT_1]}}\n{key}: {value}\n")
    with pytest.raises(RuntimeError, match=key):
        load_groups_file(profile)


def test_fraction_percent_display_and_footer(report_data):
    run, r = report_data
    canvas, ax = plt.subplots()
    foot = fig.draw_plot(ax, context(run, plot_style='replicate-simple'),
        'rna1_nuclear_spot_fraction', r['well'], r['field'], pd.DataFrame(),
        r['contrasts'], 'fraction', None)
    assert ax.get_ylabel() == 'BIN1 intron puncta: % nuclear (per nucleus)'
    assert ax.get_ylim() == (0, 100)
    for artist in ax.collections:
        if str(artist.get_gid()).startswith('well:'):
            np.testing.assert_allclose(artist.get_offsets()[:,1], 90)
    assert len(foot.splitlines()) == 3
    assert 'Non-detection' not in foot and str(run) in foot
    plt.close(canvas)


@pytest.mark.parametrize('endpoint', ['rna2_nuclear_spot_fraction',
    'paired_frac_rna1_at_partner_shuffle', 'frac_called_coloc_shuffle_runthr',
    'manders_m1_costes_only', 'paired_frac_partner_at_rna1_minus_shuffle'])
def test_all_fraction_endpoints_use_percent(endpoint):
    assert fig.fraction_scale(endpoint) == 100
