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
    with Image.open(tmp_path / "simple_focus.png") as image:
        assert image.info["dpi"] == pytest.approx((600, 600), abs=.02)
    svg = (tmp_path / "simple_focus.svg").read_text(encoding="utf-8")
    for label in ("<text", "Arial", "Filter:", "Welch", "p=", "#595959", "#d67ae5"):
        assert label in svg
    assert "replicate-simple" in manifest[0]["description"]
    assert manifest[0]["png"] == "simple_focus.png"
    assert manifest[0]["full_png"] == "simple_full.png"
    for variant in ("focus", "full"):
        assert (tmp_path / f"simple_{variant}.png").is_file()
        assert (tmp_path / f"simple_{variant}.svg").is_file()
        assert f"axis: {variant} " in (tmp_path / f"simple_{variant}.svg").read_text(encoding="utf-8")


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
    assert ax.get_ylim()[0] == 50
    assert ax.get_ylim()[1] >= 100
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


@pytest.mark.parametrize("endpoint", ["rna1_nuclear_spot_fraction", "manders_m1_costes_only"])
@pytest.mark.parametrize("values,lower", [([.5, .9], 0), ([.51, .9], 50), ([.1, .2], 0)])
def test_percentage_window_uses_every_well(tmp_path, endpoint, values, lower):
    wells = pd.DataFrame(dict(endpoint=[endpoint]*2, group=["WT", "QKI-KO"],
        well_id=["WT_1", "KO_1"], well_mean_of_field_values=values))
    canvas, ax = plt.subplots()
    foot = fig.draw_plot(ax, context(tmp_path, plot_style="replicate-simple"), endpoint,
        wells, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "fraction", None)
    assert ax.get_ylim() == (lower, 100)
    assert "axis: focus window" in foot
    plt.close(canvas)


@pytest.mark.parametrize("endpoint,well_value,field_value", [
    ("rna1_nuclear_spot_fraction", .99, 1.), ("rna1_spots_per_nucleus", 10., 100.)])
@pytest.mark.parametrize("axis_mode", ["focus", "full"])
def test_bracket_clears_all_drawn_points(report_data, endpoint, well_value, field_value, axis_mode):
    run, r = report_data
    wells, fields = r["well"].copy(), r["field"].copy()
    wells.loc[wells.endpoint == endpoint, "well_mean_of_field_values"] = well_value
    fields.loc[fields.endpoint == endpoint, "field_value"] = field_value
    canvas, ax = plt.subplots(figsize=(2.8, 3.2))
    fig.draw_plot(ax, context(run, plot_style="replicate-simple", technical_layer="fov"),
        endpoint, wells, fields, pd.DataFrame(), r["contrasts"], "Value", None)
    ax._replicate_simple_axis(axis_mode)
    maximum = max(float(a.get_offsets()[:, 1].max()) for a in ax.collections
                  if str(a.get_gid()).startswith(("well:", "fov:")))
    lo, hi = ax.get_ylim()
    bracket = ax.lines[-1].get_ydata()[0]
    assert bracket >= maximum + .13 * (hi - lo)
    assert maximum < bracket < ax.texts[-1].get_position()[1] < hi
    canvas.canvas.draw()
    assert ax.texts[-1].get_window_extent().y1 < ax.get_window_extent().y1
    if endpoint == "rna1_spots_per_nucleus":
        assert 0 <= lo < maximum
    plt.close(canvas)


@pytest.mark.parametrize("groups,width", [(["WT", "QKI-KO"], 2.8),
                                           (["WT", "QKI-KO", "extra"], 3.7)])
def test_compact_standalone_size(report_data, tmp_path, monkeypatch, groups, width):
    run, r = report_data
    ctx = context(run, plot_style="replicate-simple")
    ctx.group_order = groups
    ctx.colors = fig.group_colors(groups)
    def capture(canvas, *args, **kwargs):
        assert canvas.get_figwidth() == pytest.approx(width)
        assert canvas.get_figheight() == pytest.approx(3.2)
        assert canvas.axes[0].get_xlim() == pytest.approx((-.6, len(groups)-.4))
        plt.close(canvas)
        return {}
    monkeypatch.setattr(fig, "save", capture)
    fig.superplot_standalone(ctx, "rna1_spots_per_nucleus", "Puncta", "Puncta per nucleus",
        r["well"], r["field"], pd.DataFrame(), r["contrasts"], None, tmp_path, "size", [])


@pytest.mark.parametrize("title", ["Puncta", "BIN1 intron puncta, % nuclear"])
def test_title_stays_single_line_and_footer_is_six_pt(tmp_path, title):
    canvas, ax = plt.subplots(figsize=(2.8, 3.2))
    fig.layout_replicate_simple(canvas, ax, context(tmp_path), title, "Two-sided Welch; well replicates.")
    canvas.canvas.draw()
    heading = canvas.texts[0]
    assert "\n" not in heading.get_text()
    assert 9 <= heading.get_fontsize() <= 10.5
    assert heading.get_window_extent().width <= canvas.bbox.width * .96
    assert canvas.texts[-1].get_fontsize() == 6
    plt.close(canvas)


@pytest.mark.parametrize("values,focus_low", [([60., 80.], 50.), ([20., 80.], 0.)])
def test_saved_percentage_pair(report_data, tmp_path, values, focus_low):
    run, r = report_data
    endpoint = "rna1_nuclear_spot_fraction"
    wells = r["well"].query("endpoint == @endpoint").copy()
    wells["well_mean_of_field_values"] = np.resize(np.array(values)/100, len(wells))
    canvas, ax = plt.subplots()
    foot = fig.draw_plot(ax, context(run, plot_style="replicate-simple"), endpoint,
        wells, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "fraction", None)
    canvas.text(.01, .01, foot)
    assert ax.get_ylim() == (focus_low, 100)
    records = []
    fig.save(canvas, tmp_path, "percent", records, "", "")
    assert ax.get_ylim() == (focus_low, 100)
    for variant in ("full", "focus"):
        assert (tmp_path / f"percent_{variant}.png").is_file()
        assert (tmp_path / f"percent_{variant}.svg").is_file()


def test_nonnegative_focus_floor_and_full_zero(report_data, tmp_path):
    run, r = report_data
    endpoint = "rna1_spots_per_nucleus"
    wells = r["well"].query("endpoint == @endpoint").copy()
    wells["well_mean_of_field_values"] = np.linspace(71, 79, len(wells))
    canvas, ax = plt.subplots()
    fig.draw_plot(ax, context(run, plot_style="replicate-simple"), endpoint,
        wells, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "count", None)
    assert 0 < ax.get_ylim()[0] < 71
    assert ax.get_ylim()[1] > 79
    ax._replicate_simple_axis("full")
    assert ax.get_ylim()[0] == 0
    assert ax.get_ylim()[1] > 79
    plt.close(canvas)


@pytest.mark.parametrize("variant,floor", [("full", 0), ("focus", 50)])
@pytest.mark.parametrize("values", [[.6, .6, .99], [.6, .99, .99]])
def test_mean_column_sample_sd_and_bracket_clearance(tmp_path, variant, floor, values):
    endpoint = "rna1_nuclear_spot_fraction"
    values = np.array(values)
    wells = pd.DataFrame(dict(endpoint=[endpoint]*4, group=["WT"]*3+["QKI-KO"],
        well_id=["WT_1", "WT_2", "WT_3", "KO_1"],
        well_mean_of_field_values=[*values, .8]))
    contrasts = pd.DataFrame([dict(endpoint=endpoint, test_group="QKI-KO",
                                  reference_group="WT", p_welch=.2)])
    canvas, ax = plt.subplots()
    foot = fig.draw_plot(ax, context(tmp_path, plot_style="replicate-simple"), endpoint,
        wells, pd.DataFrame(), pd.DataFrame(), contrasts, "fraction", None)
    ax._replicate_simple_axis(variant)
    bars = [p for p in ax.patches if p.get_gid() == "group-mean:WT"]
    assert len(bars) == 1
    bar = bars[0]
    mean, sd = np.mean(values*100), np.std(values*100, ddof=1)
    assert bar.get_y() == floor
    assert bar.get_y() + bar.get_height() == pytest.approx(mean)
    assert bar.get_width() == pytest.approx(.6)
    assert bar.get_facecolor() == pytest.approx(to_rgba("#595959", .35))
    assert bar.get_edgecolor() == pytest.approx(to_rgba("#595959"))
    assert bar.get_linewidth() == 1
    errors = [c for c in ax.containers if c.get_label() == "group-sd:WT"]
    assert len(errors) == 1
    _, caps, stems = errors[0].lines
    np.testing.assert_allclose(stems[0].get_segments()[0], [[0, mean-sd], [0, mean+sd]])
    assert len(caps) == 2
    points = next(c for c in ax.collections if c.get_gid() == "well:WT")
    assert len(points.get_offsets()) == 3
    assert bar.get_zorder() < points.get_zorder()
    assert stems[0].get_zorder() < points.get_zorder()
    assert all(c.get_zorder() < points.get_zorder() for c in caps)
    assert not any(p.get_gid() == "group-mean:QKI-KO" for p in ax.patches)
    assert not any(c.get_label() == "group-sd:QKI-KO" for c in ax.containers)
    lo, hi = ax.get_ylim()
    assert ax.lines[-1].get_ydata()[0] >= max(mean+sd, max(values*100)) + .13*(hi-lo)
    assert "bar = mean of well means, error bar = ± SD" in foot
    plt.close(canvas)
