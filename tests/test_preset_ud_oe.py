from pathlib import Path

import pytest
import yaml

from fishsuite.config.hierarchy import discovery_roster, select_inputs
from fishsuite.config.schema import FishsuiteConfig
from fishsuite.core.io import discover_inputs
from fishsuite.core.nuclear_size import resolve_nuclear_size_px


ROOT = Path(__file__).resolve().parents[1]
PRESET = ROOT / 'src/fishsuite/config/presets/miat_qki_coloc_ud_OEvControl_PLAIN_2026-09-17.yaml'
SOURCE = ROOT / 'src/fishsuite/config/presets/miat_qki_coloc_ud_ALLARMS_PLAIN_strictMIAT_2026-06-05.yaml'
ROSTER = {
    'g2 +Dox (MIAT OE)': ('g2/with Dox', ['UD-MIAT-FISH-QKI-IF-g2-Dox_01.vsi', 'UD-MIAT-FISH-QKI-IF-g2-Dox_02.vsi']),
    'g2 noDox (control)': ('g2/no Dox', ['UD-MIAT-FISH-QKI-IF-g2-no Dox_03.vsi', 'UD-MIAT-FISH-QKI-IF-g2-no Dox_04.vsi']),
    'Nog +Dox': ('Nog/with Dox', ['UD-MIAT-FISH-QKI-IF-Nog-with Dox_10.vsi']),
    'Nog noDox': ('Nog/no Dox', ['UD-MIAT-FISH-QKI-IF-Nog-no Dox_07.vsi', 'UD-MIAT-FISH-QKI-IF-Nog-no Dox_08.vsi']),
    'VPR +Dox': ('VPR/with Dox', ['UD-MIAT-FISH-QKI-IF-VPR-with Dox_11.vsi', 'UD-MIAT-FISH-QKI-IF-VPR-with Dox_12.vsi']),
    'VPR noDox': ('VPR/no Dox', ['UD-MIAT-FISH-QKI-IF-VPR-no Dox_14.vsi', 'UD-MIAT-FISH-QKI-IF-VPR-no Dox_15.vsi']),
}


def discover(root, cfg):
    fields = ('subfolder_conditions', 'strict_subfolders', 'exclude_subfolders',
              'sec_only_folders', 'sec_only_files', 'filename_conditions', 'strict_filenames', 'recursive_discovery')
    found = discover_inputs(root, **{k: getattr(cfg.conditions, k) for k in fields})
    return select_inputs(found, root, cfg.input_file_subset)


def plant(root):
    for _, (folder, names) in ROSTER.items():
        leaf = root / folder
        leaf.mkdir(parents=True, exist_ok=True)
        for name in names:
            (leaf / name).touch()
            (leaf / name.replace('.vsi', '(decon).vsi')).touch()
    (root / 'Nog/with Dox/UD-MIAT-FISH-QKI-IF-Nog-with Dox_09.vsi').touch()
    duplicate = root / 'OneDrive_1_6-3-2026/g2/with Dox'
    duplicate.mkdir(parents=True)
    (duplicate / 'UD-MIAT-FISH-QKI-IF-g2-Dox_01.vsi').touch()
    (duplicate / 'unmapped.vsi').touch()


def test_load_and_physical_nuclear_sizes():
    cfg = FishsuiteConfig.from_yaml(PRESET)
    size = resolve_nuclear_size_px(cfg.nuclei, 0.13, cfg.nuclei.cellpose_downsample_factor)
    assert size.diameter_px == pytest.approx(42.30769230769231)
    assert size.native_diameter_px == pytest.approx(84.61538461538461)
    assert size.min_area_px == pytest.approx(4000.0)
    assert size.reject_ghost_min_area_px == pytest.approx(6000.0)
    assert cfg.nuclei.cellpose_downsample_factor == 2.0
    assert not {'cellpose_diameter_px', 'min_area_px', 'reject_ghost_min_area_px'} & cfg.nuclei.model_fields_set


def test_six_nested_conditions_roster_and_one_field_per_well(tmp_path):
    cfg = FishsuiteConfig.from_yaml(PRESET)
    plant(tmp_path)
    found = discover(tmp_path, cfg)
    expected = {name: label for label, (_, names) in ROSTER.items() for name in names}
    assert len(found) == 11
    assert {im.path.name: im.condition for im in found} == expected
    assert all('OneDrive_1_6-3-2026' not in im.path.parts for im in found)
    assert list(cfg.conditions.condition_order) == list(ROSTER)
    roster = discovery_roster(found, tmp_path, cfg.conditions)
    assert roster.well_id.nunique() == 11
    assert roster.groupby('well_id').size().eq(1).all()
    assert roster.groupby('group').well_id.nunique().to_dict() == {
        'g2 +Dox (MIAT OE)': 2, 'g2 noDox (control)': 2,
        'Nog +Dox': 1, 'Nog noDox': 2, 'VPR +Dox': 2, 'VPR noDox': 2,
    }


def test_stray_filename_raises_before_subset_filter(tmp_path):
    cfg = FishsuiteConfig.from_yaml(PRESET)
    plant(tmp_path)
    (tmp_path / 'junk').mkdir()
    (tmp_path / 'junk/unmapped.vsi').touch()
    with pytest.raises(ValueError, match='unmapped.vsi.*matches no condition pattern'):
        discover(tmp_path, cfg)


def test_colors_and_unchanged_analysis_settings():
    cfg = FishsuiteConfig.from_yaml(PRESET)
    old = FishsuiteConfig.from_yaml(SOURCE)
    assert cfg.conditions.group_colors == {
        'g2 +Dox (MIAT OE)': '#56B4E9', 'g2 noDox (control)': '#7F7F7F',
        'Nog +Dox': '#7F7F7F', 'Nog noDox': '#7F7F7F',
        'VPR +Dox': '#7F7F7F', 'VPR noDox': '#7F7F7F',
    }
    for key in ('channels', 'z_stack', 'foci', 'nucleolus', 'output', 'parallel',
                'cytoplasm', 'spot_coloc', 'pixel_coloc', 'input_file_subset'):
        assert getattr(cfg, key) == getattr(old, key), key
    raw = yaml.safe_load(PRESET.read_text(encoding='utf-8'))
    raw['conditions']['group_colors']['Nog +Dox'] = 'invalid-color'
    with pytest.raises(ValueError, match='invalid group color'):
        FishsuiteConfig.model_validate(raw)


def test_filename_discovery_preserves_nested_relative_folder_and_skips_sidecars(tmp_path):
    leaf = tmp_path / 'g2/with Dox'
    leaf.mkdir(parents=True)
    (leaf / 'sample-g2-Dox_01.vsi').touch()
    hidden = leaf / '_sidecar'
    hidden.mkdir()
    (hidden / 'unmapped.tif').touch()
    found = discover_inputs(tmp_path, filename_conditions=[['-g2-Dox_', 'OE']], strict_filenames=True,
                            recursive_discovery=True)
    assert [(im.path.name, im.condition, im.subfolder) for im in found] == [
        ('sample-g2-Dox_01.vsi', 'OE', 'g2/with Dox')
    ]


@pytest.mark.parametrize('explicit_flag', [False, None])
def test_legacy_discovery_literal_roster_ignores_root_and_nested_files(tmp_path, explicit_flag):
    # Literal roster read from integration/miat-oe-2026-09-all:core/io.py:
    # any non-underscore directory selects the sorted one-level-only branch.
    for relative in ['root-sample.tif', 'A/a-sample.tif', 'A/nested/deep-sample.tif',
                     'B/b-sample.tif', 'B/c-sec.tif', '_hidden/h-sample.tif',
                     'duplicate/d-sample.tif']:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    kwargs = {} if explicit_flag is None else {'recursive_discovery': explicit_flag}
    found = discover_inputs(tmp_path, filename_conditions=[['sample', 'mapped']],
                            sec_only_files=['-sec'], exclude_subfolders=['duplicate'], **kwargs)
    assert [(im.path.relative_to(tmp_path).as_posix(), im.condition, im.subfolder, im.sec_only)
            for im in found] == [
        ('A/a-sample.tif', 'mapped', 'A', False),
        ('B/b-sample.tif', 'mapped', 'B', False),
        ('B/c-sec.tif', 'Sec-Only', 'B', True),
    ]


def test_recursive_discovery_posix_keys_and_pruned_folders(tmp_path):
    for relative in ['root-sample.tif', 'arm/no Dox/a-sample.tif',
                     'arm/with Dox/b-sample.tif', 'one/c-sample.tif',
                     'duplicate/inner/d-sample.tif', 'arm/_hidden/e-sample.tif']:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    found = discover_inputs(tmp_path, recursive_discovery=True,
        filename_conditions=[['sample', 'mapped']], sec_only_folders=['arm/no Dox'],
        exclude_subfolders=['duplicate'], subfolder_conditions={'arm/no Dox': 'nested control'})
    assert [(im.path.relative_to(tmp_path).as_posix(), im.condition, im.subfolder, im.sec_only)
            for im in found] == [
        ('arm/no Dox/a-sample.tif', 'nested control', 'arm/no Dox', True),
        ('arm/with Dox/b-sample.tif', 'mapped', 'arm/with Dox', False),
        ('one/c-sample.tif', 'mapped', 'one', False),
        ('root-sample.tif', 'mapped', '', False),
    ]


def test_only_new_preset_enables_recursive_discovery():
    assert FishsuiteConfig().conditions.recursive_discovery is False
    enabled = []
    for path in sorted(PRESET.parent.glob('*.yaml')):
        raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        if (raw.get('conditions') or {}).get('recursive_discovery', False):
            enabled.append(path.name)
    assert enabled == [PRESET.name]


def test_nested_condition_output_stem_is_flat_and_collision_checked(tmp_path):
    from fishsuite.core.io import DiscoveredImage
    from fishsuite.runner import _stem_with_condition, _validate_output_stems
    stem = _stem_with_condition('field', 'g2/no Dox')
    assert '/' not in stem and '\\' not in stem
    assert Path(stem).name == stem
    images = [DiscoveredImage(tmp_path / 'a/field.tif', 'g2/no Dox', False, 'g2/no Dox'),
              DiscoveredImage(tmp_path / 'b/field.tif', 'g2\\no Dox', False, 'g2/no Dox')]
    with pytest.raises(ValueError, match='(?i)(collision|ambiguous|duplicate)'):
        _validate_output_stems(images)
