import logging

import pytest

from fishsuite.config.schema import ConditionsCfg
from fishsuite.core.io import discover_inputs


def _images(root, *names):
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_condition_guardrail_defaults_and_config_values():
    cfg = ConditionsCfg()
    assert cfg.strict_subfolders is False
    assert cfg.strict_filenames is False
    assert cfg.exclude_subfolders == []
    cfg = ConditionsCfg(strict_subfolders=True, strict_filenames=True,
                        exclude_subfolders=["discard"])
    assert cfg.strict_subfolders is True
    assert cfg.strict_filenames is True
    assert cfg.exclude_subfolders == ["discard"]


def test_strict_subfolders_rejects_unmapped_folder(tmp_path):
    _images(tmp_path, "unexpected/a.tif")
    with pytest.raises(ValueError) as exc:
        discover_inputs(tmp_path, subfolder_conditions={"control": "NT", "treated": "KD"},
                        strict_subfolders=True)
    assert "unexpected" in str(exc.value)
    assert "control" in str(exc.value)
    assert "treated" in str(exc.value)


def test_unmapped_subfolder_warns_once_and_keeps_label(tmp_path, caplog):
    _images(tmp_path, "unexpected/a.tif", "unexpected/b.tif", "control/c.tif")
    with caplog.at_level(logging.WARNING):
        images = discover_inputs(tmp_path, subfolder_conditions={"control": "NT"})
    assert [image.condition for image in images] == ["NT", "unexpected", "unexpected"]
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "unexpected" in warnings[0].message


def test_strict_subfolders_allows_no_mapping(tmp_path):
    _images(tmp_path, "control/a.tif")
    images = discover_inputs(tmp_path, strict_subfolders=True)
    assert [image.condition for image in images] == ["control"]


def test_excluded_folder_skipped_before_strict_validation_and_logged_once(tmp_path, caplog):
    _images(tmp_path, "discard/a.tif", "discard/b.tif", "control/c.tif")
    with caplog.at_level(logging.INFO):
        images = discover_inputs(tmp_path, subfolder_conditions={"control": "NT"},
                                 strict_subfolders=True, exclude_subfolders=["discard"])
    assert [image.path.name for image in images] == ["c.tif"]
    notices = [record for record in caplog.records if "discard" in record.message]
    assert len(notices) == 1


def test_exclusion_matches_directory_component_not_substring(tmp_path):
    _images(tmp_path, "discard_extra/a.tif", "control/discard.tif")
    images = discover_inputs(tmp_path, exclude_subfolders=["discard"])
    assert {image.path.name for image in images} == {"a.tif", "discard.tif"}


def test_strict_filenames_rejects_unmatched_filename(tmp_path):
    _images(tmp_path, "unknown.tif")
    with pytest.raises(ValueError) as exc:
        discover_inputs(tmp_path, filename_conditions=[["-NT_", "NT"], ["-KD_", "KD"]],
                        strict_filenames=True)
    assert "unknown.tif" in str(exc.value)
    assert "-nt_" in str(exc.value).lower()
    assert "-kd_" in str(exc.value).lower()


def test_filename_fallback_warns_and_keeps_condition(tmp_path, caplog):
    _images(tmp_path, "unknown.tif")
    with caplog.at_level(logging.WARNING):
        images = discover_inputs(tmp_path, subfolder_conditions={"": "fallback"},
                                 filename_conditions=[["-NT_", "NT"]])
    assert [image.condition for image in images] == ["fallback"]
    assert len(caplog.records) == 1
    assert "unknown.tif" in caplog.records[0].message


def test_filename_resolution_takes_precedence_over_subfolder_strictness(tmp_path):
    _images(tmp_path, "sample-NT_01.tif")
    images = discover_inputs(tmp_path, subfolder_conditions={"elsewhere": "unused"},
                             filename_conditions=[["-nt_", "NT"], ["sample", "other"]],
                             strict_subfolders=True, strict_filenames=True)
    assert [image.condition for image in images] == ["NT"]


def test_strict_filenames_keeps_secondary_only_precedence(tmp_path):
    _images(tmp_path, "sec-only.tif")
    images = discover_inputs(tmp_path, filename_conditions=[["-NT_", "NT"]],
                             sec_only_files=["sec-only"], strict_filenames=True)
    assert images[0].sec_only is True
    assert images[0].condition == "Sec-Only"


def test_strict_filenames_without_patterns_keeps_legacy_behavior(tmp_path):
    _images(tmp_path, "sample.tif")
    images = discover_inputs(tmp_path, subfolder_conditions={"": "NT"},
                             strict_filenames=True)
    assert [image.condition for image in images] == ["NT"]
