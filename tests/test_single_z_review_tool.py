from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.single_z_review.review import (
    ChannelSpec,
    ReviewConfig,
    analyze_stacks,
    main,
    run_review,
)


class _FakeBio:
    def __init__(self, channels: dict[int, np.ndarray]) -> None:
        self.channels = channels
        self.requested_channels: list[int] = []

    def get_image_data(self, order: str, *, T: int, C: int) -> np.ndarray:
        assert order == "ZYX"
        assert T == 0
        self.requested_channels.append(C)
        return self.channels[C]


@dataclass
class _FakeImage:
    path: Path
    bio: _FakeBio
    channel_names: list[str]
    voxel_xy_nm: float = 65.0
    voxel_z_nm: float = 210.0

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)

    @property
    def n_z(self) -> int:
        return int(next(iter(self.bio.channels.values())).shape[0])


def _textured_stack(*, focus_z0: int, offset: int = 0, n_z: int = 5) -> np.ndarray:
    yy, xx = np.mgrid[:24, :24]
    texture = ((yy + xx) % 2).astype(np.float32) * 700.0
    stack = np.empty((n_z, 24, 24), dtype=np.float32)
    for z0 in range(n_z):
        amplitude = max(0.0, 1.0 - 0.35 * abs(z0 - focus_z0))
        stack[z0] = 300.0 + float(offset + 10 * z0) + amplitude * texture
    return stack.astype(np.uint16)


def _dapi_only_config() -> ReviewConfig:
    return ReviewConfig(
        channels=(
            ChannelSpec(
                role="dapi",
                index=2,
                label="DAPI-405",
                display_low=250.0,
                display_high=1800.0,
            ),
        ),
        focus_roles=("dapi",),
        candidate_count=3,
        central_fraction=1.0,
        nuclear_mask=False,
        tile_size=48,
        contact_sheet_columns=3,
    )


def _write_inventory(path: Path, source: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_vsi", "condition", "set_id", "n_z"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_vsi": str(source),
                "condition": "WT",
                "set_id": "WT_1",
                "n_z": "5",
            }
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_analyze_stacks_nominates_ranked_planes_but_does_not_select_one() -> None:
    stack = _textured_stack(focus_z0=2)

    result = analyze_stacks(
        {"dapi": stack},
        _dapi_only_config(),
    )

    assert result.candidate_z[0] == 3
    assert len(result.candidate_z) == 3
    assert not hasattr(result, "selected_z")
    assert {row["z"] for row in result.diagnostics} == {1, 2, 3, 4, 5}
    assert all(row["role"] == "dapi" for row in result.diagnostics)


def test_nuclear_mask_reference_is_one_dapi_plane_not_a_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fishsuite.core import io as fish_io

    stack = _textured_stack(focus_z0=2)
    captured_references: list[np.ndarray] = []
    real_derive = fish_io._derive_nuclear_mask

    def capture_reference(reference: np.ndarray):
        captured_references.append(np.asarray(reference).copy())
        return real_derive(reference)

    monkeypatch.setattr(fish_io, "_derive_nuclear_mask", capture_reference)
    config = ReviewConfig(
        channels=(ChannelSpec("dapi", 2, "DAPI-405", 250, 1800),),
        focus_roles=("dapi",),
        candidate_count=3,
        central_fraction=1.0,
        nuclear_mask=True,
        tile_size=48,
        contact_sheet_columns=3,
    )

    result = analyze_stacks({"dapi": stack}, config)

    assert result.nuclear_mask_reference_z == 3
    assert len(captured_references) == 1
    np.testing.assert_array_equal(captured_references[0], stack[2])
    assert not np.array_equal(captured_references[0], stack.max(axis=0))


def test_dapi_only_run_writes_pending_review_outputs_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw" / "WT_1" / "field01.vsi"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic reader placeholder")
    inventory = tmp_path / "inventory.csv"
    _write_inventory(inventory, source)
    dapi = _textured_stack(focus_z0=2)
    fake = _FakeImage(
        path=source,
        bio=_FakeBio({2: dapi}),
        channel_names=["640 CSU", "561 CSU", "405 CSU"],
    )

    summary = run_review(
        inventory,
        tmp_path / "review",
        _dapi_only_config(),
        reader=lambda _: fake,
    )

    assert summary.images_succeeded == 1
    assert summary.images_failed == 0
    assert fake.bio.requested_channels == [2]
    manifest = _read_csv(summary.manifest_csv)
    assert manifest[0]["condition"] == "WT"
    assert manifest[0]["set_id"] == "WT_1"
    assert manifest[0]["processing_status"] == "READY_FOR_MANUAL_REVIEW"
    assert manifest[0]["candidate_z_1"] == "3"
    status = _read_csv(summary.selection_template_csv)
    assert status[0]["review_status"] == "PENDING_MANUAL_REVIEW"
    assert status[0]["selected_z"] == ""
    assert status[0]["candidate_z_1"] == "3"
    diagnostics = _read_csv(summary.z_diagnostics_csv)
    assert len(diagnostics) == 5
    assert [int(row["z"]) for row in diagnostics] == [1, 2, 3, 4, 5]
    contact_sheet = Path(manifest[0]["contact_sheet"])
    assert contact_sheet.is_file()
    with Image.open(contact_sheet) as image:
        assert image.width > 0
        assert image.height > 0


def test_three_channel_run_reads_each_configured_channel_and_never_projects_z(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw" / "KO_1" / "field02.vsi"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic reader placeholder")
    inventory = tmp_path / "inventory.csv"
    _write_inventory(inventory, source)
    stacks = {
        0: _textured_stack(focus_z0=2, offset=100),
        1: _textured_stack(focus_z0=2, offset=200),
        2: _textured_stack(focus_z0=2, offset=300),
    }
    fake = _FakeImage(
        path=source,
        bio=_FakeBio(stacks),
        channel_names=["640 CSU", "561 CSU", "405 CSU"],
    )
    config = ReviewConfig(
        channels=(
            ChannelSpec("bin1_intron", 0, "BIN1 intron-640", 250, 2400),
            ChannelSpec("partner", 1, "Partner-561", 250, 2400),
            ChannelSpec("dapi", 2, "DAPI-405", 250, 2400),
        ),
        focus_roles=("dapi", "bin1_intron", "partner"),
        candidate_count=2,
        central_fraction=1.0,
        nuclear_mask=False,
        tile_size=40,
        contact_sheet_columns=2,
    )

    summary = run_review(
        inventory,
        tmp_path / "review",
        config,
        reader=lambda _: fake,
    )

    assert fake.bio.requested_channels == [0, 1, 2]
    diagnostics = _read_csv(summary.z_diagnostics_csv)
    assert len(diagnostics) == 15
    assert {row["role"] for row in diagnostics} == {
        "bin1_intron",
        "partner",
        "dapi",
    }
    dapi_means = [
        float(row["slice_mean"])
        for row in diagnostics
        if row["role"] == "dapi"
    ]
    expected_means = [float(plane.mean()) for plane in stacks[2]]
    assert dapi_means == pytest.approx(expected_means)
    assert len(set(dapi_means)) > 1


def test_rerun_preserves_a_completed_manual_selection(tmp_path: Path) -> None:
    source = tmp_path / "raw" / "WT_1" / "field03.vsi"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic reader placeholder")
    inventory = tmp_path / "inventory.csv"
    _write_inventory(inventory, source)
    fake = _FakeImage(
        path=source,
        bio=_FakeBio({2: _textured_stack(focus_z0=2)}),
        channel_names=["640 CSU", "561 CSU", "405 CSU"],
    )
    output = tmp_path / "review"
    first = run_review(inventory, output, _dapi_only_config(), reader=lambda _: fake)
    rows = _read_csv(first.selection_template_csv)
    rows[0].update(
        {
            "review_status": "REVIEWED",
            "selected_z": "4",
            "reviewer": "BA",
            "review_notes": "manual choice",
        }
    )
    with first.selection_template_csv.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    second = run_review(inventory, output, _dapi_only_config(), reader=lambda _: fake)

    preserved = _read_csv(second.selection_template_csv)[0]
    assert preserved["review_status"] == "REVIEWED"
    assert preserved["selected_z"] == "4"
    assert preserved["reviewer"] == "BA"
    assert preserved["review_notes"] == "manual choice"


def test_output_inside_common_raw_source_tree_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "raw" / "WT_1" / "field04.vsi"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic reader placeholder")
    inventory = tmp_path / "inventory.csv"
    _write_inventory(inventory, source)

    with pytest.raises(ValueError, match="raw source tree"):
        run_review(
            inventory,
            tmp_path / "raw" / "review_outputs",
            _dapi_only_config(),
            reader=lambda _: pytest.fail("reader must not be called"),
        )


def test_cli_builds_dapi_only_review_from_repeated_channel_options(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw" / "WT_1" / "field05.vsi"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic reader placeholder")
    inventory = tmp_path / "inventory.csv"
    _write_inventory(inventory, source)
    fake = _FakeImage(
        path=source,
        bio=_FakeBio({2: _textured_stack(focus_z0=2)}),
        channel_names=["640 CSU", "561 CSU", "405 CSU"],
    )
    output = tmp_path / "review"

    exit_code = main(
        [
            str(inventory),
            str(output),
            "--channel",
            "dapi=2",
            "--label",
            "dapi=DAPI-405",
            "--window",
            "dapi=250,1800",
            "--focus-role",
            "dapi",
            "--candidate-count",
            "2",
            "--central-fraction",
            "1.0",
            "--no-nuclear-mask",
            "--tile-size",
            "32",
            "--columns",
            "2",
        ],
        reader=lambda _: fake,
    )

    assert exit_code == 0
    config = (output / "review_config.json").read_text(encoding="utf-8")
    assert '\"role\": \"dapi\"' in config
    assert '\"index\": 2' in config
    assert '\"display_high\": 1800.0' in config
    assert _read_csv(output / "selected_z_review.csv")[0]["selected_z"] == ""
