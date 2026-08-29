"""End-to-end contracts for the MIAT--QKI final-publication orchestrator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from PIL import Image

import tools.miat_qki_publication.generate_final_publication as publication_orchestrator
from tools.miat_qki_publication.generate_final_publication import (
    FinalPublicationInputs,
    finalize_visual_qa,
    generate_final_publication,
)


ENDPOINT_VALUES = {
    "n_spots_all": (100.0, 50.0),
    "n_spots_floor": (100.0, 50.0),
    "miat_footprint_mass_all_union_deduplicated": (200.0, 100.0),
    "miat_footprint_mass_floor_union_deduplicated": (200.0, 100.0),
    "association_fraction_among_usable_q95": (0.40, 0.20),
    "association_fraction_among_all_floor_spots_q95": (0.30, 0.15),
    "threshold_positive_spots_per_nucleus_q95": (20.0, 5.0),
    "miat_footprint_mass_q95_positive_union_deduplicated": (40.0, 10.0),
}


def _write_selected_planes(path: Path, keys: list[str]) -> None:
    with h5py.File(path, "w") as handle:
        root = handle.create_group("images")
        for key in keys:
            group = root.create_group(key)
            group.attrs.update(
                {
                    "image_key": key,
                    "selected_z_1based": 9,
                    "selected_z_0based": 8,
                    "miat_channel_index": 0,
                    "qki_channel_index": 1,
                    "dapi_channel_index": 2,
                    "same_plane_all_channels": True,
                    "complete": True,
                }
            )
            base = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
            for role, offset in (("miat", 400), ("qki", 555), ("dapi", 334)):
                group.create_dataset(role, data=base + offset)
            labels = np.zeros((64, 64), dtype=np.int32)
            labels[10:25, 10:25] = 1
            group.create_dataset("nucleus_labels", data=labels)
            group.create_dataset("nucleolus_labels", data=np.zeros_like(labels))


def _fixture_inputs(tmp_path: Path) -> FinalPublicationInputs:
    source = tmp_path / "retained_sources"
    source.mkdir()
    catalog = source / "CATALOG" / "INPUT"
    catalog.mkdir(parents=True)
    nuclei_rows: list[dict[str, object]] = []
    fov_rows: list[dict[str, object]] = []
    set_rows: list[dict[str, object]] = []
    inference_rows: list[dict[str, object]] = []
    images: list[dict[str, object]] = []
    nuclei_qc: list[dict[str, object]] = []
    spots: list[dict[str, object]] = []
    pixels: list[dict[str, object]] = []
    catalog_ledger_rows: list[dict[str, object]] = []
    for arm_index, arm in enumerate(("NT", "KD")):
        for replicate in range(1, 7):
            slide = 1 if replicate <= 3 else 2
            biological_set = f"S{slide}_{arm}_{replicate}"
            key = f"{arm.lower()}_set_{replicate}.vsi"
            catalog_file = catalog / biological_set / key
            catalog_file.parent.mkdir(exist_ok=True)
            catalog_file.write_bytes(b"fixture-vsi-record")
            catalog_ledger_rows.append(
                {
                    "source": f"fixture-raw/{biological_set}/{key}",
                    "destination": str(catalog_file),
                    "status": "copied",
                    "bytes": catalog_file.stat().st_size,
                    "sha256": hashlib.sha256(catalog_file.read_bytes()).hexdigest(),
                }
            )
            image_row = {
                "image_key": key,
                "image": key,
                "slide": slide,
                "arm": arm,
                "biological_set": biological_set,
                "is_control": False,
                "secondary_only": False,
                "selected_z_1based": 9,
                "voxel_xy_nm": 1000.0,
                "miat_channel_index": 0,
                "qki_channel_index": 1,
                "dapi_channel_index": 2,
                "plane_lock_pass": True,
                "mask_qc_pass": True,
                "footprint_parity_pass": True,
                "population_reconciliation_pass": True,
                "image_qc_status": "pass",
                "analyzed_vsi_path": str(catalog_file),
                "source_checksum": hashlib.sha256(catalog_file.read_bytes()).hexdigest(),
            }
            images.append(image_row)
            nuclei_qc.append(
                {
                    "image_key": key,
                    "nucleus_id": 1,
                    "nucleus_uid": f"{key}:nucleus:1",
                    "nucleus_qc_status": "pass",
                }
            )
            nucleus_row: dict[str, object] = {
                "sampled_in_analysis": True,
                "include": True,
                "is_control": False,
                "secondary_only": False,
                "arm": arm,
                "slide": slide,
                "replicate": replicate,
                "biological_set": biological_set,
                "image": key,
                "image_key": key,
                "fov": 1,
                "nucleus_id": 1,
            }
            for endpoint, values in ENDPOINT_VALUES.items():
                nucleus_row[endpoint] = values[arm_index]
                shared = {
                    "cohort": "sampled_primary",
                    "endpoint": endpoint,
                    "slide": slide,
                    "arm": arm,
                    "replicate": replicate,
                    "biological_set": biological_set,
                    "n_nuclei_total": 1,
                    "n_nuclei_finite": 1,
                    "value": values[arm_index],
                }
                fov_rows.append(
                    {
                        **shared,
                        "image": key,
                        "image_key": key,
                        "fov": 1,
                        "is_control": False,
                    }
                )
                set_rows.append(
                    {
                        **shared,
                        "n_fovs_total": 1,
                        "n_fovs_finite": 1,
                        "complete_for_inference": True,
                    }
                )
            nuclei_rows.append(nucleus_row)
            for spot_number in range(4):
                spot_uid = f"{key}:spot:{spot_number}"
                spots.append(
                    {
                        "image_key": key,
                        "nucleus_id": 1,
                        "spot_uid": spot_uid,
                        "null_usable": spot_number < 3,
                        "qki_threshold_positive_q95": spot_number == 0,
                        "stored_in_nucleolus": False,
                        "qki_footprint_enrichment_vs_nucleoplasm": float(replicate),
                        "center_y_px": 15 + spot_number,
                        "center_x_px": 15 + spot_number,
                        "footprint_method": "half_max_component",
                    }
                )
                pixels.append(
                    {
                        "spot_uid": spot_uid,
                        "pixel_index": 0,
                        "y_px": 15 + spot_number,
                        "x_px": 15 + spot_number,
                    }
                )
    control_key = "dual_omission_control.vsi"
    control_file = catalog / "FULL_OMISSION" / control_key
    control_file.parent.mkdir(exist_ok=True)
    control_file.write_bytes(b"fixture-vsi-record")
    catalog_ledger_rows.append(
        {
            "source": f"fixture-raw/FULL_OMISSION/{control_key}",
            "destination": str(control_file),
            "status": "copied",
            "bytes": control_file.stat().st_size,
            "sha256": hashlib.sha256(control_file.read_bytes()).hexdigest(),
        }
    )
    ledger_only_payload = catalog / "S1_NT_1" / "fixture_payload.ets"
    ledger_only_payload.write_bytes(b"fixture-catalog-payload")
    catalog_ledger_rows.append(
        {
            "source": "fixture-raw/S1_NT_1/fixture_payload.ets",
            "destination": str(ledger_only_payload),
            "status": "copied",
            "bytes": ledger_only_payload.stat().st_size,
            "sha256": hashlib.sha256(ledger_only_payload.read_bytes()).hexdigest(),
        }
    )
    images.append(
        {
            **images[0],
            "image_key": control_key,
            "image": control_key,
            "arm": "OMISSION",
            "biological_set": "",
            "is_control": True,
            "secondary_only": True,
            "analyzed_vsi_path": str(control_file),
            "source_checksum": hashlib.sha256(control_file.read_bytes()).hexdigest(),
        }
    )
    nuclei_qc.append(
        {
            "image_key": control_key,
            "nucleus_id": 1,
            "nucleus_uid": f"{control_key}:nucleus:1",
            "nucleus_qc_status": "pass",
        }
    )
    for endpoint, values in ENDPOINT_VALUES.items():
        inference_rows.append(
            {
                "cohort": "sampled_primary",
                "endpoint": endpoint,
                "analysis_role": "primary",
                "excluded_biological_set": np.nan,
                "inference_status": "complete",
                "n_nt": 6,
                "n_kd": 6,
                "mean_nt": values[0],
                "mean_kd": values[1],
                "difference_kd_minus_nt": values[1] - values[0],
                "percent_change_kd_vs_nt": -50.0,
                "welch_p_two_sided": 0.03,
                "slide_adjusted_p_two_sided": 0.04,
                "permutation_test": "exact_within_slide_label_permutation",
                "permutation_n_permutations": 400,
                "permutation_p_exact_two_sided": 0.0125,
            }
        )
    ratios = pd.DataFrame(
        [
            {
                "cohort": "sampled_primary",
                "numerator_endpoint": numerator,
                "denominator_endpoint": denominator,
                "n_nt": 6,
                "n_kd": 6,
                "ratio_of_ratios": 0.5,
                "ratio_of_ratios_ci95_low": 0.3,
                "ratio_of_ratios_ci95_high": 0.8,
                "p_two_sided": 0.009,
                "ci_method": "multivariate_delta_within_arm_covariance",
                "pairing_semantics": "none_arm_mean_ratio_of_ratios",
            }
            for numerator, denominator in (
                ("threshold_positive_spots_per_nucleus_q95", "n_spots_floor"),
                (
                    "miat_footprint_mass_q95_positive_union_deduplicated",
                    "miat_footprint_mass_floor_union_deduplicated",
                ),
            )
        ]
    )
    table_paths: dict[str, Path] = {}
    for name, frame in (
        ("nucleus_endpoints", pd.DataFrame(nuclei_rows)),
        ("fov_endpoint_means", pd.DataFrame(fov_rows)),
        ("biological_set_endpoint_means", pd.DataFrame(set_rows)),
        ("endpoint_inference", pd.DataFrame(inference_rows)),
        ("ratio_of_ratios", ratios),
        ("image_audit", pd.DataFrame(images)),
        ("nucleus_qc", pd.DataFrame(nuclei_qc)),
    ):
        path = source / f"{name}.csv"
        frame.to_csv(path, index=False)
        table_paths[name] = path
    for name, frame in (
        ("spot_calls", pd.DataFrame(spots)),
        ("footprint_pixels", pd.DataFrame(pixels)),
    ):
        path = source / f"{name}.csv.gz"
        frame.to_csv(path, index=False, compression="gzip")
        table_paths[name] = path
    selected_planes = source / "selected_planes_and_masks.h5"
    _write_selected_planes(selected_planes, [row["image_key"] for row in images])
    postrun_manifest = source / "postrun_manifest.json"
    postrun_manifest.write_text(
        json.dumps(
            {
                "analysis": "MIAT_QKI_exact_footprint_postrun_statistics",
                "pairing_semantics": "none; NT and KD biological sets are independent",
                "quantitation_plane": "one reviewed z plane per FOV; same DAPI/MIAT/QKI z; no projection",
            }
        ),
        encoding="utf-8",
    )
    backfill_validation = source / "validation_report.json"
    backfill_validation.write_text(
        json.dumps(
            {
                "analysis_scope": "full_manifest",
                "full_parity_gate_pass": True,
                "population_reconciliation_pass": True,
                "run_status": "complete",
                "n_images_complete": 13,
            }
        ),
        encoding="utf-8",
    )
    backfill_parameters = source / "analysis_parameters.json"
    backfill_parameters.write_text(
        json.dumps(
            {
                "analysis_scope": "full_manifest",
                "n_null": 1000,
                "primary_threshold_percentile": 95,
                "placement_geometry": "rotated_center_translated_exact_footprint",
            }
        ),
        encoding="utf-8",
    )
    methods = source / "MICROSCOPY_ACQUISITION_METHODS.md"
    terms = source / "FIGURE_TERMS_PLAIN_LANGUAGE.md"
    methods.write_text("# Microscopy acquisition methods\n\nExact recorded single-z; no projection.\n", encoding="utf-8")
    terms.write_text("# Figure terms\n\nMIAT spot-pixel intensity; q95 association.\n", encoding="utf-8")
    catalog_ledger = source / "CATALOG_SHA256.csv"
    pd.DataFrame(catalog_ledger_rows).to_csv(catalog_ledger, index=False)
    input_kwargs = dict(
        **table_paths,
        selected_planes_h5=selected_planes,
        postrun_manifest=postrun_manifest,
        backfill_validation=backfill_validation,
        backfill_parameters=backfill_parameters,
        catalog_input_dir=catalog,
        microscopy_methods=methods,
        figure_terms=terms,
    )
    if "catalog_sha256_ledger" in FinalPublicationInputs.__dataclass_fields__:
        input_kwargs["catalog_sha256_ledger"] = catalog_ledger
    return FinalPublicationInputs(**input_kwargs)


def test_generation_requires_every_explicit_source_before_output_creation(
    tmp_path: Path,
) -> None:
    """Catches late input failure after a final output directory was created."""

    inputs = _fixture_inputs(tmp_path)
    inputs.backfill_validation.unlink()
    output = tmp_path / "FINAL_PUBLICATION_missing"
    with pytest.raises(FileNotFoundError, match="backfill_validation"):
        generate_final_publication(inputs, output, png_dpi=600)
    assert not output.exists()


def test_generation_refuses_any_existing_output_without_touching_it(
    tmp_path: Path,
) -> None:
    """Catches reuse or cleanup of a prior final-publication directory."""

    inputs = _fixture_inputs(tmp_path)
    output = tmp_path / "FINAL_PUBLICATION_existing"
    output.mkdir()
    sentinel = output / "KEEP.txt"
    sentinel.write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        generate_final_publication(inputs, output, png_dpi=600)
    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"


def test_fixture_build_is_clean_hashed_and_explicitly_six_example(
    tmp_path: Path,
) -> None:
    """Catches incomplete packages, hidden discovery, PDFs, or unhashed copies."""

    inputs = _fixture_inputs(tmp_path)
    output = tmp_path / "FINAL_PUBLICATION_fixture"
    result = generate_final_publication(inputs, output, png_dpi=600)

    assert result.output_dir == output
    assert {path.name for path in output.iterdir() if path.is_dir()} == {
        "STATISTICS",
        "MICROGRAPHS",
        "METHODS",
        "SOURCE_DATA",
        "QA",
    }
    assert not list(output.rglob("*.pdf"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["png_dpi"] == 600
    assert manifest["figure_suffixes"] == [".png", ".svg"]
    assert manifest["pdf_files"] == []
    assert len(manifest["micrograph_examples"]) == 6
    assert {row["condition"] for row in manifest["micrograph_examples"]} == {
        "NT",
        "MIAT-KD",
    }
    copied = manifest["copied_methods"]
    for source_path, destination in (
        (inputs.microscopy_methods, output / "METHODS" / inputs.microscopy_methods.name),
        (inputs.figure_terms, output / "METHODS" / inputs.figure_terms.name),
    ):
        assert destination.read_bytes() == source_path.read_bytes()
        assert copied[destination.name]["source_sha256"] == hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
        assert copied[destination.name]["copied_sha256"] == copied[destination.name][
            "source_sha256"
        ]
    source_manifest = json.loads(
        result.source_hash_manifest_path.read_text(encoding="utf-8")
    )
    assert source_manifest["input_discovery"] == "none; every input was explicit"
    assert set(source_manifest["sources"]) == set(FinalPublicationInputs.__dataclass_fields__)
    assert source_manifest["sources"]["footprint_pixels"]["rows"] == 48
    catalog_integrity = source_manifest["catalog_byte_integrity"]
    assert catalog_integrity["aggregate"]["files_listed"] == 14
    assert catalog_integrity["aggregate"]["files_passed"] == 14
    assert catalog_integrity["aggregate"]["files_failed"] == 0
    assert len(catalog_integrity["files"]) == 14
    for statistical_png in sorted((output / "STATISTICS").glob("*.png")):
        with Image.open(statistical_png) as image:
            assert image.info["dpi"][0] == pytest.approx(600, rel=0.01)
            pixels = np.asarray(image.convert("RGB"))
        border = np.concatenate(
            (
                pixels[:5].reshape(-1, 3),
                pixels[-5:].reshape(-1, 3),
                pixels[:, :5].reshape(-1, 3),
                pixels[:, -5:].reshape(-1, 3),
            )
        )
        assert border.min() >= 250, f"clipped content at canvas edge: {statistical_png}"
    for svg in output.rglob("*.svg"):
        assert "<text" in svg.read_text(encoding="utf-8")
    automated = json.loads(result.qa_json_path.read_text(encoding="utf-8"))
    assert automated["automated_status"] == "pass"
    assert automated["visual_status"] == "pending"
    assert all(item["verdict"] == "pass" for item in automated["file_checks"])
    assert automated["catalog_byte_integrity"]["aggregate"]["files_passed"] == 14
    endpoint_source_columns = set(pd.read_csv(inputs.endpoint_inference).columns)
    assert all(
        set(item["compared_columns"]) == endpoint_source_columns
        and item["comparison_semantics"]
        == "exact strings/integers/booleans; floats within declared CSV-roundtrip tolerance"
        for item in automated["statistic_comparisons"]
    )
    ratio_source_columns = set(pd.read_csv(inputs.ratio_of_ratios).columns)
    assert all(
        set(item["compared_source_columns"]) == ratio_source_columns
        for item in automated["ratio_of_ratios_comparisons"]
    )

    visual_targets = [
        *sorted((output / "STATISTICS").glob("*.png")),
        *sorted((output / "MICROGRAPHS" / "contact_sheets").glob("*.png")),
    ]
    finalize_visual_qa(
        output,
        [
            {
                "path": str(path),
                "verdict": "pass",
                "evidence": "fixture visual-contract inspection",
            }
            for path in visual_targets
        ],
    )
    finalized = json.loads(result.qa_json_path.read_text(encoding="utf-8"))
    assert finalized["visual_status"] == "pass"
    assert not (output / "SUPERSEDED_DO_NOT_USE.md").exists()
    assert not (output / "README_RETAINED_BEFORE_SUPERSESSION.md").exists()
    assert (output / "README.md").read_text(encoding="utf-8").splitlines()[0] == (
        "# MIAT–QKI final publication outputs"
    )
    assert {item["path"] for item in finalized["visual_inspections"]} == {
        str(path.resolve()) for path in visual_targets
    }
    with pytest.raises(RuntimeError, match="already finalized"):
        finalize_visual_qa(output, finalized["visual_inspections"])


def test_generation_rejects_non_600_dpi_before_output_creation(tmp_path: Path) -> None:
    """Catches accidental low-resolution final figures."""

    inputs = _fixture_inputs(tmp_path)
    output = tmp_path / "FINAL_PUBLICATION_low_dpi"
    with pytest.raises(ValueError, match="exactly 600 dpi"):
        generate_final_publication(inputs, output, png_dpi=300)
    assert not output.exists()


@pytest.mark.parametrize("tamper", ("missing", "wrong"))
def test_generation_requires_exact_400_primary_permutations_before_output_creation(
    tmp_path: Path, tamper: str
) -> None:
    """Catches missing or non-exhaustive retained permutation metadata."""

    inputs = _fixture_inputs(tmp_path)
    inference = pd.read_csv(inputs.endpoint_inference)
    if tamper == "missing":
        inference = inference.drop(columns="permutation_n_permutations")
    else:
        inference.loc[
            inference["endpoint"].eq("n_spots_all"),
            "permutation_n_permutations",
        ] = 399
    inference.to_csv(inputs.endpoint_inference, index=False)
    output = tmp_path / f"FINAL_PUBLICATION_permutations_{tamper}"

    with pytest.raises(ValueError, match="permutation_n_permutations|exactly 400"):
        generate_final_publication(inputs, output, png_dpi=600)
    assert not output.exists()


def test_generation_rejects_disk_derived_footprints_before_output_creation(
    tmp_path: Path,
) -> None:
    """Catches a synthetic disk entering any final exact-footprint source table."""

    inputs = _fixture_inputs(tmp_path)
    spots = pd.read_csv(inputs.spot_calls)
    spots.loc[0, "footprint_method"] = "fitted_disk_fallback"
    spots.to_csv(inputs.spot_calls, index=False, compression="gzip")
    output = tmp_path / "FINAL_PUBLICATION_disk_footprint"

    with pytest.raises(ValueError, match="disk-derived footprint"):
        generate_final_publication(inputs, output, png_dpi=600)
    assert not output.exists()


def test_png_dpi_gate_accepts_only_png_quantization_scale_drift(
    tmp_path: Path,
) -> None:
    """Catches a broad approximately-600-DPI gate accepting material drift."""

    exact = tmp_path / "exact.png"
    drifted = tmp_path / "drifted.png"
    Image.new("RGB", (8, 8), "white").save(exact, dpi=(600, 600))
    Image.new("RGB", (8, 8), "white").save(drifted, dpi=(599, 599))

    assert publication_orchestrator._png_checks([exact])[0]["verdict"] == "pass"
    assert publication_orchestrator._png_checks([drifted])[0]["verdict"] == "fail"


def test_complete_retained_qa_rejects_unchecked_ror_string_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a generated retained RoR column escaping field-by-field QA."""

    inputs = _fixture_inputs(tmp_path)
    output = tmp_path / "FINAL_PUBLICATION_ror_mismatch"
    original = publication_orchestrator.render_publication_plot_package

    def tampering_renderer(*args: object, **kwargs: object):
        outputs = original(*args, **kwargs)
        frame = pd.read_csv(outputs.ratio_of_ratios_statistics_path)
        frame.loc[0, "ci_method"] = "tampered_method"
        frame.to_csv(outputs.ratio_of_ratios_statistics_path, index=False)
        return outputs

    monkeypatch.setattr(
        publication_orchestrator, "render_publication_plot_package", tampering_renderer
    )
    with pytest.raises(RuntimeError, match="automated final-publication QA failed"):
        generate_final_publication(inputs, output, png_dpi=600)
    qa = json.loads((output / "QA" / "qa_report.json").read_text(encoding="utf-8"))
    ci_checks = [
        column
        for comparison in qa["ratio_of_ratios_comparisons"]
        for column in comparison["column_comparisons"]
        if column["source_column"] == "ci_method"
    ]
    assert ci_checks and ci_checks[0]["comparison"] == "exact"
    assert ci_checks[0]["verdict"] == "fail"


def test_tiny_csv_float_drift_passes_and_is_recorded_as_nonexact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches truthful tolerance metadata being omitted or mislabeled exact."""

    inputs = _fixture_inputs(tmp_path)
    output = tmp_path / "FINAL_PUBLICATION_float_roundtrip"
    original = publication_orchestrator.render_publication_plot_package

    def drifting_renderer(*args: object, **kwargs: object):
        outputs = original(*args, **kwargs)
        frame = pd.read_csv(outputs.master_statistics_path)
        frame.loc[0, "welch_p_two_sided"] += 5e-14
        frame.to_csv(outputs.master_statistics_path, index=False)
        return outputs

    monkeypatch.setattr(
        publication_orchestrator, "render_publication_plot_package", drifting_renderer
    )
    result = generate_final_publication(inputs, output, png_dpi=600)
    qa = json.loads(result.qa_json_path.read_text(encoding="utf-8"))
    comparison = next(
        item
        for item in qa["statistic_comparisons"]
        if item["endpoint"] == "n_spots_all"
    )
    drift = next(
        item
        for item in comparison["column_comparisons"]
        if item["source_column"] == "welch_p_two_sided"
    )
    assert drift["comparison"] == "float_tolerance"
    assert drift["exactly_equal"] is False
    assert drift["within_tolerance"] is True
    assert comparison["nonzero_float_differences"] >= 1
    assert comparison["verdict"] == "pass_within_tolerance"


def test_catalog_ledger_tamper_fails_closed_before_output_creation(
    tmp_path: Path,
) -> None:
    """Catches trusting ledger strings without hashing every listed destination."""

    inputs = _fixture_inputs(tmp_path)
    ledger = pd.read_csv(inputs.catalog_input_dir.parents[1] / "CATALOG_SHA256.csv")
    payload = Path(
        ledger.loc[
            ledger["destination"].str.endswith("fixture_payload.ets"), "destination"
        ].iloc[0]
    )
    payload.write_bytes(b"tampered-after-ledger")
    output = tmp_path / "FINAL_PUBLICATION_catalog_tamper"
    with pytest.raises(ValueError, match="catalog byte integrity failed"):
        generate_final_publication(inputs, output, png_dpi=600)
    assert not output.exists()


def test_superseded_marker_is_explicit_and_preserves_original_readme(
    tmp_path: Path,
) -> None:
    """Catches a retired production root remaining visually usable or losing context."""

    output = tmp_path / "FINAL_PUBLICATION_old"
    output.mkdir()
    original = "# Original package\n\nRetained scientific context.\n"
    (output / "README.md").write_text(original, encoding="utf-8")
    replacement = tmp_path / "FINAL_PUBLICATION_new"

    marker = publication_orchestrator.mark_publication_superseded(
        output,
        status="replaced_after_provenance_qa_strengthening",
        reason="Complete catalog-byte and retained-value QA replaced this package.",
        replacement_path=replacement,
    )

    assert marker == output / "SUPERSEDED_DO_NOT_USE.md"
    assert marker.is_file()
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert readme.splitlines()[0] == "SUPERSEDED — DO NOT USE"
    assert "replaced_after_provenance_qa_strengthening" in readme
    assert str(replacement.resolve()) in readme
    assert (output / "README_RETAINED_BEFORE_SUPERSESSION.md").read_text(
        encoding="utf-8"
    ) == original
    assert "Complete catalog-byte" in marker.read_text(encoding="utf-8")


@pytest.mark.parametrize("replacement_name", [None, "FINAL_PUBLICATION_replacement"])
def test_visual_fail_finalization_automatically_marks_root_superseded(
    tmp_path: Path, replacement_name: str | None
) -> None:
    """Catches a visual-fail finalizer leaving an apparently usable package."""

    inputs = _fixture_inputs(tmp_path)
    output = tmp_path / "FINAL_PUBLICATION_visual_fail"
    result = generate_final_publication(inputs, output, png_dpi=600)
    original_readme = (output / "README.md").read_text(encoding="utf-8")
    qa = json.loads(result.qa_json_path.read_text(encoding="utf-8"))
    replacement = tmp_path / replacement_name if replacement_name else None
    inspections = [
        {
            "path": path,
            "verdict": "fail" if index == 0 else "pass",
            "evidence": "focused visual-failure fixture",
        }
        for index, path in enumerate(qa["visual_required_paths"])
    ]

    finalize_visual_qa(output, inspections, replacement_path=replacement)

    finalized = json.loads(result.qa_json_path.read_text(encoding="utf-8"))
    assert finalized["visual_status"] == "fail"
    marker = (output / "SUPERSEDED_DO_NOT_USE.md").read_text(encoding="utf-8")
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert readme.splitlines()[0] == "SUPERSEDED — DO NOT USE"
    assert "visual_status=fail" in marker
    assert "visual_status=fail" in readme
    assert (output / "README_RETAINED_BEFORE_SUPERSESSION.md").read_text(
        encoding="utf-8"
    ) == original_readme
    expected_replacement = (
        str(replacement.resolve())
        if replacement is not None
        else "replacement not yet available"
    )
    assert expected_replacement in marker
    assert expected_replacement in readme
