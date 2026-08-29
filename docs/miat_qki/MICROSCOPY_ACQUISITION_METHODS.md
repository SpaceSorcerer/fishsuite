# MIAT-QKI microscopy acquisition methods

MIAT-QKI images from human undifferentiated H9 hESCs in an NT-versus-MIAT-KD comparison were acquired on 2026-08-26 as 44 retained multichannel stacks (18 NT, 19 MIAT-KD, and 7 dual-omission controls) with a Hamamatsu ORCA-Fusion #1 detector and a recorded 100×/NA 1.5 objective. Each stack received an independently algorithmically proposed plane, visually reviewed and accepted, with zero manual overrides. One reviewed recorded optical plane per image was used: DAPI, MIAT, and QKI were taken from that same recorded plane for segmentation, quantification, overlays, and representative images. No maximum-intensity projection (MIP) or other projection was used for QKI association or spatial colocalization.

## Acquisition and provenance

The retained stacks are 2304×2304 uint16 images acquired with 1×1 binning and 0.065 µm XY sampling. They contain 39–54 planes. Z spacing was mixed: 28 stacks at about 0.21 µm z spacing and 16 stacks at about 0.30 µm z spacing. The recorded channel names and exposures were C0 `640 CSU` MIAT-647 at 0.7 s, C1 `561 CSU` QKI-568 at 0.7 s, and C2 `405 CSU` DAPI at 0.5 s. These are recorded channel names, not recovered excitation wavelengths.

For the colocalization workflow, the selected plane was an exact single reviewed z plane, and the same recorded plane was required across DAPI, MIAT, and QKI. A separate MIAT-count sensitivity analysis, a narrow physically matched MIAT projection sensitivity analysis, used seven planes for about 0.21 µm stacks and five planes for about 0.30 µm stacks; it was not used for exact-footprint colocalization, global exact-single-plane endpoints, or representative micrographs.

The MIAT probe and QKI primary antibody were omitted in the dual-omission controls.

| Evidence status | Detail |
| --- | --- |
| Metadata-confirmed | Acquisition date: 2026-08-26; 44 retained stacks: 18 NT, 19 MIAT-KD, and 7 dual-omission controls. |
| Metadata-confirmed | Hamamatsu ORCA-Fusion #1 detector; recorded 100×/NA 1.5 objective; 2304×2304 uint16; 1×1 binning; 0.065 µm XY sampling; 39–54 planes. |
| Metadata-confirmed | Mixed z sampling: 28 stacks at about 0.21 µm z spacing and 16 stacks at about 0.30 µm z spacing. |
| Metadata-confirmed | C0 `640 CSU` MIAT-647 at 0.7 s; C1 `561 CSU` QKI-568 at 0.7 s; C2 `405 CSU` DAPI at 0.5 s. |
| Experimenter-reported | Laser powers were experimenter-reported as held constant; numeric settings and metadata confirmation are unavailable. |
| Missing / not recovered | Microscope stand/model, CSU scan-head model, acquisition software, laser-power values, and several staining and filter details. |

| Acquisition detail | Status | Value / limitation |
| --- | --- | --- |
| Microscope stand/model | Not recovered | No model value recovered |
| CSU scan-head model | Not recovered | No model value recovered |
| Acquisition software/version | Not recovered | No software or version recovered |
| Numeric laser powers | Experimenter-reported only | Held constant (experimenter-reported); no numeric values recovered |

| Analysis boundary | Projection used | Definition |
| --- | --- | --- |
| Exact-footprint colocalization | No | Exact single reviewed recorded plane for DAPI, MIAT, and QKI |
| Global exact-single-plane endpoints | No | Exact single reviewed recorded plane |
| Representative micrographs | No | Exact single reviewed recorded plane |
| MIAT-count sensitivity analysis | Narrow physically matched MIAT projection | Seven planes for about 0.21 µm stacks; five planes for about 0.30 µm stacks |

## Missing information checklist

- Microscope stand/model and CSU scan-head model.
- Acquisition software and version.
- Numeric laser-power settings and any acquisition metadata that would confirm them.
- Staining details not recovered from the audited stack metadata, including remaining probe/antibody, secondary, incubation, and filter details.
