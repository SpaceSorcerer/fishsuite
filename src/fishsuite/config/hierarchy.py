"""Resolve experiment metadata without changing any image measurements."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SECONDARY_ONLY = 'Secondary-only'


def select_inputs(images, input_dir, subset):
    """An explicit relative path selects one FOV, even with repeated basenames."""
    if not subset:
        return list(images)
    wanted = {str(value).replace(chr(92), '/').strip() for value in subset if str(value).strip()}
    bare = {value for value in wanted if '/' not in value}
    return [im for im in images if im.path.name in bare or im.path.as_posix() in wanted
            or im.path.relative_to(input_dir).as_posix() in wanted]


def group_mapping(groups):
    owners = {}
    for group, wells in groups.items():
        for well in wells:
            if well in owners:
                raise ValueError(f'well {well!r} assigned to both {owners[well]!r} and {group!r}')
            owners[str(well)] = str(group)
    return owners


def well_pattern(pattern):
    if not pattern:
        return None
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f'Invalid well_from_image regex: {exc}') from exc
    if rx.groups != 1:
        raise ValueError('well_from_image must have exactly one capture group')
    return rx


def resolve_hierarchy(frame, mapping=None, well_from_image=None, *, strict=False):
    """Well-first assignment; compatible original-label fallback for old reports.

    ``image`` is the unique field identity, not necessarily its basename. Controls
    retain their own well identity but never enter biological groups. Explicit
    grouped preflight/native mode rejects unassigned or absent declared members.
    Legacy reports may keep unlisted source labels visible (strict=False).
    """
    mapping = mapping or {}
    required = {'image', 'condition', 'secondary_only'}
    if not required <= set(frame):
        raise ValueError(f'Missing hierarchy columns: {sorted(required - set(frame))}')
    out = frame[[c for c in ('image', 'condition', 'secondary_only', 'well_id', 'source_condition', 'source_path', 'output_stem')
                 if c in frame]].copy()
    if out.image.isna().any() or out.image.astype(str).str.strip().eq('').any():
        raise ValueError('Empty FOV image identity')
    if out.image.duplicated().any():
        raise ValueError('Duplicate image identities; use unique source-relative FOV paths')
    sec = out.secondary_only.astype(str).str.lower()
    if not sec.isin(['true', 'false', '1', '0', '1.0', '0.0']).all():
        raise ValueError('Invalid secondary_only flags')
    out['secondary_only'] = sec.isin(['true', '1', '1.0'])
    out['source_condition'] = out.get('source_condition', out.condition)
    rx = well_pattern(well_from_image)
    rows = []
    used = set()
    well_owners = {}
    for row in out.to_dict('records'):
        source = str(row['source_condition'])
        match = rx.search(str(row['image']).replace('\\', '/').rsplit('/', 1)[-1]) if rx else None
        found = match.group(1) if match else None
        saved = row.get('well_id')
        saved = str(saved) if pd.notna(saved) and str(saved).strip() else None
        if rx and not found and not row['secondary_only']:
            raise ValueError(f'well_from_image matched no well in biological image {row["image"]!r}')
        if found and saved and found != saved:
            raise ValueError(f'Conflicting saved/filename well for {row["image"]!r}: {saved!r} / {found!r}')
        well = found or saved or source
        if not well.strip() or well.lower() == 'nan':
            if strict or rx:
                raise ValueError(f'Empty biological well/source label for {row["image"]!r}')
            well = '(unassigned)'
        if row['secondary_only']:
            group = SECONDARY_ONLY
        else:
            candidates = {mapping[k] for k in (well, source) if k in mapping}
            if source in mapping.values() and (candidates or not strict):
                candidates.add(source)
            if len(candidates) > 1:
                raise ValueError(f'Conflicting group mappings for well {well!r}, source {source!r}: {sorted(candidates)}')
            if strict and mapping and not candidates:
                raise ValueError(f'Unassigned biological well {well!r} (image {row["image"]!r}); add it to conditions.groups')
            group = next(iter(candidates)) if candidates else source
            used.update(k for k in (well, source) if k in mapping)
            if group == SECONDARY_ONLY:
                raise ValueError('Secondary-only is reserved for controls')
            if well in well_owners and well_owners[well] != group:
                raise ValueError(f'Biological well {well!r} belongs to conflicting groups {well_owners[well]!r} and {group!r}; use globally unique well IDs')
            well_owners[well] = group
        row.update(well_id=well, group=group, field_id=str(row['image']))
        rows.append(row)
    if strict and set(mapping) - used:
        raise ValueError(f'declared wells missing from biological run: {sorted(set(mapping) - used)}')
    return pd.DataFrame(rows, columns=list(out.columns) + [c for c in ('well_id', 'group', 'field_id') if c not in out])


def discovery_roster(images, input_dir, conditions):
    """Preflight roster shared by the GUI and runner; no image pixels are read."""
    names = pd.Series([im.path.name for im in images])
    repeated = set(names[names.duplicated(keep=False)])
    rows = [dict(image=(im.path.relative_to(input_dir).as_posix() if im.path.name in repeated else im.path.name),
                 condition=im.condition, secondary_only=im.sec_only,
                 source_path=str(im.path)) for im in images]
    raw = pd.DataFrame(rows, columns=['image', 'condition', 'secondary_only', 'source_path'])
    result = resolve_hierarchy(raw, group_mapping(conditions.groups), conditions.well_from_image,
                               strict=bool(conditions.groups))
    result['source_path'] = raw.source_path.to_numpy()
    return result


def annotate_frame(frame, roster):
    """Attach the validated identifiers to any exported measurement table."""
    if frame.empty or 'image' not in frame:
        return frame
    columns = ['source_condition', 'well_id', 'group', 'field_id'] + [c for c in ('source_path', 'output_stem') if c in roster]
    out = frame.drop(columns=columns, errors='ignore').merge(
        roster[['image'] + columns], on='image', how='left', validate='many_to_one')
    if out.group.isna().any():
        raise ValueError('Measurement FOV absent from validated experiment hierarchy')
    return out


def set_result_identity(result, image):
    """Disambiguate duplicate basenames consistently across all result carriers."""
    result.image = image
    for name in ('per_image', 'thresholds'):
        getattr(result, name)['image'] = image
    for name in ('nuclei', 'spots', 'morphology'):
        table = getattr(result, name, None)
        if isinstance(table, pd.DataFrame) and 'image' in table:
            table['image'] = image
    for table in getattr(result, 'extra', {}).values():
        if isinstance(table, pd.DataFrame) and 'image' in table:
            table['image'] = image
