"""Report-time DAPI classification, without detector or segmentation calls."""
import numpy as np
import pandas as pd
import pytest
from fishsuite.report.dapi_mask import classify_plane


def fixture():
    y, x = np.mgrid[:100, :160]
    a = (x-35)**2 + (y-50)**2 < 20**2
    b = (x-110)**2 + (y-50)**2 < 20**2
    dapi = (a | b).astype(float)*100
    retained = a.astype(int)
    retained[5,75] = 1  # a retained boundary pixel outside DAPI is extranuclear
    spots = pd.DataFrame(dict(spot_id=[1,2,3], nucleus_id=[1,1,1],
                              x_px=[35,110,75], y_px=[50,50,5]))
    return dapi, retained, spots


def test_two_objects_classes_census_and_fraction():
    from fishsuite.report.figures import fraction_scale
    assert fraction_scale('nuclear_spot_fraction_dapi') == 100.
    dapi, retained, spots = fixture()
    r = classify_plane(dapi, retained, spots, [1], .1)
    assert r['spots']['class'].tolist() == ['in_retained_nucleus',
        'in_unretained_dapi_object', 'extranuclear']
    assert len(r['objects']) == 2
    assert r['objects'].retained_nuclei_count.sum() == 1
    assert r['objects'].unretained.sum() == 1
    assert r['objects'].loc[r['objects'].unretained, 'bin1_spots'].item() == 1
    assert r['nuclei'].extranuclear_spot_count_per_cell_territory.item() == 1
    assert r['nuclei'].nuclear_spot_fraction_dapi.item() == .5
    assert r['spots'].distance_to_nearest_dapi_edge_um.notna().all()


def test_no_dapi_is_na():
    dapi, retained, spots = fixture()
    r = classify_plane(np.zeros_like(dapi), retained, spots, [1], .1)
    assert r['objects'].empty
    assert r['nuclei'].nuclear_spot_fraction_dapi.isna().all()
    assert r['nuclei'].extranuclear_spot_count_per_cell_territory.isna().all()
    assert r['spots'].distance_to_nearest_dapi_edge_um.isna().all()


def test_reject_duplicate_and_out_of_frame():
    dapi, retained, spots = fixture()
    with pytest.raises(ValueError, match='duplicate'):
        classify_plane(dapi, retained, pd.concat([spots, spots.iloc[:1]]), [1], .1)
    spots.loc[0, 'x_px'] = -1
    with pytest.raises(ValueError, match='outside'):
        classify_plane(dapi, retained, spots, [1], .1)
