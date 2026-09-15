import pandas as pd

from fishsuite.report import endpoints


def test_rna_spot_count_resolves_the_primary_rna_count_endpoint():
    nuclei = pd.DataFrame({"rna_spot_count": [0, 3]})

    resolved, absent = endpoints.resolve(nuclei, pd.DataFrame())

    count = next(item for item in resolved if item.name == "rna1_spots_per_nucleus")
    assert count.column == "rna_spot_count"
    assert count.name not in absent
    assert nuclei["rna_spot_count"].tolist() == [0, 3]
