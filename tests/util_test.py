"""
Test util functions
"""

from powergenome.util import apply_all_tag_to_regions


def test_apply_all_tag_to_regions():
    settings = {
        "model_regions": ["a", "b", "c"],
        "renewables_clusters": [
            {
                "region": "all",
                "technology": "landbasedwind",
                "bin": {"feature": "lcoe", "q": 4},
            },
            {
                "region": "b",
                "technology": "landbasedwind",
                "filter": {"feature": "lcoe", "max": 50},
            },
            {"region": "all", "technology": "utilitypv", "group": ["state"]},
        ],
    }

    settings = apply_all_tag_to_regions(settings)

    assert len(settings["renewables_clusters"]) == 6
