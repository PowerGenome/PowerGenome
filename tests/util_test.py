"""
Test util functions
"""
import logging
import pytest

import powergenome
from powergenome.util import apply_all_tag_to_regions


logger = logging.getLogger(powergenome.__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    # More extensive test-like formatter...
    "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s",
    # This is the datetime format string.
    "%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def test_apply_all_tag_to_regions(caplog):
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
            {
                "region": "all",
                "technology": "offshorewind",
                "pref_site": True,
                "bin": {"feature": "lcoe", "q": 4},
            },
            {
                "region": "c",
                "technology": "offshorewind",
                "pref_site": True,
                "cluster": {"feature": "lcoe", "n_clusters": 4},
            },
            {
                "region": "all",
                "technology": "offshorewind",
                "pref_site": True,
                "group": ["metro_id"],
            },
        ],
    }

    # Check for warning that "all" is applied to offshore wind more than once
    caplog.set_level(logging.WARNING)
    settings = apply_all_tag_to_regions(settings)

    assert "Multiple 'all' tags applied" in caplog.text

    assert len(settings["renewables_clusters"]) == 9
    for d in settings["renewables_clusters"]:
        if d["technology"] == "landbasedwind":
            if d["region"] == "b":
                assert "filter" in d.keys()
            else:
                assert "bin" in d.keys()
        if d["technology"] == "utilitypv":
            assert "group" in d.keys()
        if d["technology"] == "offshorewind":
            if d["region"] == "c":
                assert "cluster" in d.keys()
            else:
                assert "group" in d.keys()

    # Test two ways to raise a KeyError: no "region" and no "technology" when region is "all"
    d = {"technology": "solarpv"}
    settings["renewables_clusters"].append(d)
    with pytest.raises(KeyError):
        apply_all_tag_to_regions(settings)

    settings["renewables_clusters"].pop()

    d = {"region": "ALL"}
    settings["renewables_clusters"].append(d)
    with pytest.raises(KeyError):
        apply_all_tag_to_regions(settings)
