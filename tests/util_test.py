"""
Test util functions
"""
import logging
import pytest

import powergenome
from powergenome.util import apply_all_tag_to_regions, sort_nested_dict


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


def test_sort_nested_dict():
    test_dict1 = {"one": 1, "threeee": 3, "twoo": 2}
    sorted_dict1 = sort_nested_dict(test_dict1)
    # Here, the keys should be ordered as 'one', 'twoo' and 'threeee'
    assert list(sorted_dict1.keys()) == ["one", "twoo", "threeee"]

    test_dict2 = {"threeee": {"fourrrrr": 4, "twoo": 2}, "one": 1, "fiveeeee": 5}
    sorted_dict2 = sort_nested_dict(test_dict2)
    # Here, the keys at the top level should be ordered as 'one', 'threeee' and 'fiveeeee'
    assert list(sorted_dict2.keys()) == ["one", "threeee", "fiveeeee"]
    # And within the dictionary mapped to by 'threeee', the keys should be 'twoo' and 'fourrrrr'
    assert list(sorted_dict2["threeee"].keys()) == ["twoo", "fourrrrr"]

    test_dict3 = {"a": {"b": 4, "a": 2}, "c": 1, "b": 5}
    sorted_dict3 = sort_nested_dict(test_dict3)
    assert list(sorted_dict3.keys()) == ["a", "c", "b"]
    assert list(sorted_dict3["a"].keys()) == ["b", "a"]


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

    settings = {"model_regions": ["a", "b", "c"], "renewables_clusters": None}
    apply_all_tag_to_regions(settings)
    settings = {"model_regions": ["a", "b", "c"]}
    apply_all_tag_to_regions(settings)
