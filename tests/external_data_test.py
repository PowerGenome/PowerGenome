import pandas as pd

import powergenome.external_data as external_data


def test_load_demand_segments_uses_data_manager(monkeypatch):
    expected = pd.DataFrame({"Voll": [1000], "Demand_segment": [1]})
    calls = []

    def fake_get_data(table_name):
        calls.append(table_name)
        return expected

    monkeypatch.setattr(external_data, "get_data", fake_get_data)

    result = external_data.load_demand_segments(
        {"demand_segments_fn": "segments.parquet"}
    )

    pd.testing.assert_frame_equal(result, expected)
    assert calls == ["demand_segments"]


def test_load_policy_scenarios_uses_data_manager(monkeypatch):
    expected = pd.DataFrame(
        {"case_id": ["base"], "year": [2030], "region": ["A"], "CO2_cap": [1]}
    )
    monkeypatch.setattr(external_data, "get_data", lambda table_name: expected)

    result = external_data.load_policy_scenarios(
        {"emission_policies_fn": "policies.parquet", "case_id": "base"}
    )

    assert result.index.names == ["case_id", "year"]
    assert result.loc[("base", 2030), "CO2_cap"] == 1
