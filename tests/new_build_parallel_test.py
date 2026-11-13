import logging

import pandas as pd

from powergenome import new_build
from powergenome.settings import Settings


class _UnpicklableBuilder:
    def __getstate__(self):
        raise TypeError("Cannot pickle this object")


def _minimal_resource_inputs():
    resource_costs = pd.DataFrame(
        {
            "technology": ["NaturalGas"],
            "tech_detail": ["CT"],
            "cost_case": ["Moderate"],
            "basis_year": [2030],
            "fixed_o_m_mw": [1.0],
            "fixed_o_m_mwh": [0.0],
            "variable_o_m_mwh": [1.0],
            "capex_mw": [1000.0],
            "capex_mwh": [0.0],
            "wacc_real": [0.07],
        }
    )
    resource_hr = pd.DataFrame(
        {
            "technology": ["NaturalGas"],
            "tech_detail": ["CT"],
            "cost_case": ["Moderate"],
            "basis_year": [2030],
            "heat_rate": [7.0],
        }
    )
    return resource_costs, resource_hr


def _minimal_settings(clustering_n_jobs, new_gen_not_available=None):
    settings = {
        "new_resources": [("NaturalGas", "CT", "Moderate", 100)],
        "model_year": 2030,
        "model_regions": ["R1", "R2"],
        "resource_cap_recovery_years": 20,
        "clustering_n_jobs": clustering_n_jobs,
    }
    if new_gen_not_available is not None:
        settings["new_gen_not_available"] = new_gen_not_available
    return settings


def _region_resource_frame():
    """New-build rows for one region, as seen by parallel_region_renewables."""
    return pd.DataFrame(
        {
            "technology": ["NaturalGas_CT_Moderate"],
            "Fixed_OM_Cost_per_MWyr": [1.0],
            "Var_OM_Cost_per_MWh": [1.0],
            "Inv_Cost_per_MWyr": [100.0],
            "Inv_Cost_per_MWhyr": [0.0],
        }
    )


def _regional_cost_factors():
    return pd.DataFrame(
        {
            "region": ["R1", "R2"],
            "technology": ["NaturalGas", "NaturalGas"],
            "value": [1.0, 1.1],
        }
    )


_USE_FAKE_BUILDER = object()


def _run_region(region, settings, cluster_builder=_USE_FAKE_BUILDER):
    """Call parallel_region_renewables with single-region boilerplate."""
    if cluster_builder is _USE_FAKE_BUILDER:
        cluster_builder = object()
    return new_build.parallel_region_renewables(
        settings,
        _region_resource_frame(),
        _regional_cost_factors(),
        rev_mult_region_map={"R1": "R1", "R2": "R2"},
        rev_mult_tech_map={"NaturalGas_CT_Moderate": "NaturalGas"},
        region=region,
        cluster_builder=cluster_builder,
    )


def test_parallel_clustering_drops_unpicklable_cluster_builder(monkeypatch):
    captured_cluster_builders = []

    class FakeParallel:
        def __init__(self, n_jobs):
            self.n_jobs = n_jobs

        def __call__(self, tasks):
            outputs = []
            for func, args, kwargs in tasks:
                cluster_builder = (
                    args[6] if len(args) > 6 else kwargs.get("cluster_builder")
                )
                captured_cluster_builders.append(cluster_builder)
                outputs.append(
                    pd.DataFrame(
                        {
                            "technology": ["NaturalGas_CT_Moderate"],
                            "Fixed_OM_Cost_per_MWyr": [1],
                            "Fixed_OM_Cost_per_MWhyr": [0],
                            "Inv_Cost_per_MWyr": [1],
                            "Inv_Cost_per_MWhyr": [0],
                            "Var_OM_Cost_per_MWh": [1.0],
                        }
                    )
                )
            return outputs

    monkeypatch.setattr(new_build, "Parallel", FakeParallel)
    monkeypatch.setattr(new_build, "apply_all_tag_to_regions", lambda s: s)
    monkeypatch.setattr(
        new_build,
        "get_data",
        lambda table_name, **kwargs: (
            pd.DataFrame(
                {
                    "region": ["R1"],
                    "technology": ["NaturalGas"],
                    "value": [1.0],
                }
            )
            if table_name == "regional_cost_factor"
            else pd.DataFrame()
        ),
    )

    resource_costs, resource_hr = _minimal_resource_inputs()
    settings = _minimal_settings(clustering_n_jobs=2)

    result = new_build.build_new_resources(
        resource_costs=resource_costs,
        resource_hr=resource_hr,
        settings=settings,
        cluster_builder=_UnpicklableBuilder(),
    )

    assert all(cb is None for cb in captured_cluster_builders)
    assert not result.empty


def test_single_job_clustering_keeps_cluster_builder(monkeypatch):
    captured_cluster_builders = []
    builder = _UnpicklableBuilder()

    class FakeParallel:
        def __init__(self, n_jobs):
            self.n_jobs = n_jobs

        def __call__(self, tasks):
            outputs = []
            for func, args, kwargs in tasks:
                cluster_builder = (
                    args[6] if len(args) > 6 else kwargs.get("cluster_builder")
                )
                captured_cluster_builders.append(cluster_builder)
                outputs.append(
                    pd.DataFrame(
                        {
                            "technology": ["NaturalGas_CT_Moderate"],
                            "Fixed_OM_Cost_per_MWyr": [1],
                            "Fixed_OM_Cost_per_MWhyr": [0],
                            "Inv_Cost_per_MWyr": [1],
                            "Inv_Cost_per_MWhyr": [0],
                            "Var_OM_Cost_per_MWh": [1.0],
                        }
                    )
                )
            return outputs

    monkeypatch.setattr(new_build, "Parallel", FakeParallel)
    monkeypatch.setattr(new_build, "apply_all_tag_to_regions", lambda s: s)
    monkeypatch.setattr(
        new_build,
        "get_data",
        lambda table_name, **kwargs: (
            pd.DataFrame(
                {
                    "region": ["R1"],
                    "technology": ["NaturalGas"],
                    "value": [1.0],
                }
            )
            if table_name == "regional_cost_factor"
            else pd.DataFrame()
        ),
    )

    resource_costs, resource_hr = _minimal_resource_inputs()
    settings = _minimal_settings(clustering_n_jobs=1)

    result = new_build.build_new_resources(
        resource_costs=resource_costs,
        resource_hr=resource_hr,
        settings=settings,
        cluster_builder=builder,
    )

    assert all(cb is builder for cb in captured_cluster_builders)
    assert not result.empty


def test_parallel_clustering_passes_plain_dict_settings(monkeypatch):
    captured_settings = []

    class FakeParallel:
        def __init__(self, n_jobs):
            self.n_jobs = n_jobs

        def __call__(self, tasks):
            outputs = []
            for func, args, kwargs in tasks:
                settings_arg = args[0] if len(args) > 0 else kwargs.get("settings")
                captured_settings.append(settings_arg)
                outputs.append(
                    pd.DataFrame(
                        {
                            "technology": ["NaturalGas_CT_Moderate"],
                            "Fixed_OM_Cost_per_MWyr": [1],
                            "Fixed_OM_Cost_per_MWhyr": [0],
                            "Inv_Cost_per_MWyr": [1],
                            "Inv_Cost_per_MWhyr": [0],
                            "Var_OM_Cost_per_MWh": [1.0],
                        }
                    )
                )
            return outputs

    monkeypatch.setattr(new_build, "Parallel", FakeParallel)
    monkeypatch.setattr(new_build, "apply_all_tag_to_regions", lambda s: s)
    monkeypatch.setattr(
        new_build,
        "get_data",
        lambda table_name, **kwargs: (
            pd.DataFrame(
                {
                    "region": ["R1"],
                    "technology": ["NaturalGas"],
                    "value": [1.0],
                }
            )
            if table_name == "regional_cost_factor"
            else pd.DataFrame()
        ),
    )

    resource_costs, resource_hr = _minimal_resource_inputs()
    settings = Settings(data=_minimal_settings(clustering_n_jobs=2))

    result = new_build.build_new_resources(
        resource_costs=resource_costs,
        resource_hr=resource_hr,
        settings=settings,
    )

    assert captured_settings
    assert all(isinstance(s, dict) for s in captured_settings)
    assert not any(isinstance(s, Settings) for s in captured_settings)
    assert not result.empty


# ─────────────────────────────────────────────────────────────────────────────
# new_gen_not_available applied per region
# ─────────────────────────────────────────────────────────────────────────────


def test_region_exclusion_drops_technology_in_one_region_only():
    settings = {"new_gen_not_available": {"R1": ["NaturalGas_CT_Moderate"]}}

    r1 = _run_region("R1", settings)
    r2 = _run_region("R2", settings)

    assert r1.empty
    assert list(r2["technology"]) == ["NaturalGas_CT_Moderate"]
    assert list(r2["region"]) == ["R2"]


def test_all_regions_key_excludes_everywhere():
    settings = {"new_gen_not_available": {"ALL_REGIONS": ["NaturalGas*"]}}

    assert _run_region("R1", settings).empty
    assert _run_region("R2", settings).empty


def test_all_regions_and_region_keys_are_additive():
    settings = {
        "new_gen_not_available": {
            "all": ["Coal_*"],
            "R1": ["NaturalGas*"],
        }
    }

    assert _run_region("R1", settings).empty
    assert list(_run_region("R2", settings)["region"]) == ["R2"]


def test_exclusion_wildcard_is_case_insensitive():
    settings = {"new_gen_not_available": {"R1": ["naturalgas_*"]}}

    assert _run_region("R1", settings).empty
    assert not _run_region("R2", settings).empty


def test_fully_excluded_region_skips_clustering(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("build_resource_clusters must not be called")

    monkeypatch.setattr(new_build, "build_resource_clusters", fail)
    settings = {"new_gen_not_available": {"R1": ["NaturalGas_CT_Moderate"]}}

    result = _run_region("R1", settings, cluster_builder=None)

    assert result.empty
    # build_new_resources post-processing needs these columns even when a region
    # contributes no rows.
    for column in (
        "Inv_Cost_per_MWyr",
        "Fixed_OM_Cost_per_MWyr",
        "Var_OM_Cost_per_MWh",
        "technology",
        "region",
    ):
        assert column in result.columns


class _ExecutingParallel:
    """Run the delayed tasks in-process instead of discarding their output."""

    def __init__(self, n_jobs):
        self.n_jobs = n_jobs

    def __call__(self, tasks):
        return [func(*args, **kwargs) for func, args, kwargs in tasks]


def test_build_new_resources_applies_exclusions_end_to_end(monkeypatch):
    monkeypatch.setattr(new_build, "Parallel", _ExecutingParallel)
    monkeypatch.setattr(new_build, "apply_all_tag_to_regions", lambda s: s)
    monkeypatch.setattr(
        new_build,
        "get_data",
        lambda table_name, **kwargs: (
            _regional_cost_factors()
            if table_name == "regional_cost_factor"
            else pd.DataFrame()
        ),
    )
    resource_costs, resource_hr = _minimal_resource_inputs()
    settings = _minimal_settings(
        clustering_n_jobs=1,
        new_gen_not_available={"R1": ["NaturalGas*"]},
    )

    result = new_build.build_new_resources(
        resource_costs=resource_costs,
        resource_hr=resource_hr,
        settings=settings,
        cluster_builder=_UnpicklableBuilder(),
    )

    assert not result.empty
    assert set(result["region"]) == {"R2"}
    assert set(result["technology"]) == {"NaturalGas_CT_Moderate"}
    assert pd.api.types.is_integer_dtype(result["Inv_Cost_per_MWyr"])
    assert pd.api.types.is_float_dtype(result["Var_OM_Cost_per_MWh"])


# ─────────────────────────────────────────────────────────────────────────────
# renewables_clusters parity with exclusions
# ─────────────────────────────────────────────────────────────────────────────


def _cluster_input_frame():
    return pd.DataFrame({"technology": ["NaturalGas_CT_Moderate"], "region": ["R1"]})


def _cluster_settings():
    return {
        "renewables_clusters": [{"region": "R1", "technology": "utilitypv"}],
        "region_aggregations": {},
    }


def test_excluded_renewable_cluster_does_not_warn(caplog):
    excluded_df = pd.DataFrame(
        {"technology": ["UtilityPV_Class1_Moderate"], "region": ["R1"]}
    )

    with caplog.at_level(logging.WARNING, logger="powergenome.new_build"):
        result = new_build.add_renewables_clusters(
            _cluster_input_frame(),
            "R1",
            _cluster_settings(),
            cluster_builder=object(),
            excluded_df=excluded_df,
        )

    # The cluster entry has no comparable new-build technology only because the
    # technology was excluded, which is expected rather than a settings mistake.
    assert "no comparable new-build technology" not in caplog.text
    assert list(result["technology"]) == ["NaturalGas_CT_Moderate"]


def test_cluster_target_never_in_build_list_still_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="powergenome.new_build"):
        new_build.add_renewables_clusters(
            _cluster_input_frame(),
            "R1",
            _cluster_settings(),
            cluster_builder=object(),
        )

    assert "no comparable new-build technology" in caplog.text


# ─────────────────────────────────────────────────────────────────────────────
# Unmatched exclusion patterns
# ─────────────────────────────────────────────────────────────────────────────


def test_warn_unmatched_exclusions_warns_once(caplog):
    df = pd.DataFrame({"technology": ["NaturalGas_CT_Moderate"]})
    settings = {"new_gen_not_available": {"R1": ["Coal_new"]}}

    with caplog.at_level(logging.WARNING, logger="powergenome.new_build"):
        new_build._warn_unmatched_exclusions(df, settings)

    messages = [
        record.getMessage()
        for record in caplog.records
        if "matches no new-build technology" in record.getMessage()
    ]
    assert len(messages) == 1
    assert "Coal_new" in messages[0]


def test_warn_unmatched_exclusions_ignores_matching_pattern(caplog):
    df = pd.DataFrame({"technology": ["NaturalGas_CT_Moderate"]})
    settings = {"new_gen_not_available": {"all": ["naturalgas_*"]}}

    with caplog.at_level(logging.WARNING, logger="powergenome.new_build"):
        new_build._warn_unmatched_exclusions(df, settings)

    assert "matches no new-build technology" not in caplog.text
