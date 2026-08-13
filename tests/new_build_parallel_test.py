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


def _minimal_settings(clustering_n_jobs):
    return {
        "new_resources": [("NaturalGas", "CT", "Moderate", 100)],
        "model_year": 2030,
        "model_regions": ["R1", "R2"],
        "resource_cap_recovery_years": 20,
        "clustering_n_jobs": clustering_n_jobs,
    }


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
