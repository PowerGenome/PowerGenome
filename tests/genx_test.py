import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from powergenome.GenX import (
    DEFAULT_COLS,
    RESOURCE_COLUMNS,
    FolderStructure,
    GenXInputData,
    check_vre_profiles,
    create_multistage_df,
    create_policy_df,
    create_resource_df,
    filter_empty_columns,
    process_genx_data,
    set_must_run_generation,
    split_generators_data,
    update_newbuild_canretire,
)


# Tests setting the generation of a single must run resource to 1 in all hours.
def test_set_must_run_generation_single_tech():
    gen_variability = pd.DataFrame(
        {"gen_1": [0.5, 0.6, 0.7], "gen_2": [0.8, 0.9, 1.0], "gen_3": [0.0, 0.0, 0.0]}
    )
    must_run_techs = ["gen_3"]
    expected_output = pd.DataFrame(
        {"gen_1": [0.5, 0.6, 0.7], "gen_2": [0.8, 0.9, 1.0], "gen_3": [1.0, 1.0, 1.0]}
    )
    assert set_must_run_generation(gen_variability, must_run_techs).equals(
        expected_output
    )


# Tests setting the generation of multiple must run resources to 1 in all hours.
def test_set_must_run_generation_multiple_techs():
    gen_variability = pd.DataFrame(
        {"gen_1": [0.5, 0.6, 0.7], "gen_2": [0.8, 0.9, 1.0], "gen_3": [0.0, 0.0, 0.0]}
    )
    must_run_techs = ["gen_2", "gen_3"]
    expected_output = pd.DataFrame(
        {"gen_1": [0.5, 0.6, 0.7], "gen_2": [1.0, 1.0, 1.0], "gen_3": [1.0, 1.0, 1.0]}
    )
    assert set_must_run_generation(gen_variability, must_run_techs).equals(
        expected_output
    )


# Tests behavior when input dataframe is empty.
def test_set_must_run_generation_empty_dataframe():
    gen_variability = pd.DataFrame()
    must_run_techs = ["gen_3"]
    expected_output = pd.DataFrame()
    assert set_must_run_generation(gen_variability, must_run_techs).equals(
        expected_output
    )


# Tests behavior when list of must run techs is empty.
def test_set_must_run_generation_empty_tech_list():
    gen_variability = pd.DataFrame(
        {"gen_1": [0.5, 0.6, 0.7], "gen_2": [0.8, 0.9, 1.0], "gen_3": [0.0, 0.0, 0.0]}
    )
    must_run_techs = []
    expected_output = pd.DataFrame(
        {"gen_1": [0.5, 0.6, 0.7], "gen_2": [0.8, 0.9, 1.0], "gen_3": [0.0, 0.0, 0.0]}
    )
    assert set_must_run_generation(gen_variability, must_run_techs).equals(
        expected_output
    )


# Tests setting the generation of no must run resources to 1 in all hours.
def test_set_must_run_generation_no_techs():
    gen_variability = pd.DataFrame(
        {"gen_1": [0.5, 0.6, 0.7], "gen_2": [0.8, 0.9, 1.0], "gen_3": [0.0, 0.0, 0.0]}
    )
    expected_output = pd.DataFrame(
        {"gen_1": [0.5, 0.6, 0.7], "gen_2": [0.8, 0.9, 1.0], "gen_3": [0.0, 0.0, 0.0]}
    )
    assert set_must_run_generation(gen_variability).equals(expected_output)


# Tests that the function logs a warning message when trying to set a must run resource that is not found in the generation variability dataframe.
def test_set_must_run_generation_tech_not_found(caplog):
    import logging

    from pandas.testing import assert_frame_equal

    gen_variability = pd.DataFrame(
        {"gen_1": [0.5, 0.6, 0.7], "gen_2": [0.8, 0.9, 1.0], "gen_3": [0.0, 0.0, 0.0]}
    )
    must_run_techs = ["gen_4"]
    expected_output = pd.DataFrame(
        {"gen_1": [0.5, 0.6, 0.7], "gen_2": [0.8, 0.9, 1.0], "gen_3": [0.0, 0.0, 0.0]}
    )

    with caplog.at_level(logging.WARNING):
        assert_frame_equal(
            set_must_run_generation(gen_variability, must_run_techs), expected_output
        )
        assert "Trying to set gen_4 as a must run resource" in caplog.text


class TestCheckVreProfiles:

    # Given a dataframe of generators with a "Resource" column and a dataframe of hourly generation for each resource, when all VRE resources have variable generation profiles, then no warning is issued.
    def test_all_vre_variable_profiles(self, caplog):
        gen_df = pd.DataFrame({"Resource": ["VRE1", "VRE2", "VRE3"], "VRE": [1, 1, 1]})
        gen_var_df = pd.DataFrame(
            {"VRE1": [0, 1, 0], "VRE2": [1, 0, 1], "VRE3": [0, 1, 0]}
        )

        with caplog.at_level(logging.WARNING):
            check_vre_profiles(gen_df, gen_var_df)

        assert len(caplog.records) == 0

    # Given an empty dataframe of generators and an empty dataframe of hourly generation for each resource, then no warning is issued.
    def test_empty_dataframes(self, caplog):
        gen_df = pd.DataFrame(columns=["Resource"])
        gen_var_df = pd.DataFrame()

        with caplog.at_level(logging.WARNING):
            check_vre_profiles(gen_df, gen_var_df)

        assert len(caplog.records) == 0

    # Given a dataframe of generators with a "Resource" column and a dataframe of hourly generation for each resource, when all VRE resources have variable generation profiles, then no warning should be issued.
    def test_non_variable_vre_profiles_warning(self, caplog):
        gen_df = pd.DataFrame({"Resource": ["VRE1", "VRE2", "VRE3"], "VRE": [1, 1, 1]})
        gen_var_df = pd.DataFrame(
            {"VRE1": [0, 1, 0], "VRE2": [1, 1, 1], "VRE3": [0, 1, 0]}
        )

        with caplog.at_level(logging.WARNING):
            check_vre_profiles(gen_df, gen_var_df)

        assert len(caplog.records) == 1
        assert (
            "The variable resources ['VRE2'] have non-variable generation profiles."
            in caplog.records[0].message
        )

    def test_custom_vre_column(self, caplog):
        gen_df = pd.DataFrame(
            {
                "Resource": ["VRE1", "VRE2", "VRE3"],
                "VRE": [0, 1, 0],
                "VRE_STOR": [1, 0, 1],
            }
        )
        gen_var_df = pd.DataFrame(
            {"VRE1": [0, 1, 0], "VRE2": [1, 1, 1], "VRE3": [0, 1, 0]}
        )

        with caplog.at_level(logging.WARNING):
            check_vre_profiles(gen_df, gen_var_df, vre_cols=["VRE", "VRE_STOR"])

        assert len(caplog.records) == 1
        assert (
            "The variable resources ['VRE2'] have non-variable generation profiles."
            in caplog.records[0].message
        )


def test_update_newbuild_canretire():
    """Test the update_newbuild_canretire function."""
    # Test case 1: Default behaviour
    input_df = pd.DataFrame({"New_Build": [1, -1, 0, 1], "Other_Column": [1, 2, 3, 4]})
    expected_df = pd.DataFrame(
        {
            "New_Build": [1, 0, 0, 1],
            "Other_Column": [1, 2, 3, 4],
            "Can_Retire": [1, 0, 1, 1],
        }
    )
    result_df = update_newbuild_canretire(input_df)
    pd.testing.assert_frame_equal(result_df, expected_df)

    # Test case 2: Missing New_Build column
    input_df = pd.DataFrame({"Other_Column": [1, 2, 3]})
    result_df = update_newbuild_canretire(input_df)
    pd.testing.assert_frame_equal(result_df, input_df)

    # Test case 3: Empty DataFrame
    input_df = pd.DataFrame(
        {
            "New_Build": pd.Series([], dtype=int),
            "Other_Column": pd.Series([], dtype=int),
        }
    )
    expected_df = pd.DataFrame(
        {
            "New_Build": pd.Series([], dtype=int),
            "Other_Column": pd.Series([], dtype=int),
            "Can_Retire": pd.Series([], dtype=int),
        }
    )
    result_df = update_newbuild_canretire(input_df)
    pd.testing.assert_frame_equal(result_df, expected_df)

    # Test case 4: DataFrame with NaN values
    input_df = pd.DataFrame(
        {"New_Build": [1, np.nan, -1, 0], "Other_Column": [1, 2, 3, 4]}
    )
    expected_df = pd.DataFrame(
        {"New_Build": [1, np.nan, -1, 0], "Other_Column": [1, 2, 3, 4]}
    )
    result_df = update_newbuild_canretire(input_df)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_filter_empty_columns():
    """Test filter_empty_columns function with various test cases."""
    # Test case 1: Default behaviour
    df_basic = pd.DataFrame(
        {
            "valid_col": [1, 2, 3],
            "zero_col": [0, 0, 0],
            "none_col": [None, None, None],
            "string_none_col": ["None", "None", "None"],
            "mixed_col": [1, None, "None"],
            "valid_string": ["a", "b", "c"],
        }
    )
    result = filter_empty_columns(df_basic)
    assert sorted(result) == sorted(
        ["valid_col", "mixed_col", "valid_string"]
    ), "Basic case failed"

    # Test case 2: Empty DataFrame
    df_empty = pd.DataFrame({})
    assert filter_empty_columns(df_empty) == [], "Empty DataFrame case failed"

    # Test case 3: All invalid columns
    df_invalid = pd.DataFrame(
        {
            "zero_col": [0, 0],
            "none_col": [None, None],
            "string_none_col": ["None", "None"],
        }
    )
    assert filter_empty_columns(df_invalid) == [], "All invalid columns case failed"

    # Test case 4: Mixed types
    df_mixed = pd.DataFrame(
        {
            "mixed_types": [1, "text", 3.14],
            "mixed_invalid": [0, None, "None"],  # For now, mixed types are allowed
            "bool_col": [True, False, True],
        }
    )
    assert sorted(filter_empty_columns(df_mixed)) == sorted(
        ["mixed_types", "mixed_invalid", "bool_col"]
    ), "Mixed types case failed"

    # Test case 5: Edge cases
    df_edge = pd.DataFrame(
        {
            "empty_str": ["", "", ""],
            "whitespace": [" ", "  ", "\t"],
            "zero_float": [0.0, 0.0, 0.0],
            "valid_zero_one": [0, 1, 0],
        }
    )
    assert sorted(filter_empty_columns(df_edge)) == sorted(
        ["whitespace", "valid_zero_one"]
    ), "Edge cases failed"

    # Test case 6: NaN values
    df_nan = pd.DataFrame(
        {
            "nan_col": [np.nan, np.nan, np.nan],
            "mixed_nan": [1, np.nan, 3],
            "inf_col": [np.inf, -np.inf, np.inf],
        }
    )
    assert sorted(filter_empty_columns(df_nan)) == sorted(
        ["mixed_nan", "inf_col"]
    ), "NaN values case failed"


def test_filter_empty_columns_errors():
    """Test error handling in filter_empty_columns."""
    # Test with non-DataFrame input
    with pytest.raises(AttributeError):
        filter_empty_columns([1, 2, 3])


@pytest.fixture
def sample_gen_data():
    """Create sample generator data for testing."""
    return pd.DataFrame(
        {
            # Default columns
            "Resource": [
                "therm_nc",
                "stor_sym",
                "therm_uc",
                "stor_asym",
                "must_run_1",
                "flex_1",
                "vre_1",
                "hydro_1",
                "vre_2",
            ],
            "Zone": [1, 2, 1, 3, 4, 5, 6, 2, 2],
            "New_Build": [1, 0, 1, 0, 0, 1, 1, 1, 1],
            "Can_Retire": [1, 1, 1, 1, 0, 1, 0, 1, 1],
            # Resource tag columns
            "THERM": [1, 0, 2, 0, 0, 0, 0, 0, 0],
            "STOR": [0, 1, 0, 2, 0, 0, 0, 0, 0],
            "VRE": [0, 0, 0, 0, 0, 0, 1, 0, 1],
            "MUST_RUN": [0, 0, 0, 0, 1, 0, 0, 0, 0],
            "FLEX": [0, 0, 0, 0, 0, 1, 0, 0, 0],
            "HYDRO": [0, 0, 0, 0, 0, 0, 0, 1, 0],
            # Resource specific columns
            "Min_Power": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # THERM column
            "Down_Time": [6.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # THERM column
            "Up_Time": [6.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # THERM column
            "Eff_Down": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # HYDRO column
            "Eff_Up": [0.0, 0.8, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0, 0.0],  # STOR column
            "Max_Cap_MWh": [0.0, 200, 0.0, 100, 0.0, 0.0, 0.0, 0.0, 0.0],  # STOR column
            "LDS": [0, 1, 0, 1, 0, 0, 0, 1, 0],  # HYDRO and STOR column
            "Num_VRE_Bins": [0, 0, 0, 0, 0, 0, 1, 0, 1],  # VRE column
            "Ramp_Up_Percentage": [
                0.1,
                0.0,
                0.2,
                0.0,
                0.0,
                0.0,
                0.0,
                0.8,
                0.0,
            ],  # THERM column
            "Ramp_Dn_Percentage": [
                0.3,
                0.0,
                0.5,
                0.0,
                0.0,
                0.0,
                0.0,
                0.7,
                0.0,
            ],  # THERM column
            "Flexible_Demand_Energy_Eff": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.9,
                0.0,
                0.0,
                0.0,
            ],  # FLEX column
            "Max_Flexible_Demand_Advance": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.3,
                0.0,
                0.0,
                0.0,
            ],  # FLEX column
            "Max_Flexible_Demand_Delay": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.3,
                0.0,
                0.0,
                0.0,
            ],  # FLEX column
            "Var_OM_Cost_per_MWh_In": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                100,
                0.0,
                0.0,
                0.0,
            ],  # FLEX column
            "Hydro_Energy_to_Power_Ratio": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.8,
                0.0,
            ],  # HYDRO column
            # Policy columns
            "ESR_1": [1, 0, 1, 0, 0, 1, 0, 0, 0],
            "ESR_2": [0, 1, 1, 0, 0, 0, 1, 0, 0],
            "CapRes_1": [0.5, 0.0, 0.7, 0.1, 0, 0.5, 0, 0, 0],
            "CapRes_2": [0.0, 0.5, 0.0, 0.3, 0, 0.0, 0, 0, 0],
            "MinCapTag_1": [1, 0, 1, 0, 0, 0, 0, 0, 0],
            "MinCapTag_2": [0, 1, 1, 0, 0, 1, 0, 0, 0],
            "MaxCapTag_1": [1, 0, 1, 0, 0, 0, 1, 0, 1],
            "MaxCapTag_2": [0, 1, 1, 0, 0, 1, 0, 0, 1],
            # Multi-stage columns
            "WACC": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            "Capital_Recovery_Period": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "Lifetime": [10, 10, 20, 20, 30, 30, 40, 40, 50],
            # R_ID
            "R_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            # Other columns
            "Extra_Col": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )


def test_create_resource_df(sample_gen_data):
    """Test create_resource_df function with various test cases."""

    input_df = sample_gen_data

    # Test case 1: THERM resource
    therm_result = create_resource_df(input_df, "THERM")
    assert (
        list(therm_result.columns[:4]) == DEFAULT_COLS
    ), "Default columns should be first"
    assert "Model" in therm_result.columns, "THERM should be renamed to Model"
    assert therm_result["Model"].tolist() == [
        1,
        2,
    ]  # Check that the Model column is correct
    assert "Eff_Up" not in therm_result.columns, "STOR columns should be removed"
    assert len(therm_result) == 2, "Should only include rows where THERM > 0"
    assert all(
        col in therm_result.columns for col in RESOURCE_COLUMNS["THERM"]
    ), "All THERM specific columns should be present"

    # Test case 2: STOR resource
    stor_result = create_resource_df(input_df, "STOR")
    assert (
        list(stor_result.columns[:4]) == DEFAULT_COLS
    ), "Default columns should be first"
    assert "Model" in stor_result.columns, "STOR should be renamed to Model"
    assert stor_result["Model"].tolist() == [
        1,
        2,
    ]  # Check that the Model column is correct
    assert "Min_Power" not in stor_result.columns, "THERM columns should be removed"
    assert stor_result["LDS"].tolist() == [1, 1]  # Check that the LDS column is correct
    assert len(stor_result) == 2, "Should only include rows where STOR > 0"

    # Test case 3: VRE resource
    vre_result = create_resource_df(input_df, "VRE")
    assert (
        list(vre_result.columns[:4]) == DEFAULT_COLS
    ), "Default columns should be first"
    assert "Model" not in vre_result.columns, "VRE should not be renamed to Model"
    assert "Min_Power" not in vre_result.columns, "THERM columns should be removed"
    assert len(vre_result) == 2, "Should only include rows where VRE=1"
    assert all(
        col in vre_result.columns for col in RESOURCE_COLUMNS["VRE"]
    ), "All VRE specific columns should be present"

    # Test case 4: MUST_RUN resource
    must_run_result = create_resource_df(input_df, "MUST_RUN")
    assert (
        list(must_run_result.columns[:4]) == DEFAULT_COLS
    ), "Default columns should be first"
    assert (
        "Model" not in must_run_result.columns
    ), "MUST_RUN should not be renamed to Model"
    assert "Min_Power" not in must_run_result.columns, "THERM columns should be removed"
    assert len(must_run_result) == 1, "Should only include rows where MUST_RUN=1"
    assert all(
        col in must_run_result.columns for col in RESOURCE_COLUMNS["MUST_RUN"]
    ), "All MUST_RUN specific columns should be present"

    # Test case 5: FLEX resource
    flex_result = create_resource_df(input_df, "FLEX")
    assert (
        list(flex_result.columns[:4]) == DEFAULT_COLS
    ), "Default columns should be first"
    assert "Model" not in flex_result.columns, "FLEX should not be renamed to Model"
    assert "Min_Power" not in flex_result.columns, "THERM columns should be removed"
    assert len(flex_result) == 1, "Should only include rows where FLEX=1"
    assert all(
        col in flex_result.columns for col in RESOURCE_COLUMNS["FLEX"]
    ), "All FLEX specific columns should be present"

    # Test case 6: HYDRO resource
    hydro_result = create_resource_df(input_df, "HYDRO")
    assert (
        list(hydro_result.columns[:4]) == DEFAULT_COLS
    ), "Default columns should be first"
    assert "Model" not in hydro_result.columns, "HYDRO should not be renamed to Model"
    assert "Min_Power" in hydro_result.columns, "THERM columns should be removed"
    assert len(hydro_result) == 1, "Should only include rows where HYDRO=1"
    assert hydro_result["LDS"].tolist() == [1]  # Check that the LDS column is correct

    # Test case 7: Invalid resource tag
    invalid_result = create_resource_df(input_df, "INVALID")
    assert (
        invalid_result.empty
    ), "Should return empty DataFrame for invalid resource tag"

    # Test case 8: Empty DataFrame
    empty_result = create_resource_df(pd.DataFrame(), "THERM")
    assert empty_result.empty, "Should return empty DataFrame for empty input"

    # Test case 9: Missing required columns
    incomplete_df = pd.DataFrame({"THERM": [1, 0, 1], "Extra_Col": [1, 2, 3]})
    incomplete_result = create_resource_df(incomplete_df, "THERM")
    assert set(DEFAULT_COLS) - set(
        incomplete_result.columns
    ), "Should warn about missing required columns"


def test_create_resource_df_all_zero_columns():
    """Test handling of columns with all zero values."""
    input_df = pd.DataFrame(
        {
            "Resource": ["therm_1", "therm_2"],
            "Zone": [1, 2],
            "New_Build": [1, 0],
            "THERM": [1, 1],
            "Min_Power": [0, 0],  # All zeros
            "Ramp_Up_Percentage": [0.1, 0.2],
        }
    )

    result = create_resource_df(input_df, "THERM")
    assert "Ramp_Up_Percentage" in result.columns, "Should keep non-zero columns"
    assert (
        "Min_Power" not in result.columns
    ), "Should remove all-zero columns unless in DEFAULT_COLS"


def test_create_resource_df_none_values():
    """Test handling of None values."""
    input_df = pd.DataFrame(
        {
            "Resource": ["therm_1", "therm_2"],
            "Zone": [1, 2],
            "New_Build": [1, 0],
            "THERM": [1, 1],
            "Min_Power": [None, None],  # All None
            "Ramp_Up_Percentage": [0.1, None],  # Mixed None
        }
    )

    result = create_resource_df(input_df, "THERM")
    assert (
        "Ramp_Up_Percentage" in result.columns
    ), "Should keep columns with some valid values"
    assert (
        "Min_Power" not in result.columns
    ), "Should remove columns with all None values"


def test_create_resource_df_column_order():
    """Test that column order is correct."""
    input_df = pd.DataFrame(
        {
            "Extra_First": [1, 2],
            "Resource": ["therm_1", "therm_2"],
            "Zone": [1, 2],
            "New_Build": [1, 0],
            "Can_Retire": [1, 1],
            "THERM": [1, 1],
            "Extra_Last": [3, 4],
        }
    )

    result = create_resource_df(input_df, "THERM")
    assert list(result.columns)[: len(DEFAULT_COLS)] == [
        col for col in DEFAULT_COLS if col in result.columns
    ], "Default columns should be first and in correct order"


def test_create_policy_df():
    """Test create_policy_df function with various test cases."""
    # Mock data
    input_df = pd.DataFrame(
        {
            "Resource": ["gen_1", "gen_2", "gen_3", "gen_4"],
            "Zone": [1, 2, 3, 4],
            "New_Build": [1, 0, 1, 0],
            "Can_Retire": [1, 1, 1, 1],
            "ESR_1": [1, 0, 1, 0],
            "ESR_2": [0, 1, 1, 0],
            "CapRes_1": [0.5, 0, 0.7, 0],
            "CapRes_2": [0, 0.5, 0, 0.3],
            "MinCapTag_1": [1, 0, 0, 1],
            "MinCapTag_2": [0, 1, 0, 0],
            "MaxCapTag_1": [1, 0, 1, 0],
            "MaxCapTag_2": [0, 1, 0, 1],
            "Other_Col": [1, 2, 3, 4],
        }
    )

    # Test case 1: ESR policy
    esr_policy = {"oldtag": "ESR_", "newtag": "ESR_"}
    esr_result = create_policy_df(input_df.copy(), esr_policy)

    assert list(esr_result.columns) == [
        "Resource",
        "ESR_1",
        "ESR_2",
    ], "Should only include Resource and ESR columns"
    assert len(esr_result) == 3, "Should only include rows with non-zero ESR values"
    assert not any(
        col.startswith("CapRes_") for col in esr_result.columns
    ), "Should not include other policy columns"

    # Test case 2: Capacity reserve policy with tag renaming
    cap_res_policy = {"oldtag": "CapRes_", "newtag": "Derating_factor_"}
    cap_res_result = create_policy_df(input_df.copy(), cap_res_policy)

    assert (
        "Derating_factor_1" in cap_res_result.columns
    ), "Should rename CapRes_ to Derating_factor_"
    assert (
        "Derating_factor_2" in cap_res_result.columns
    ), "Should rename CapRes_ to Derating_factor_"
    assert (
        len(cap_res_result) == 4
    ), "Should only include rows with non-zero CapRes values"
    assert not any(
        col.startswith("ESR_") for col in cap_res_result.columns
    ), "Should not include other policy columns"
    assert not any(
        col.startswith("MinCapTag_") for col in cap_res_result.columns
    ), "Should not include other policy columns"
    assert not any(
        col.startswith("MaxCapTag_") for col in cap_res_result.columns
    ), "Should not include other policy columns"
    assert cap_res_result["Derating_factor_1"].tolist() == [
        0.5,
        0.0,
        0.7,
        0.0,
    ], "Should include the correct values"
    assert cap_res_result["Derating_factor_2"].tolist() == [
        0.0,
        0.5,
        0.0,
        0.3,
    ], "Should include the correct values"

    # Test case 3: Min capacity policy with tag renaming
    min_cap_policy = {"oldtag": "MinCapTag_", "newtag": "Min_Cap_"}
    min_cap_result = create_policy_df(input_df.copy(), min_cap_policy)

    assert "Min_Cap_1" in min_cap_result.columns, "Should rename MinCapTag_ to Min_Cap_"
    assert "Min_Cap_2" in min_cap_result.columns, "Should rename MinCapTag_ to Min_Cap_"
    assert (
        len(min_cap_result) == 3
    ), "Should only include rows with non-zero MinCapTag values"
    assert not any(
        col.startswith("CapRes_") for col in min_cap_result.columns
    ), "Should not include other policy columns"
    assert min_cap_result["Min_Cap_1"].tolist() == [
        1,
        0,
        1,
    ], "Should include the correct values"
    assert min_cap_result["Min_Cap_2"].tolist() == [
        0,
        1,
        0,
    ], "Should include the correct values"

    # Test case 4: Max capacity policy with tag renaming
    max_cap_policy = {"oldtag": "MaxCapTag_", "newtag": "Max_Cap_"}
    max_cap_result = create_policy_df(input_df.copy(), max_cap_policy)

    assert "Max_Cap_1" in max_cap_result.columns, "Should rename MaxCapTag_ to Max_Cap_"
    assert "Max_Cap_2" in max_cap_result.columns, "Should rename MaxCapTag_ to Max_Cap_"
    assert (
        len(max_cap_result) == 4
    ), "Should only include rows with non-zero MaxCapTag values"
    assert not any(
        col.startswith("CapRes_") for col in max_cap_result.columns
    ), "Should not include other policy columns"
    assert max_cap_result["Max_Cap_1"].tolist() == [
        1,
        0,
        1,
        0,
    ], "Should include the correct values"
    assert max_cap_result["Max_Cap_2"].tolist() == [
        0,
        1,
        0,
        1,
    ], "Should include the correct values"

    # Test case 5: Policy tag not present
    missing_policy = {"oldtag": "Missing_", "newtag": "New_"}
    missing_result = create_policy_df(input_df.copy(), missing_policy)

    assert (
        missing_result.empty
    ), "Should return empty DataFrame when policy tag not found"

    # Test case 6: All zero values
    zero_df = pd.DataFrame(
        {"Resource": ["gen_1", "gen_2"], "ESR_1": [0, 0], "ESR_2": [0, 0]}
    )
    zero_result = create_policy_df(zero_df.copy(), esr_policy)

    assert (
        zero_result.empty
    ), "Should return empty DataFrame when all policy values are zero"


def test_create_policy_df_edge_cases():
    """Test create_policy_df with edge cases."""
    # Test case 1: Empty DataFrame
    empty_df = pd.DataFrame()
    policy_info = {"oldtag": "ESR_", "newtag": "ESR_"}
    empty_result = create_policy_df(empty_df, policy_info)

    assert empty_result.empty, "Should handle empty DataFrame"

    # Test case 2: Missing Resource column
    no_resource_df = pd.DataFrame({"ESR_1": [1, 0], "ESR_2": [0, 1]})
    with pytest.raises(KeyError):
        create_policy_df(no_resource_df.copy(), policy_info)

    # Test case 3: Mixed data types
    mixed_df = pd.DataFrame(
        {"Resource": ["gen_1", "gen_2"], "ESR_1": [1, "high"], "ESR_2": [0.5, True]}
    )
    with pytest.raises(ValueError):
        create_policy_df(mixed_df.copy(), policy_info)


def test_create_policy_df_valid_values():
    """Test that valid values pass validation."""
    # Valid ESR values
    valid_esr_df = pd.DataFrame(
        {"Resource": ["gen_1", "gen_2"], "ESR_1": [0, 1], "ESR_2": [1, 0]}
    )
    esr_policy = {"oldtag": "ESR_", "newtag": "ESR_"}
    result = create_policy_df(valid_esr_df, esr_policy)
    assert not result.empty, "Should accept valid ESR values (0 and 1)"

    # Valid CapRes values
    valid_cap_res_df = pd.DataFrame(
        {
            "Resource": ["gen_1", "gen_2"],
            "CapRes_1": [0.5, 1.0],
            "CapRes_2": [0.0, 0.75],
        }
    )
    cap_res_policy = {"oldtag": "CapRes_", "newtag": "CapRes_"}
    result = create_policy_df(valid_cap_res_df, cap_res_policy)
    assert not result.empty, "Should accept valid CapRes values (between 0 and 1)"

    # Valid mixed policy types
    valid_mixed_df = pd.DataFrame(
        {
            "Resource": ["gen_1", "gen_2"],
            "ESR_1": [0, 1],
            "MinCapTag_1": [1, 0],
            "MaxCapTag_1": [1, 1],
            "CapRes_1": [0.5, 0.75],
        }
    )
    # Test each policy type separately
    for policy_type in ["ESR_", "MinCapTag_", "MaxCapTag_", "CapRes_"]:
        policy_info = {"oldtag": policy_type, "newtag": policy_type}
        result = create_policy_df(valid_mixed_df.copy(), policy_info)
        assert not result.empty, f"Should accept valid {policy_type} values"


def test_create_policy_df_invalid_values():
    """Test value validation rules for different policy types."""
    # Test ESR_ validation (must be 0 or 1)
    invalid_esr_df = pd.DataFrame(
        {
            "Resource": ["gen_1", "gen_2"],
            "ESR_1": [0.5, 1],  # 0.5 is invalid
            "ESR_2": [0, 1],
        }
    )
    esr_policy = {"oldtag": "ESR_", "newtag": "ESR_"}
    with pytest.raises(ValueError, match="can only have values 0 or 1"):
        create_policy_df(invalid_esr_df, esr_policy)

    # Test MinCapTag_ validation (must be 0 or 1)
    invalid_min_cap_df = pd.DataFrame(
        {
            "Resource": ["gen_1", "gen_2"],
            "MinCapTag_1": [2, 1],  # 2 is invalid
            "MinCapTag_2": [0, 1],
        }
    )
    min_cap_policy = {"oldtag": "MinCapTag_", "newtag": "Min_Cap_"}
    with pytest.raises(ValueError, match="can only have values 0 or 1"):
        create_policy_df(invalid_min_cap_df, min_cap_policy)

    # Test MaxCapTag_ validation (must be 0 or 1)
    invalid_max_cap_df = pd.DataFrame(
        {
            "Resource": ["gen_1", "gen_2"],
            "MaxCapTag_1": [-1, 1],  # -1 is invalid
            "MaxCapTag_2": [0, 1],
        }
    )
    max_cap_policy = {"oldtag": "MaxCapTag_", "newtag": "Max_Cap_"}
    with pytest.raises(ValueError, match="can only have values 0 or 1"):
        create_policy_df(invalid_max_cap_df, max_cap_policy)

    # Test CapRes_ validation (must be between 0 and 1)
    invalid_cap_res_df = pd.DataFrame(
        {
            "Resource": ["gen_1", "gen_2"],
            "CapRes_1": [1.5, 0.5],  # 1.5 is invalid
            "CapRes_2": [0.3, 0.7],
        }
    )
    cap_res_policy = {"oldtag": "CapRes_", "newtag": "Derating_factor_"}
    with pytest.raises(ValueError):
        create_policy_df(invalid_cap_res_df, cap_res_policy)


def test_create_policy_df_inplace_modification():
    """Test that original DataFrame is modified correctly."""
    input_df = pd.DataFrame(
        {
            "Resource": ["gen_1", "gen_2"],
            "ESR_1": [1, 0],
            "ESR_2": [0, 1],
            "Other_Col": [1, 2],
        }
    )

    policy_info = {"oldtag": "ESR_", "newtag": "ESR_"}
    _ = create_policy_df(input_df, policy_info)

    assert not any(
        col.startswith("ESR_") for col in input_df.columns
    ), "Policy columns should be removed from original DataFrame"
    assert (
        "Other_Col" in input_df.columns
    ), "Non-policy columns should remain in original DataFrame"
    assert (
        "Resource" in input_df.columns
    ), "Resource column should remain in original DataFrame"


def test_create_policy_df_column_values():
    """Test that policy values are preserved correctly."""
    # Mock data
    input_df = pd.DataFrame(
        {
            "Resource": ["gen_1", "gen_2", "gen_3"],
            "ESR_1": [1, 0, 1],
            "ESR_2": [0, 1, 1],
            "Other_Col": [1, 2, 3],
        }
    )

    policy_info = {"oldtag": "ESR_", "newtag": "New_ESR_"}
    result = create_policy_df(input_df.copy(), policy_info)

    assert (
        result["New_ESR_1"] == [1, 0, 1]
    ).all(), "Values should be preserved after renaming"
    assert (
        result["New_ESR_2"] == [0, 1, 1]
    ).all(), "Values should be preserved after filtering zeros"
    assert "Other_Col" not in result.columns, "Non-policy columns should be removed"


def test_create_multistage_df_basic():
    """Test basic functionality of create_multistage_df."""
    # Mock data
    input_df = pd.DataFrame(
        {
            "Resource": ["gen_1", "gen_2"],
            "WACC": [100, 200],
            "Capital_Recovery_Period": [150, 250],
            "Lifetime": [1, 2],
            "Min_Retired_Cap_MW": [1, 2],
            "Min_Retired_Energy_Cap_MW": [1, 2],
            "Min_Retired_Charge_Cap_MW": [1, 2],
            "Other_Col": [1, 2],
        }
    )

    multistage_cols = [
        "WACC",
        "Capital_Recovery_Period",
        "Lifetime",
        "Min_Retired_Cap_MW",
        "Min_Retired_Energy_Cap_MW",
        "Min_Retired_Charge_Cap_MW",
    ]

    # Get result and verify
    result = create_multistage_df(input_df.copy(), multistage_cols)

    assert list(result.columns) == [
        "Resource",
        "WACC",
        "Capital_Recovery_Period",
        "Lifetime",
        "Min_Retired_Cap_MW",
        "Min_Retired_Energy_Cap_MW",
        "Min_Retired_Charge_Cap_MW",
    ], "Result should contain Resource and multistage columns"
    assert len(result) == 2, "Should preserve all rows"
    assert (result["WACC"] == [100, 200]).all(), "Should preserve WACC values"
    assert (
        result["Capital_Recovery_Period"] == [150, 250]
    ).all(), "Should preserve Capital_Recovery_Period values"
    assert (result["Lifetime"] == [1, 2]).all(), "Should preserve Lifetime values"
    assert (
        result["Min_Retired_Cap_MW"] == [1, 2]
    ).all(), "Should preserve Min_Retired_Cap_MW values"
    assert (
        result["Min_Retired_Energy_Cap_MW"] == [1, 2]
    ).all(), "Should preserve Min_Retired_Energy_Cap_MW values"
    assert (
        result["Min_Retired_Charge_Cap_MW"] == [1, 2]
    ).all(), "Should preserve Min_Retired_Charge_Cap_MW values"


def test_create_multistage_df_missing_cols():
    """Test behavior when requested columns don't exist."""
    input_df = pd.DataFrame(
        {
            "Resource": ["gen_1", "gen_2"],
            "WACC": [100, 200],
            "Capital_Recovery_Period": [150, 250],
            "Lifetime": [1, 2],
            "Other_Col": [1, 2],
        }
    )

    multistage_cols = [
        "WACC",
        "Capital_Recovery_Period",
        "Lifetime",
        "Min_Retired_Cap_MW",
        "Min_Retired_Energy_Cap_MW",
        "Min_Retired_Charge_Cap_MW",
    ]  # Stage_2 and Stage_3 don't exist

    result = create_multistage_df(input_df.copy(), multistage_cols)

    assert list(result.columns) == [
        "Resource",
        "WACC",
        "Capital_Recovery_Period",
        "Lifetime",
    ], "Should only include existing columns"
    assert len(result.columns) == 4, "Should not include non-existent columns"


def test_create_multistage_df_inplace_modification():
    """Test that original DataFrame is modified correctly."""
    input_df = pd.DataFrame(
        {
            "Resource": ["gen_1", "gen_2"],
            "WACC": [100, 200],
            "Capital_Recovery_Period": [150, 250],
            "Lifetime": [1, 2],
            "Min_Retired_Cap_MW": [1, 2],
            "Min_Retired_Energy_Cap_MW": [1, 2],
            "Min_Retired_Charge_Cap_MW": [1, 2],
            "Other_Col": [1, 2],
        }
    )

    multistage_cols = [
        "WACC",
        "Capital_Recovery_Period",
        "Lifetime",
        "Min_Retired_Cap_MW",
        "Min_Retired_Energy_Cap_MW",
        "Min_Retired_Charge_Cap_MW",
    ]

    _ = create_multistage_df(input_df, multistage_cols)

    assert list(input_df.columns) == [
        "Resource",
        "Other_Col",
    ], "Should remove multistage columns from original DataFrame"
    assert "WACC" not in input_df.columns, "WACC should be removed"
    assert (
        "Capital_Recovery_Period" not in input_df.columns
    ), "Capital_Recovery_Period should be removed"
    assert "Lifetime" not in input_df.columns, "Lifetime should be removed"
    assert (
        "Min_Retired_Cap_MW" not in input_df.columns
    ), "Min_Retired_Cap_MW should be removed"
    assert (
        "Min_Retired_Energy_Cap_MW" not in input_df.columns
    ), "Min_Retired_Energy_Cap_MW should be removed"
    assert (
        "Min_Retired_Charge_Cap_MW" not in input_df.columns
    ), "Min_Retired_Charge_Cap_MW should be removed"


def test_create_multistage_df_empty_cols():
    """Test behavior with empty multistage columns list."""
    input_df = pd.DataFrame(
        {
            "Resource": ["gen_1", "gen_2"],
            "WACC": [100, 200],
            "Capital_Recovery_Period": [150, 250],
            "Lifetime": [1, 2],
            "Min_Retired_Cap_MW": [1, 2],
            "Min_Retired_Energy_Cap_MW": [1, 2],
            "Min_Retired_Charge_Cap_MW": [1, 2],
            "Other_Col": [1, 2],
        }
    )

    result = create_multistage_df(input_df.copy(), [])

    assert list(result.columns) == [
        "Resource"
    ], "Should only contain Resource column when no multistage columns specified"
    assert len(result) == 2, "Should preserve all rows"


def test_create_multistage_df_data_types():
    """Test preservation of data types."""
    input_df = pd.DataFrame(
        {
            "Resource": ["gen_1", "gen_2"],
            "WACC": [100.0, 200.0],  # float
            "Capital_Recovery_Period": [1, 2],  # int
            "Lifetime": [10, 20],  # int
        }
    )

    multistage_cols = ["WACC", "Capital_Recovery_Period", "Lifetime"]

    result = create_multistage_df(input_df.copy(), multistage_cols)

    assert result["WACC"].dtype == float, "Should preserve float dtype"
    assert result["Capital_Recovery_Period"].dtype == int, "Should preserve int dtype"
    assert result["Lifetime"].dtype == int, "Should preserve int dtype"


def test_create_multistage_df_empty_df():
    """Test behavior with empty DataFrame."""
    input_df = pd.DataFrame(
        columns=["Resource", "WACC", "Capital_Recovery_Period", "Lifetime"]
    )
    multistage_cols = ["WACC", "Capital_Recovery_Period", "Lifetime"]

    result = create_multistage_df(input_df.copy(), multistage_cols)

    assert result.empty, "Should return empty DataFrame"
    assert list(result.columns) == [
        "Resource",
        "WACC",
        "Capital_Recovery_Period",
        "Lifetime",
    ], "Should preserve column structure even with empty DataFrame"


@pytest.fixture
def folder_structure():
    """Create a test folder structure."""
    return FolderStructure(
        resource=Path("test_resources"),
        policy_assignments=Path("test_policies"),
        system=Path("test_system"),
        policy=Path("test_policy"),
    )


def test_split_generators_basic(folder_structure, sample_gen_data):
    """Test basic functionality of split_generators_data."""
    result = split_generators_data(folder_structure, sample_gen_data)

    # Check that we get a list of GenXInputData objects
    assert isinstance(result, list)
    assert all(isinstance(x, GenXInputData) for x in result)

    # Check that R_ID is dropped
    assert all("R_ID" not in data.dataframe.columns for data in result)

    # Verify we have the expected number of output files
    # (THERM, STOR, VRE, MUST_RUN, FLEX, HYDRO, ESR policy, CapRes policy, and multistage data)
    resource_files = [
        data for data in result if data.folder == folder_structure.resource
    ]
    policy_files = [
        data for data in result if data.folder == folder_structure.policy_assignments
    ]

    assert len(resource_files) > 0, "Should have resource files"
    assert len(policy_files) > 0, "Should have policy files"


def test_split_generators_resource_types(folder_structure, sample_gen_data):
    """Test that each resource type is properly separated."""
    result = split_generators_data(folder_structure, sample_gen_data)

    # Check THERM resources
    therm_data = next(data for data in result if data.tag == "THERM")
    assert len(therm_data.dataframe) == 2, "Should have 2 THERM resources"
    assert "Model" in therm_data.dataframe.columns, "THERM should have Model column"

    # Check STOR resources
    stor_data = next(data for data in result if data.tag == "STOR")
    assert len(stor_data.dataframe) == 2, "Should have 2 STOR resources"
    assert "Model" in stor_data.dataframe.columns, "STOR should have Model column"

    # Check VRE resources
    vre_data = next(data for data in result if data.tag == "VRE")
    assert len(vre_data.dataframe) == 2, "Should have 2 VRE resources"
    assert "Model" not in vre_data.dataframe.columns, "VRE should not have Model column"
    assert (
        "Num_VRE_Bins" in vre_data.dataframe.columns
    ), "VRE should have Num_VRE_Bins column"

    # Check MUST_RUN resources
    must_run_data = next(data for data in result if data.tag == "MUST_RUN")
    assert len(must_run_data.dataframe) == 1, "Should have 1 MUST_RUN resources"
    assert (
        "Model" not in must_run_data.dataframe.columns
    ), "MUST_RUN should not have Model column"

    # Check FLEX resources
    flex_data = next(data for data in result if data.tag == "FLEX")
    assert len(flex_data.dataframe) == 1, "Should have 1 FLEX resources"
    assert (
        "Model" not in flex_data.dataframe.columns
    ), "FLEX should not have Model column"
    assert (
        "Num_VRE_Bins" not in flex_data.dataframe.columns
    ), "FLEX should not have Num_VRE_Bins column"
    assert (
        "Max_Flexible_Demand_Advance" in flex_data.dataframe.columns
    ), "FLEX should have Max_Flexible_Demand_Advance column"
    assert (
        "Max_Flexible_Demand_Delay" in flex_data.dataframe.columns
    ), "FLEX should have Max_Flexible_Demand_Delay column"

    # Check HYDRO resources
    hydro_data = next(data for data in result if data.tag == "HYDRO")
    assert len(hydro_data.dataframe) == 1, "Should have 1 HYDRO resources"
    assert (
        "Model" not in hydro_data.dataframe.columns
    ), "HYDRO should not have Model column"
    assert (
        "Num_VRE_Bins" not in hydro_data.dataframe.columns
    ), "HYDRO should not have Num_VRE_Bins column"
    assert (
        "Eff_Down" not in hydro_data.dataframe.columns
    ), "HYDRO should not have Eff_Down column as the column has only zeros"
    assert "Eff_Up" in hydro_data.dataframe.columns, "HYDRO should have Eff_Up column"
    assert (
        "Hydro_Energy_to_Power_Ratio" in hydro_data.dataframe.columns
    ), "HYDRO should have Hydro_Energy_to_Power_Ratio column"
    assert "LDS" in hydro_data.dataframe.columns, "HYDRO should have LDS column"
    assert (
        "Min_Power" in hydro_data.dataframe.columns
    ), "HYDRO should have Min_Power column"
    assert (
        "Ramp_Up_Percentage" in hydro_data.dataframe.columns
    ), "HYDRO should have Ramp_Up_Percentage column"
    assert (
        "Ramp_Dn_Percentage" in hydro_data.dataframe.columns
    ), "HYDRO should have Ramp_Dn_Percentage column"


def test_split_generators_policy_data(folder_structure, sample_gen_data):
    """Test that policy data is properly processed."""
    result = split_generators_data(folder_structure, sample_gen_data)

    # Check ESR policy data
    esr_data = next(data for data in result if data.tag == "RES_ESR")
    assert all(col in esr_data.dataframe.columns for col in ["ESR_1", "ESR_2"])
    assert len(esr_data.dataframe) > 0, "Should have ESR policy data"

    # Check Derating_factor policy data
    cap_res_data = next(data for data in result if data.tag == "RES_CAP_RES")
    assert "Derating_factor_1" in cap_res_data.dataframe.columns
    assert len(cap_res_data.dataframe) > 0, "Should have Derating_factor policy data"

    # Check Min_Cap policy data
    min_cap_data = next(data for data in result if data.tag == "RES_MIN_CAP")
    assert "Min_Cap_1" in min_cap_data.dataframe.columns
    assert len(min_cap_data.dataframe) > 0, "Should have Min_Cap policy data"

    # Check Max_Cap policy data
    max_cap_data = next(data for data in result if data.tag == "RES_MAX_CAP")
    assert "Max_Cap_1" in max_cap_data.dataframe.columns
    assert len(max_cap_data.dataframe) > 0, "Should have Max_Cap policy data"


def test_split_generators_multistage(folder_structure, sample_gen_data):
    """Test that multistage data is properly processed."""
    result = split_generators_data(folder_structure, sample_gen_data)

    multistage_data = next(data for data in result if data.tag == "MULTISTAGE")
    assert all(
        col in multistage_data.dataframe.columns
        for col in ["WACC", "Capital_Recovery_Period", "Lifetime"]
    )
    assert (
        len(multistage_data.dataframe) == 9
    ), "Should have all rows in multistage data"


def test_split_generators_empty_data(folder_structure):
    """Test behavior with empty generator data."""
    empty_df = pd.DataFrame(columns=["Resource", "THERM", "ESR_1"])
    result = split_generators_data(folder_structure, empty_df)

    assert isinstance(result, list)
    assert len(result) == 0, "Should return empty list for empty input"


def test_split_generators_missing_columns(folder_structure, sample_gen_data):
    """Test behavior when optional columns are missing."""
    # Remove some columns
    df_subset = sample_gen_data.drop(columns=["New_Build", "R_ID"])
    result = split_generators_data(folder_structure, df_subset)

    assert isinstance(result, list)
    assert len(result) > 0, "Should still process data without optional columns"


def test_split_generators_data_integrity(folder_structure, sample_gen_data):
    """Test that data values are preserved correctly."""
    result = split_generators_data(folder_structure, sample_gen_data)

    # Check that original values are preserved in appropriate output files
    for data in result:
        if data.tag == "THERM":
            assert (
                data.dataframe["Resource"].isin(["therm_nc", "therm_uc"]).all()
            ), "THERM resources should match original data"
        elif data.tag == "STOR":
            assert (
                data.dataframe["Resource"].isin(["stor_sym", "stor_asym"]).all()
            ), "STOR resources should match original data"
        elif data.tag == "VRE":
            assert (
                data.dataframe["Resource"].isin(["vre_1", "vre_2"]).all()
            ), "VRE resources should match original data"
        elif data.tag == "MUST_RUN":
            assert (
                data.dataframe["Resource"].isin(["must_run_1"]).all()
            ), "MUST_RUN resources should match original data"
        elif data.tag == "FLEX":
            assert (
                data.dataframe["Resource"].isin(["flex_1"]).all()
            ), "FLEX resources should match original data"
        elif data.tag == "HYDRO":
            assert (
                data.dataframe["Resource"].isin(["hydro_1"]).all()
            ), "HYDRO resources should match original data"
        elif data.tag == "MULTISTAGE":
            assert (
                data.dataframe["WACC"]
                == [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
            ).all(), "Multistage data should preserve original values"


def test_split_generators_newbuild_canretire(folder_structure):
    """Test New_Build and Can_Retire column handling."""
    df = pd.DataFrame(
        {"Resource": ["gen_1", "gen_2"], "New_Build": [1, 0], "THERM": [2, 0]}
    )

    result = split_generators_data(folder_structure, df)

    # Find THERM resource data
    therm_data = next(data for data in result if data.tag == "THERM")
    assert (
        "Can_Retire" in therm_data.dataframe.columns
    ), "Should add Can_Retire column based on New_Build"


@pytest.fixture
def case_folder(tmp_path):
    """Create a temporary case folder structure."""
    return tmp_path / "test_case"


@pytest.fixture
def sample_genx_data_dict():
    """Create sample data dictionary with all required DataFrames."""
    return {
        "gen_data": pd.DataFrame(
            {
                "Resource": ["gen_1", "gen_2"],
                "THERM": [2, 1],
                "ESR_1": [1, 0],
                "CapRes_1": [0.8, 0.9],
                "MinCapTag_1": [1, 0],
                "MaxCapTag_1": [0, 1],
                "Lifetime": [10, 20],
            }
        ),
        "gen_variability": pd.DataFrame({"Resource": ["gen_1"], "Var_1": [0.5]}),
        "demand_data": pd.DataFrame({"Time": range(24), "Load": [100] * 24}),
        "period_map": pd.DataFrame({"Period": range(12), "Map": ["Jan"] * 12}),
        "rep_period": pd.DataFrame({"Period": range(12), "Weight": [1] * 12}),
        "network": pd.DataFrame({"Line": ["L1"], "From": ["A"], "To": ["B"]}),
        "op_reserves": pd.DataFrame({"Reserve": ["R1"], "Requirement": [0.1]}),
        "fuels": pd.DataFrame({"Fuel": ["Coal"], "Price": [50]}),
        "esr": pd.DataFrame({"Year": [2030], "Target": [0.5]}),
        "cap_reserves": pd.DataFrame({"Period": [1], "Margin": [0.15]}),
        "co2_cap": pd.DataFrame({"Year": [2030], "Cap": [1000]}),
        "min_cap": pd.DataFrame({"Resource": ["gen_1"], "Min": [100]}),
        "max_cap": pd.DataFrame({"Resource": ["gen_1"], "Max": [500]}),
    }


def test_process_genx_data_basic(case_folder, sample_genx_data_dict):
    """Test basic functionality of process_genx_data."""
    result = process_genx_data(case_folder, sample_genx_data_dict)

    # Check that we get a list of GenXInputData objects
    assert isinstance(result, list)
    assert all(isinstance(x, GenXInputData) for x in result)

    # Check that all expected files are present
    expected_tags = {
        "THERM",
        "RES_ESR",
        "RES_CAP_RES",
        "RES_MIN_CAP",
        "RES_MAX_CAP",
        "MULTISTAGE",
        "GENERATORS_VARIABILITY",
        "DEMAND",
        "PERIOD_MAP",
        "REP_PERIOD",
        "NETWORK",
        "OP_RESERVES",
        "FUELS",
        "ESR",
        "CAP_RESERVES",
        "CO2_CAP",
        "MIN_CAP",
        "MAX_CAP",
    }
    result_tags = {data.tag for data in result}
    assert expected_tags == result_tags, "All expected data types should be present"


def test_process_genx_data_folder_structure(case_folder, sample_genx_data_dict):
    """Test that files are assigned to correct folders."""
    result = process_genx_data(case_folder, sample_genx_data_dict)

    # Check system folder assignments
    system_files = [data for data in result if data.folder == case_folder / "system"]
    system_tags = {
        "GENERATORS_VARIABILITY",
        "DEMAND",
        "PERIOD_MAP",
        "REP_PERIOD",
        "NETWORK",
        "OP_RESERVES",
        "FUELS",
    }
    assert {data.tag for data in system_files} == system_tags

    # Check policy folder assignments
    policy_files = [data for data in result if data.folder == case_folder / "policy"]
    policy_tags = {"ESR", "CAP_RESERVES", "CO2_CAP", "MIN_CAP", "MAX_CAP"}
    assert {data.tag for data in policy_files} == policy_tags


def test_process_genx_data_file_names(case_folder, sample_genx_data_dict):
    """Test that correct file names are assigned."""
    result = process_genx_data(case_folder, sample_genx_data_dict)

    expected_files = {
        "THERM": "Thermal.csv",
        "RES_ESR": "Resource_energy_share_requirement.csv",
        "RES_CAP_RES": "Resource_capacity_reserve_margin.csv",
        "RES_MIN_CAP": "Resource_minimum_capacity_requirement.csv",
        "RES_MAX_CAP": "Resource_maximum_capacity_requirement.csv",
        "MULTISTAGE": "Resource_multistage_data.csv",
        "GENERATORS_VARIABILITY": "Generators_variability.csv",
        "DEMAND": "Demand_data.csv",
        "PERIOD_MAP": "Period_map.csv",
        "REP_PERIOD": "Representative_Period.csv",
        "NETWORK": "Network.csv",
        "OP_RESERVES": "Operational_reserves.csv",
        "FUELS": "Fuels_data.csv",
        "ESR": "Energy_share_requirement.csv",
        "CAP_RESERVES": "Capacity_reserve_margin.csv",
        "CO2_CAP": "CO2_cap.csv",
        "MIN_CAP": "Minimum_capacity_requirement.csv",
        "MAX_CAP": "Maximum_capacity_requirement.csv",
    }

    for data in result:
        if data.tag in expected_files:
            assert (
                data.file_name == expected_files[data.tag]
            ), f"File name mismatch for {data.tag}"


def test_process_genx_data_empty_dataframes(case_folder, sample_genx_data_dict):
    """Test handling of empty DataFrames."""
    # Modify some DataFrames to be empty
    sample_genx_data_dict["demand_data"] = pd.DataFrame()
    sample_genx_data_dict["network"] = pd.DataFrame()

    result = process_genx_data(case_folder, sample_genx_data_dict)

    # Check that empty DataFrames are still included
    tags = {data.tag for data in result}
    assert "DEMAND" in tags
    assert "NETWORK" in tags


def test_process_genx_data_missing_keys(case_folder, sample_genx_data_dict):
    """Test handling of missing dictionary keys."""
    # Remove some keys from the dictionary
    del sample_genx_data_dict["network"]
    del sample_genx_data_dict["op_reserves"]

    with pytest.raises(KeyError):
        process_genx_data(case_folder, sample_genx_data_dict)


def test_process_genx_data_generators_split(case_folder, sample_genx_data_dict):
    """Test that generator data is properly split."""
    result = process_genx_data(case_folder, sample_genx_data_dict)

    # Find generator-related outputs
    resource_folder = case_folder / "resources"
    gen_files = [data for data in result if data.folder == resource_folder]

    assert any(
        data.tag == "THERM" for data in gen_files
    ), "Should have THERM resources from generator split"
    assert any(
        data.tag == "MULTISTAGE" for data in gen_files
    ), "Should have multistage data from generator split"


def test_process_genx_data_dataframe_integrity(case_folder, sample_genx_data_dict):
    """Test that DataFrame contents are preserved."""
    result = process_genx_data(case_folder, sample_genx_data_dict)

    for data in result:
        if data.tag == "DEMAND":
            assert len(data.dataframe) == 24, "Demand data should have 24 rows"
        elif data.tag == "FUELS":
            assert (
                data.dataframe["Fuel"].iloc[0] == "Coal"
            ), "Fuel data should preserve values"
        elif data.tag == "ESR":
            assert (
                data.dataframe["Target"].iloc[0] == 0.5
            ), "ESR data should preserve values"
