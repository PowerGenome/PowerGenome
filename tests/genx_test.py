import logging

import numpy as np
import pandas as pd
import pytest

from powergenome.GenX import (
    check_vre_profiles,
    set_must_run_generation,
    update_newbuild_canretire,
    filter_empty_columns,
    create_resource_df,
    RESOURCE_COLUMNS,
    DEFAULT_COLS,


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
    input_df = pd.DataFrame({
        'New_Build': [1, -1, 0, 1],
        'Other_Column': [1, 2, 3, 4]
    })
    expected_df = pd.DataFrame({
        'New_Build': [1, 0, 0, 1],
        'Other_Column': [1, 2, 3, 4],
        'Can_Retire': [1, 0, 1, 1],
    })
    result_df = update_newbuild_canretire(input_df)
    pd.testing.assert_frame_equal(result_df, expected_df)

    # Test case 2: Missing New_Build column
    input_df = pd.DataFrame({
        'Other_Column': [1, 2, 3]
    })
    result_df = update_newbuild_canretire(input_df)
    pd.testing.assert_frame_equal(result_df, input_df)

    # Test case 3: Empty DataFrame
    input_df = pd.DataFrame({'New_Build': pd.Series([], dtype=int), 'Other_Column': pd.Series([], dtype=int)})
    expected_df = pd.DataFrame({
        'New_Build': pd.Series([], dtype=int),
        'Other_Column': pd.Series([], dtype=int),
        'Can_Retire': pd.Series([], dtype=int),
    })
    result_df = update_newbuild_canretire(input_df)
    pd.testing.assert_frame_equal(result_df, expected_df)

    # Test case 4: DataFrame with NaN values
    input_df = pd.DataFrame({
        'New_Build': [1, np.nan, -1, 0],
        'Other_Column': [1, 2, 3, 4]
    })
    expected_df = pd.DataFrame({
        'New_Build': [1, np.nan, -1, 0],
        'Other_Column': [1, 2, 3, 4]
    })
    result_df = update_newbuild_canretire(input_df)
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_filter_empty_columns():
    """Test filter_empty_columns function with various test cases."""
    # Test case 1: Default behaviour
    df_basic = pd.DataFrame({
        'valid_col': [1, 2, 3],
        'zero_col': [0, 0, 0],
        'none_col': [None, None, None],
        'string_none_col': ["None", "None", "None"],
        'mixed_col': [1, None, "None"],
        'valid_string': ["a", "b", "c"]
    })
    result = filter_empty_columns(df_basic)
    assert sorted(result) == sorted(['valid_col', 'mixed_col', 'valid_string']), \
        "Basic case failed"

    # Test case 2: Empty DataFrame
    df_empty = pd.DataFrame({})
    assert filter_empty_columns(df_empty) == [], \
        "Empty DataFrame case failed"

    # Test case 3: All invalid columns
    df_invalid = pd.DataFrame({
        'zero_col': [0, 0],
        'none_col': [None, None],
        'string_none_col': ["None", "None"]
    })
    assert filter_empty_columns(df_invalid) == [], \
        "All invalid columns case failed"

    # Test case 4: Mixed types
    df_mixed = pd.DataFrame({
        'mixed_types': [1, "text", 3.14],
        'mixed_invalid': [0, None, "None"], # For now, mixed types are allowed
        'bool_col': [True, False, True]
    })
    assert sorted(filter_empty_columns(df_mixed)) == sorted(['mixed_types', 'mixed_invalid', 'bool_col']), \
        "Mixed types case failed"

    # Test case 5: Edge cases
    df_edge = pd.DataFrame({
        'empty_str': ["", "", ""],
        'whitespace': [" ", "  ", "\t"],
        'zero_float': [0.0, 0.0, 0.0],
        'valid_zero_one': [0, 1, 0]
    })
    assert sorted(filter_empty_columns(df_edge)) == sorted(['whitespace', 'valid_zero_one']), \
        "Edge cases failed"

    # Test case 6: NaN values
    df_nan = pd.DataFrame({
        'nan_col': [np.nan, np.nan, np.nan],
        'mixed_nan': [1, np.nan, 3],
        'inf_col': [np.inf, -np.inf, np.inf]
    })
    assert sorted(filter_empty_columns(df_nan)) == sorted(['mixed_nan', 'inf_col']), \
        "NaN values case failed"

def test_filter_empty_columns_errors():
    """Test error handling in filter_empty_columns."""
    # Test with non-DataFrame input
    with pytest.raises(AttributeError):
        filter_empty_columns([1, 2, 3])
        
@pytest.fixture
def sample_gen_data():
    """Create sample generator data for testing."""
    return pd.DataFrame({
        # Default columns
        'Resource': ['therm_nc', 'stor_sym', 'therm_uc', 'stor_asym', 'must_run_1', 'flex_1', 'vre_1', 'hydro_1', 'vre_2'],
        'Zone':       [1, 2, 1, 3, 4, 5, 6, 2, 2],
        'New_Build':  [1, 0, 1, 0, 0, 1, 1, 1, 1],
        'Can_Retire': [1, 1, 1, 1, 0, 1, 0, 1, 1],
        
        # Resource tag columns
        'THERM':    [1, 0, 2, 0, 0, 0, 0, 0, 0],
        'STOR':     [0, 1, 0, 2, 0, 0, 0, 0, 0],
        'VRE':      [0, 0, 0, 0, 0, 0, 1, 0, 1],
        'MUST_RUN': [0, 0, 0, 0, 1, 0, 0, 0, 0],
        'FLEX':     [0, 0, 0, 0, 0, 1, 0, 0, 0],
        'HYDRO':    [0, 0, 0, 0, 0, 0, 0, 1, 0],
        
        # Resource specific columns
        'Min_Power':    [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], # THERM column
        'Down_Time':    [6.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # THERM column
        'Up_Time':      [6.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # THERM column
        'Eff_Down':     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # HYDRO column
        'Eff_Up':       [0.0, 0.8, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0, 0.0], # STOR column
        'Max_Cap_MWh':  [0.0, 200, 0.0, 100, 0.0, 0.0, 0.0, 0.0, 0.0], # STOR column
        'LDS':          [0, 1, 0, 1, 0, 0, 0, 1, 0], # HYDRO and STOR column
        'Num_VRE_Bins': [0, 0, 0, 0, 0, 0, 1, 0, 1], # VRE column
        'Ramp_Up_Percentage': [0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0], # THERM column
        'Ramp_Dn_Percentage': [0.3, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0], # THERM column
        'Flexible_Demand_Energy_Eff':  [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0], # FLEX column
        'Max_Flexible_Demand_Advance': [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0], # FLEX column
        'Max_Flexible_Demand_Delay':   [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0], # FLEX column
        'Var_OM_Cost_per_MWh_In':      [0.0, 0.0, 0.0, 0.0, 0.0, 100, 0.0, 0.0, 0.0], # FLEX column
        'Hydro_Energy_to_Power_Ratio': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0], # HYDRO column
        
        # Policy columns
        'ESR_1': [1, 0, 1, 0, 0, 1, 0, 0, 0],
        'ESR_2': [0, 1, 1, 0, 0, 0, 1, 0, 0],
        'CapRes_1': [0.5, 0.0, 0.7, 0.1, 0, 0.5, 0, 0, 0],
        'CapRes_2': [0.0, 0.5, 0.0, 0.3, 0, 0.0, 0, 0, 0],
        'MinCapTag_1': [1, 0, 1, 0, 0, 0, 0, 0, 0],
        'MinCapTag_2': [0, 1, 1, 0, 0, 1, 0, 0, 0],
        'MaxCapTag_1': [1, 0, 1, 0, 0, 0, 1, 0, 1],
        'MaxCapTag_2': [0, 1, 1, 0, 0, 1, 0, 0, 1],
        
        # Multi-stage columns
        'WACC': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        'Capital_Recovery_Period': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'Lifetime': [10, 10, 20, 20, 30, 30, 40, 40, 50],
        
        # R_ID
        'R_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        
        # Other columns
        'Extra_Col': [1, 2, 3, 4, 5, 6, 7, 8, 9]
    })
    
def test_create_resource_df(sample_gen_data):
    """Test create_resource_df function with various test cases."""
    
    input_df = sample_gen_data

    # Test case 1: THERM resource
    therm_result = create_resource_df(input_df, 'THERM')
    assert list(therm_result.columns[:4]) == DEFAULT_COLS, "Default columns should be first"
    assert 'Model' in therm_result.columns, "THERM should be renamed to Model"
    assert therm_result["Model"].tolist() == [1, 2] # Check that the Model column is correct
    assert 'Eff_Up' not in therm_result.columns, "STOR columns should be removed"
    assert len(therm_result) == 2, "Should only include rows where THERM > 0"
    assert all(col in therm_result.columns for col in RESOURCE_COLUMNS['THERM']), \
        "All THERM specific columns should be present"

    # Test case 2: STOR resource
    stor_result = create_resource_df(input_df, 'STOR')
    assert list(stor_result.columns[:4]) == DEFAULT_COLS, "Default columns should be first"
    assert 'Model' in stor_result.columns, "STOR should be renamed to Model"
    assert stor_result["Model"].tolist() == [1, 2] # Check that the Model column is correct
    assert 'Min_Power' not in stor_result.columns, "THERM columns should be removed"
    assert stor_result["LDS"].tolist() == [1, 1] # Check that the LDS column is correct
    assert len(stor_result) == 2, "Should only include rows where STOR > 0"
        
    # Test case 3: VRE resource
    vre_result = create_resource_df(input_df, 'VRE')
    assert list(vre_result.columns[:4]) == DEFAULT_COLS, "Default columns should be first"
    assert 'Model' not in vre_result.columns, "VRE should not be renamed to Model"
    assert 'Min_Power' not in vre_result.columns, "THERM columns should be removed"
    assert len(vre_result) == 2, "Should only include rows where VRE=1"
    assert all(col in vre_result.columns for col in RESOURCE_COLUMNS['VRE']), \
        "All VRE specific columns should be present"
        
    # Test case 4: MUST_RUN resource
    must_run_result = create_resource_df(input_df, 'MUST_RUN')
    assert list(must_run_result.columns[:4]) == DEFAULT_COLS, "Default columns should be first"
    assert 'Model' not in must_run_result.columns, "MUST_RUN should not be renamed to Model"
    assert 'Min_Power' not in must_run_result.columns, "THERM columns should be removed"
    assert len(must_run_result) == 1, "Should only include rows where MUST_RUN=1"
    assert all(col in must_run_result.columns for col in RESOURCE_COLUMNS['MUST_RUN']), \
        "All MUST_RUN specific columns should be present"
        
    # Test case 5: FLEX resource
    flex_result = create_resource_df(input_df, 'FLEX')
    assert list(flex_result.columns[:4]) == DEFAULT_COLS, "Default columns should be first"
    assert 'Model' not in flex_result.columns, "FLEX should not be renamed to Model"
    assert 'Min_Power' not in flex_result.columns, "THERM columns should be removed"
    assert len(flex_result) == 1, "Should only include rows where FLEX=1"
    assert all(col in flex_result.columns for col in RESOURCE_COLUMNS['FLEX']), \
        "All FLEX specific columns should be present"
        
    # Test case 6: HYDRO resource
    hydro_result = create_resource_df(input_df, 'HYDRO')    
    assert list(hydro_result.columns[:4]) == DEFAULT_COLS, "Default columns should be first"
    assert 'Model' not in hydro_result.columns, "HYDRO should not be renamed to Model"
    assert 'Min_Power' in hydro_result.columns, "THERM columns should be removed"
    assert len(hydro_result) == 1, "Should only include rows where HYDRO=1"
    assert hydro_result["LDS"].tolist() == [1] # Check that the LDS column is correct
           
    # Test case 7: Invalid resource tag
    invalid_result = create_resource_df(input_df, 'INVALID')
    assert invalid_result.empty, "Should return empty DataFrame for invalid resource tag"

    # Test case 8: Empty DataFrame
    empty_result = create_resource_df(pd.DataFrame(), 'THERM')
    assert empty_result.empty, "Should return empty DataFrame for empty input"

    # Test case 9: Missing required columns
    incomplete_df = pd.DataFrame({
        'THERM': [1, 0, 1],
        'Extra_Col': [1, 2, 3]
    })
    incomplete_result = create_resource_df(incomplete_df, 'THERM')
    assert set(DEFAULT_COLS) - set(incomplete_result.columns), \
        "Should warn about missing required columns"

def test_create_resource_df_all_zero_columns():
    """Test handling of columns with all zero values."""
    input_df = pd.DataFrame({
        'Resource': ['therm_1', 'therm_2'],
        'Zone': [1, 2],
        'New_Build': [1, 0],
        'THERM': [1, 1],
        'Min_Power': [0, 0],  # All zeros
        'Ramp_Up_Percentage': [0.1, 0.2]
    })
    
    result = create_resource_df(input_df, 'THERM')
    assert 'Ramp_Up_Percentage' in result.columns, \
        "Should keep non-zero columns"
    assert 'Min_Power' not in result.columns, \
        "Should remove all-zero columns unless in DEFAULT_COLS"

def test_create_resource_df_none_values():
    """Test handling of None values."""
    input_df = pd.DataFrame({
        'Resource': ['therm_1', 'therm_2'],
        'Zone': [1, 2],
        'New_Build': [1, 0],
        'THERM': [1, 1],
        'Min_Power': [None, None],  # All None
        'Ramp_Up_Percentage': [0.1, None]  # Mixed None
    })
    
    result = create_resource_df(input_df, 'THERM')
    assert 'Ramp_Up_Percentage' in result.columns, \
        "Should keep columns with some valid values"
    assert 'Min_Power' not in result.columns, \
        "Should remove columns with all None values"

def test_create_resource_df_column_order():
    """Test that column order is correct."""
    input_df = pd.DataFrame({
        'Extra_First': [1, 2],
        'Resource': ['therm_1', 'therm_2'],
        'Zone': [1, 2],
        'New_Build': [1, 0],
        'Can_Retire': [1, 1],
        'THERM': [1, 1],
        'Extra_Last': [3, 4]
    })
    
    result = create_resource_df(input_df, 'THERM')
    assert list(result.columns)[:len(DEFAULT_COLS)] == \
        [col for col in DEFAULT_COLS if col in result.columns], \
        "Default columns should be first and in correct order" 
