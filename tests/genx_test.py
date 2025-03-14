import logging

import numpy as np
import pandas as pd
import pytest

from powergenome.GenX import (
    check_vre_profiles,
    set_must_run_generation,
    update_newbuild_canretire,
    filter_empty_columns,


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
