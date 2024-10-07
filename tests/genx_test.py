import logging

import pandas as pd
import pytest

from powergenome.GenX import check_vre_profiles, set_must_run_generation


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
