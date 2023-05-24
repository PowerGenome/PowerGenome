import pandas as pd
import pytest

from powergenome.GenX import set_must_run_generation


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
    from pandas.testing import assert_frame_equal
    import logging

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
