"""
Tests for transmission.py functions
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from powergenome.transmission import (
    _filter_and_map_regions,
    _format_transmission_output,
    _validate_transmission_data,
    agg_transmission_constraints,
    insert_tx_costs,
    load_tx_costs,
)


@pytest.fixture
def sample_transmission_data():
    """Sample transmission constraints data for testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "region_from": ["A", "B", "C", "A"],
            "region_to": ["B", "C", "A", "C"],
            "firm_ttc_mw": [100, 200, 150, 300],
            "nonfirm_ttc_mw": [120, 220, 170, 320],
        }
    )


@pytest.fixture
def sample_settings():
    """Sample settings dictionary for testing."""
    return {
        "model_regions": ["A", "B", "C"],
        "zone_num_map": {"A": 1, "B": 2, "C": 3},
        "tx_value_col": "firm_ttc_mw",
        "region_aggregations": {},
    }


class TestValidateTransmissionData:
    """Tests for _validate_transmission_data function."""

    def test_valid_data_passes(self, sample_transmission_data):
        """Test that valid data passes validation without error."""
        # Should not raise any exception
        _validate_transmission_data(
            sample_transmission_data, "test_table", "firm_ttc_mw"
        )

    def test_missing_column_raises_error(self, sample_transmission_data):
        """Test that missing transmission value column raises KeyError."""
        with pytest.raises(KeyError, match="There is no column missing_col"):
            _validate_transmission_data(
                sample_transmission_data,
                "test_table",
                "missing_col",
            )

    def test_duplicate_lines_raises_error(self):
        """Test that duplicate transmission lines raise KeyError."""
        duplicate_data = pd.DataFrame(
            {
                "region_from": ["A", "A", "B"],
                "region_to": ["B", "B", "C"],
                "firm_ttc_mw": [100, 150, 200],
            }
        )

        with pytest.raises(KeyError, match="duplicate lines"):
            _validate_transmission_data(
                duplicate_data,
                "test_table",
                "firm_ttc_mw",
            )


class TestFilterAndMapRegions:
    """Tests for _filter_and_map_regions function."""

    def test_basic_filtering_and_mapping(
        self, sample_transmission_data, sample_settings
    ):
        """Test basic region filtering and mapping without aggregation."""
        result = _filter_and_map_regions(
            sample_transmission_data,
            sample_settings["model_regions"],
            sample_settings["region_aggregations"],
            "firm_ttc_mw",
        )

        assert isinstance(result, pd.DataFrame)
        assert "firm_ttc_mw" in result.columns
        assert len(result) > 0

        # Check that index contains model region pairs
        for idx in result.index:
            assert len(idx) == 2  # Should be tuple of (from, to)

    def test_with_region_aggregations(self, sample_transmission_data):
        """Test filtering and mapping with region aggregations."""
        model_regions = ["AB", "C"]
        region_aggregations = {"AB": ["A", "B"]}

        result = _filter_and_map_regions(
            sample_transmission_data,
            model_regions,
            region_aggregations,
            "firm_ttc_mw",
        )

        assert isinstance(result, pd.DataFrame)
        # Should aggregate transmission between A and B into AB region
        assert len(result) <= len(sample_transmission_data)

    def test_drops_id_column(self, sample_transmission_data, sample_settings):
        """Test that 'id' column is dropped during processing."""
        result = _filter_and_map_regions(
            sample_transmission_data,
            sample_settings["model_regions"],
            sample_settings["region_aggregations"],
            "firm_ttc_mw",
        )

        assert "id" not in result.columns

    def test_filters_out_invalid_regions(self, sample_settings):
        """Test that regions not in model_regions are filtered out."""
        data_with_invalid = pd.DataFrame(
            {
                "region_from": ["A", "D", "B"],  # D is not in model_regions
                "region_to": ["B", "E", "C"],  # E is not in model_regions
                "firm_ttc_mw": [100, 200, 300],
            }
        )

        result = _filter_and_map_regions(
            data_with_invalid,
            sample_settings["model_regions"],
            sample_settings["region_aggregations"],
            "firm_ttc_mw",
        )

        # Should only have A-B and B-C connections
        assert len(result) == 2


class TestFormatTransmissionOutput:
    """Tests for _format_transmission_output function."""

    def test_basic_formatting(self, sample_settings):
        """Test basic output formatting."""
        # Create sample aggregated data
        input_df = pd.DataFrame(
            {"firm_ttc_mw": [100, 200, 150]},
            index=pd.MultiIndex.from_tuples(
                [("A", "B"), ("B", "C"), ("A", "C")],
                names=["model_region_from", "model_region_to"],
            ),
        )

        result = _format_transmission_output(
            input_df,
            sample_settings["model_regions"],
            sample_settings["zone_num_map"],
            "firm_ttc_mw",
        )

        # Check required columns exist
        expected_cols = [
            "Network_zones",
            "Network_Lines",
            "Start_Zone",
            "End_Zone",
            "Line_Max_Flow_MW",
            "start_region",
            "dest_region",
            "transmission_path_name",
        ]
        for col in expected_cols:
            assert col in result.columns

        # Check data types
        assert result["Network_Lines"].dtype == "Int64"
        assert result["Start_Zone"].dtype == "Int64"
        assert result["End_Zone"].dtype == "Int64"
        assert result["Line_Max_Flow_MW"].dtype == "Float64"

    def test_zone_mapping(self, sample_settings):
        """Test that zone numbers are correctly mapped."""
        input_df = pd.DataFrame(
            {"firm_ttc_mw": [100]},
            index=pd.MultiIndex.from_tuples(
                [("A", "B")], names=["model_region_from", "model_region_to"]
            ),
        )

        result = _format_transmission_output(
            input_df,
            sample_settings["model_regions"],
            sample_settings["zone_num_map"],
            "firm_ttc_mw",
        )

        assert result.iloc[0]["Start_Zone"] == 1  # A maps to 1
        assert result.iloc[0]["End_Zone"] == 2  # B maps to 2

    def test_transmission_path_names(self, sample_settings):
        """Test transmission path name generation."""
        input_df = pd.DataFrame(
            {"firm_ttc_mw": [100]},
            index=pd.MultiIndex.from_tuples(
                [("A", "B")], names=["model_region_from", "model_region_to"]
            ),
        )

        result = _format_transmission_output(
            input_df,
            sample_settings["model_regions"],
            sample_settings["zone_num_map"],
            "firm_ttc_mw",
        )

        assert result.iloc[0]["transmission_path_name"] == "A_to_B"

    def test_network_lines_sequential(self, sample_settings):
        """Test that network line numbers are sequential starting from 1."""
        input_df = pd.DataFrame(
            {"firm_ttc_mw": [100, 200, 150]},
            index=pd.MultiIndex.from_tuples(
                [("A", "B"), ("B", "C"), ("A", "C")],
                names=["model_region_from", "model_region_to"],
            ),
        )

        result = _format_transmission_output(
            input_df,
            sample_settings["model_regions"],
            sample_settings["zone_num_map"],
            "firm_ttc_mw",
        )

        expected_line_nums = list(range(1, len(result) + 1))
        actual_line_nums = sorted(result["Network_Lines"].tolist())
        assert actual_line_nums == expected_line_nums


class TestAggTransmissionConstraints:
    """Tests for the main agg_transmission_constraints function."""

    @patch("powergenome.transmission.load_data")
    def test_default_tx_value_col(
        self, mock_load_data, sample_transmission_data, sample_settings
    ):
        """Test that default tx_value_col is used when not specified."""
        mock_load_data.return_value = sample_transmission_data

        with patch("powergenome.transmission.logger") as mock_logger:
            result = agg_transmission_constraints(
                "fake_path",
                model_regions=sample_settings["model_regions"],
                zone_num_map=sample_settings["zone_num_map"],
                tx_value_col="",  # Empty string to trigger default
            )

            # Should warn about using default column
            mock_logger.warning.assert_called_once()
            assert "firm_ttc_mw" in mock_logger.warning.call_args[0][0]

    @patch("powergenome.transmission.load_data")
    def test_successful_execution(
        self, mock_load_data, sample_transmission_data, sample_settings
    ):
        """Test successful execution of the main function."""
        mock_load_data.return_value = sample_transmission_data

        result = agg_transmission_constraints(
            "fake_path",
            model_regions=sample_settings["model_regions"],
            zone_num_map=sample_settings["zone_num_map"],
            tx_value_col=sample_settings["tx_value_col"],
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

        # Check that all expected columns are present
        expected_cols = [
            "Network_zones",
            "Network_Lines",
            "Start_Zone",
            "End_Zone",
            "Line_Max_Flow_MW",
            "start_region",
            "dest_region",
            "transmission_path_name",
        ]
        for col in expected_cols:
            assert col in result.columns

    @patch("powergenome.transmission.load_data")
    def test_validation_error_propagated(self, mock_load_data, sample_settings):
        """Test that validation errors are properly propagated."""
        # Create data with missing column
        bad_data = pd.DataFrame(
            {
                "region_from": ["A"],
                "region_to": ["B"],
                "wrong_col": [100],
            }
        )
        mock_load_data.return_value = bad_data

        with pytest.raises(KeyError, match="There is no column firm_ttc_mw"):
            agg_transmission_constraints(
                "fake_path",
                model_regions=sample_settings["model_regions"],
                zone_num_map=sample_settings["zone_num_map"],
                tx_value_col="firm_ttc_mw",
            )

    @patch("powergenome.transmission.load_data")
    def test_custom_data_table(
        self, mock_load_data, sample_transmission_data, sample_settings
    ):
        """Test using custom data table name."""
        mock_load_data.return_value = sample_transmission_data

        agg_transmission_constraints(
            "fake_path",
            data_table="custom_table",
            model_regions=sample_settings["model_regions"],
            zone_num_map=sample_settings["zone_num_map"],
            tx_value_col=sample_settings["tx_value_col"],
        )

        mock_load_data.assert_called_once_with("fake_path", "custom_table")

    @patch("powergenome.transmission.load_data")
    def test_custom_region_aggregations(self, mock_load_data, sample_transmission_data):
        """Test using custom regional aggregations."""
        mock_load_data.return_value = sample_transmission_data

        result = agg_transmission_constraints(
            "fake_path",
            model_regions=["AB", "C"],
            zone_num_map={"AB": 1, "C": 2},
            region_aggregations={"AB": ["A", "B"]},
            tx_value_col="firm_ttc_mw",
        )

        assert isinstance(result, pd.DataFrame)


@pytest.mark.integration
class TestTransmissionIntegration:
    """Integration tests for the transmission functions."""

    @patch("powergenome.transmission.load_data")
    def test_end_to_end_with_aggregation(self, mock_load_data):
        """Test complete workflow with region aggregation."""
        # Create more complex test data
        test_data = pd.DataFrame(
            {
                "region_from": ["A", "B", "C", "D", "A", "B"],
                "region_to": ["B", "A", "D", "C", "C", "D"],
                "firm_ttc_mw": [100, 100, 200, 200, 150, 250],
                "id": [1, 2, 3, 4, 5, 6],
            }
        )

        mock_load_data.return_value = test_data

        result = agg_transmission_constraints(
            "fake_path",
            model_regions=["AB", "CD"],
            zone_num_map={"AB": 1, "CD": 2},
            region_aggregations={"AB": ["A", "B"], "CD": ["C", "D"]},
            tx_value_col="firm_ttc_mw",
        )

        # Should have aggregated transmission between AB and CD regions
        assert (
            len(result["Network_Lines"].dropna()) <= 1
        )  # Only one connection possible between AB and CD
        assert (
            "AB" in result["start_region"].values[0]
            or "CD" in result["start_region"].values[0]
        )


class TestLoadTxCosts:
    """Tests for load_tx_costs function."""

    @pytest.fixture
    def sample_tx_cost_data(self):
        """Sample transmission cost data for testing."""
        return pd.DataFrame(
            {
                "start_region": ["A", "B", "C"],
                "dest_region": ["B", "C", "A"],
                "total_interconnect_annuity_mw": [1000, 1500, 1200],
                "total_interconnect_cost_mw": [50000, 75000, 60000],
                "dollar_year": [2020, 2019, 2021],
                "total_line_loss_frac": [0.02, 0.03, 0.025],
            }
        )

    @pytest.fixture
    def sample_model_regions(self):
        """Sample model regions for testing."""
        return ["A", "B", "C"]

    @pytest.fixture
    def sample_settings_with_costs(self):
        """Sample settings for cost adjustment testing."""
        return {
            "zone_num_map": {"A": "z1", "B": "z2", "C": "z3"},
            "data_location": "fake_path",
            "dollar_year_table": "inflation_table",
        }

    @patch("powergenome.transmission.load_data")
    def test_basic_load_without_adjustment(
        self,
        mock_load_data,
        sample_tx_cost_data,
        sample_model_regions,
        sample_settings_with_costs,
    ):
        """Test basic loading without inflation adjustment."""
        mock_load_data.return_value = sample_tx_cost_data

        result = load_tx_costs(
            "fake_path",
            "cost_table",
            zone_num_map=sample_settings_with_costs["zone_num_map"],
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "zone_1" in result.columns
        assert "zone_2" in result.columns
        assert "adjusted_dollar_year" not in result.columns

        # Check zone mapping
        assert result.loc[0, "zone_1"] == "z1"  # A -> z1
        assert result.loc[0, "zone_2"] == "z2"  # B -> z2

    @patch("powergenome.transmission.load_data")
    def test_missing_required_columns(
        self, mock_load_data, sample_model_regions, sample_settings_with_costs
    ):
        """Test error when required columns are missing."""
        incomplete_data = pd.DataFrame(
            {
                "start_region": ["A"],
                "dest_region": ["B"],
                # Missing required cost and dollar_year columns
            }
        )
        mock_load_data.return_value = incomplete_data

        with pytest.raises(KeyError, match="Missing required columns"):
            load_tx_costs(
                "fake_path",
                "cost_table",
                zone_num_map=sample_settings_with_costs["zone_num_map"],
            )

    @patch("powergenome.transmission.load_data")
    @patch("powergenome.transmission.inflation_price_adjustment")
    def test_inflation_adjustment(
        self,
        mock_inflation,
        mock_load_data,
        sample_tx_cost_data,
        sample_model_regions,
        sample_settings_with_costs,
    ):
        """Test inflation adjustment functionality."""
        mock_load_data.return_value = sample_tx_cost_data
        mock_inflation.return_value = 1100.0  # Mock adjusted value

        result = load_tx_costs(
            "fake_path",
            "cost_table",
            target_usd_year=2022,
            zone_num_map=sample_settings_with_costs["zone_num_map"],
            dollar_year_table=sample_settings_with_costs["dollar_year_table"],
        )

        assert "adjusted_dollar_year" in result.columns
        assert result["adjusted_dollar_year"].iloc[0] == 2022
        assert mock_inflation.call_count == 6  # 2 cost columns × 3 rows

    @patch("powergenome.transmission.load_data")
    def test_inflation_adjustment_missing_dollar_year_table(
        self, mock_load_data, sample_tx_cost_data, sample_settings_with_costs
    ):
        """Test error when dollar_year_table missing for inflation adjustment."""
        mock_load_data.return_value = sample_tx_cost_data

        with pytest.raises(ValueError, match="Dollar year table is required"):
            load_tx_costs(
                "fake_path",
                "cost_table",
                target_usd_year=2022,
                zone_num_map=sample_settings_with_costs["zone_num_map"],
                dollar_year_table=None,
            )

    @patch("powergenome.transmission.load_data")
    def test_filters_invalid_regions(
        self, mock_load_data, sample_model_regions, sample_settings_with_costs
    ):
        """Test that rows with unmapped regions are filtered out."""
        data_with_invalid = pd.DataFrame(
            {
                "start_region": ["A", "D", "B"],  # D not in zone_num_map
                "dest_region": ["B", "E", "C"],  # E not in zone_num_map
                "total_interconnect_annuity_mw": [1000, 1500, 1200],
                "total_interconnect_cost_mw": [50000, 75000, 60000],
                "dollar_year": [2020, 2019, 2021],
            }
        )
        mock_load_data.return_value = data_with_invalid

        result = load_tx_costs(
            "fake_path",
            "cost_table",
            zone_num_map=sample_settings_with_costs["zone_num_map"],
        )

        # Should only have A-B and B-C, D-E should be filtered out
        assert len(result) == 2
        assert "D" not in result["start_region"].values
        assert "E" not in result["dest_region"].values


class TestInsertTxCosts:
    """Tests for insert_tx_costs function."""

    @pytest.fixture
    def sample_tx_df(self):
        """Sample transmission dataframe."""
        return pd.DataFrame(
            {
                "Network_Lines": [1, 2],
                "start_region": ["A", "B"],
                "dest_region": ["B", "C"],
                "Line_Max_Flow_MW": [1000, 1500],
                "transmission_path_name": ["A_to_B", "B_to_C"],
            }
        )

    @pytest.fixture
    def sample_tx_costs_df(self):
        """Sample transmission costs dataframe."""
        return pd.DataFrame(
            {
                "start_region": ["A", "B"],
                "dest_region": ["B", "C"],
                "zone_1": ["z1", "z2"],
                "zone_2": ["z2", "z3"],
                "total_interconnect_annuity_mw": [1000, 1500],
                "total_interconnect_cost_mw": [50000, 75000],
                "total_line_loss_frac": [0.02, 0.03],
            }
        )

    def test_empty_transmission_df(self, sample_tx_costs_df):
        """Test handling of empty transmission dataframe."""
        empty_df = pd.DataFrame()
        result = insert_tx_costs(empty_df, sample_tx_costs_df)
        assert result.empty

    def test_basic_cost_insertion(self, sample_tx_df, sample_tx_costs_df):
        """Test basic cost insertion functionality."""
        result = insert_tx_costs(sample_tx_df, sample_tx_costs_df)

        # Check that cost columns were added
        expected_cost_columns = [
            "Line_Reinforcement_Cost_per_MWyr",
            "Line_Reinforcement_Cost_per_MW",
            "Line_Loss_Percentage",
        ]
        for col in expected_cost_columns:
            assert col in result.columns

        # Check that original columns are preserved
        assert "Network_Lines" in result.columns
        assert "Line_Max_Flow_MW" in result.columns

        # Check values were correctly mapped
        assert result.loc[0, "Line_Reinforcement_Cost_per_MWyr"] == 1000
        assert result.loc[0, "Line_Loss_Percentage"] == 0.02

    def test_bidirectional_cost_creation(self, sample_tx_costs_df):
        """Test that bidirectional costs are created correctly."""
        # Create transmission df with reverse direction
        tx_df_reverse = pd.DataFrame(
            {
                "start_region": ["B", "C"],
                "dest_region": ["A", "B"],
                "Line_Max_Flow_MW": [1000, 1500],
            }
        )

        result = insert_tx_costs(tx_df_reverse, sample_tx_costs_df)

        # Should find costs for B->A (reverse of A->B in cost data)
        assert not pd.isna(result.loc[0, "Line_Reinforcement_Cost_per_MWyr"])
        assert result.loc[0, "Line_Reinforcement_Cost_per_MWyr"] == 1000

    def test_missing_cost_data(self, sample_tx_costs_df):
        """Test handling when transmission lines have no cost data."""
        tx_df_no_match = pd.DataFrame(
            {
                "start_region": ["X", "Y"],
                "dest_region": ["Y", "Z"],
                "Line_Max_Flow_MW": [1000, 1500],
            }
        )

        result = insert_tx_costs(tx_df_no_match, sample_tx_costs_df)

        # Cost columns should exist but have NaN values
        assert "Line_Reinforcement_Cost_per_MWyr" in result.columns
        assert pd.isna(result["Line_Reinforcement_Cost_per_MWyr"]).all()

    def test_partial_cost_columns(self, sample_tx_df):
        """Test handling when cost data is missing some optional columns."""
        partial_costs = pd.DataFrame(
            {
                "start_region": ["A", "B"],
                "dest_region": ["B", "C"],
                "zone_1": ["z1", "z2"],
                "zone_2": ["z2", "z3"],
                "total_interconnect_annuity_mw": [1000, 1500],
                "total_interconnect_cost_mw": [50000, 75000],
                # Missing total_line_loss_frac
            }
        )

        result = insert_tx_costs(sample_tx_df, partial_costs)

        # Should have the available cost columns
        assert "Line_Reinforcement_Cost_per_MWyr" in result.columns
        assert "Line_Reinforcement_Cost_per_MW" in result.columns
        # Should not have the missing column
        assert "Line_Loss_Percentage" not in result.columns

    def test_cost_data_with_na_zones(self, sample_tx_df):
        """Test that rows with NA zones are filtered out from cost data."""
        costs_with_na = pd.DataFrame(
            {
                "start_region": ["A", "B", "D"],
                "dest_region": ["B", "C", "E"],
                "zone_1": ["z1", "z2", None],  # Third row has None
                "zone_2": ["z2", "z3", "z5"],
                "total_interconnect_annuity_mw": [1000, 1500, 2000],
                "total_interconnect_cost_mw": [50000, 75000, 100000],
            }
        )

        result = insert_tx_costs(sample_tx_df, costs_with_na)

        # Should successfully merge available data, ignore rows with NA zones
        assert not result.empty
        assert result.loc[0, "Line_Reinforcement_Cost_per_MWyr"] == 1000
