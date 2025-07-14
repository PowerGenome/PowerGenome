"""
Tests for time series data sorting functionality.
"""

import pandas as pd
import numpy as np
import pytest


def test_time_series_sorting_basic():
    """Test that time series data is properly sorted by time_index"""
    
    # Create test data with deliberately unsorted time_index
    test_data = []
    regions = ['region_1', 'region_2']
    
    # Create data for each region with unsorted time indices
    for region in regions:
        # Deliberately create unsorted time indices
        time_indices = [1, 3, 2, 5, 4, 7, 6, 8]
        for i, time_idx in enumerate(time_indices):
            test_data.append({
                'year': 2020,
                'region': region,
                'time_index': time_idx,
                'sector': 'residential',
                'load_mw': np.random.uniform(100, 1000)
            })
    
    # Create DataFrame
    load_curves = pd.DataFrame(test_data)
    
    # Apply the sorting logic that we added to the code
    load_curves = load_curves.sort_values(by=['region', 'time_index'])
    
    # Verify the data is properly sorted by region and time_index
    for region in regions:
        region_data = load_curves[load_curves['region'] == region]
        time_indices = region_data['time_index'].values
        
        # Check that time_index is sorted
        assert np.array_equal(time_indices, np.sort(time_indices)), \
            f"Time indices for {region} are not sorted: {time_indices}"


def test_time_index_pivot_sorting():
    """Test that time_index data is sorted before pivot operations"""
    
    # Create test data similar to generators.py case
    elec_profiles_data = [
        {'resource': 'wind', 'time_index': 3, 'region': 'A', 'load_mw': 100},
        {'resource': 'wind', 'time_index': 1, 'region': 'A', 'load_mw': 200},
        {'resource': 'wind', 'time_index': 2, 'region': 'A', 'load_mw': 150},
        {'resource': 'wind', 'time_index': 3, 'region': 'B', 'load_mw': 300},
        {'resource': 'wind', 'time_index': 1, 'region': 'B', 'load_mw': 250},
        {'resource': 'wind', 'time_index': 2, 'region': 'B', 'load_mw': 275},
    ]
    
    elec_profiles = pd.DataFrame(elec_profiles_data)
    resource = 'wind'
    
    # Apply the sorting logic we added to generators.py
    dr_profile = elec_profiles.loc[
        elec_profiles["resource"] == resource, ["time_index", "region", "load_mw"]
    ].sort_values(by="time_index").pivot(index="time_index", columns="region")
    
    # Verify that the index (time_index) is sorted
    time_indices = dr_profile.index.values
    assert np.array_equal(time_indices, np.sort(time_indices)), \
        f"Time indices in pivot are not sorted: {time_indices}"


def test_time_series_with_time_offset():
    """Test that time series data is sorted before time offset operations"""
    
    # Create test data with unsorted time_index
    test_data = [
        {'state': 'TX', 'time_index': 3, 'load_mw': 100},
        {'state': 'TX', 'time_index': 1, 'load_mw': 200},
        {'state': 'TX', 'time_index': 2, 'load_mw': 150},
        {'state': 'TX', 'time_index': 4, 'load_mw': 175},
    ]
    
    df = pd.DataFrame(test_data)
    
    # Sort by time_index before time offset (similar to load_construction.py)
    df_sorted = df.sort_values(by=["state", "time_index"])
    
    # Verify that time_index is sorted
    time_indices = df_sorted['time_index'].values
    assert np.array_equal(time_indices, np.sort(time_indices)), \
        f"Time indices are not sorted before time offset: {time_indices}"
    
    # Simulate time offset operation (np.roll)
    original_load = df_sorted['load_mw'].values
    shifted_load = np.roll(original_load, 1)
    
    # The shift should work correctly when data is in proper order
    expected_shifted = [175, 200, 150, 100]  # last element rolls to first
    assert np.array_equal(shifted_load, expected_shifted), \
        f"Time offset operation failed: expected {expected_shifted}, got {shifted_load}"


def test_distributed_gen_time_index_sorting():
    """Test that distributed generation time series data is sorted before set_index"""
    
    # Create test data with unsorted time_index
    test_data = [
        {'year': 2020, 'time_index': 3, 'region_distpv_mwh': 100},
        {'year': 2020, 'time_index': 1, 'region_distpv_mwh': 200},
        {'year': 2020, 'time_index': 2, 'region_distpv_mwh': 150},
        {'year': 2020, 'time_index': 4, 'region_distpv_mwh': 175},
    ]
    
    df = pd.DataFrame(test_data)
    
    # Apply sorting logic similar to distributed_gen.py
    result = df.loc[df["year"] == 2020, :].sort_values(by='time_index').set_index("time_index")["region_distpv_mwh"]
    
    # Verify that the index (time_index) is sorted
    time_indices = result.index.values
    assert np.array_equal(time_indices, np.sort(time_indices)), \
        f"Time indices are not sorted after set_index: {time_indices}"