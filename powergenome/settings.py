"""
Parameters and settings management for PowerGenome.

This module provides a Settings class and utility functions for managing PowerGenome
configuration parameters.

Key Features:
- Load settings from YAML files or directories
- Context-based settings management for concurrent operations
- Scenario-specific settings creation
- Dictionary-like access with validation

Architecture:
The Settings class uses Python's contextvars for managing the current settings
instance across different execution contexts (see Example below).
This allows multiple settings to coexist (useful for testing and parallel processing)
while providing a clean API for accessing the current settings.

Examples:
    # Basic usage
    settings = Settings(config_path="path/to/settings.yml")

    # Access settings
    regions = settings["model_regions"]
    fuel_cost = settings.get("fuel_cost", default=0.0)

    # Context management
    with settings:
        current = get_current_settings()  # Returns the settings instance
        # All code in this block can access current settings

    # Scenario-specific settings
    base_settings = Settings(config_path="base.yml")
    scenario_settings = Settings.for_scenario(
        base_settings,
        {"model_year": 2030, "carbon_tax": 50}
    )

    # Load multiple YAML files from directory
    settings = Settings(config_path="path/to/settings/directory")

    # Create from dictionary
    settings = Settings.from_dict({"model_regions": ["CA", "TX"]})

    # Update settings
    settings.update({"new_param": "value"})

    # Convert to dictionary
    settings_dict = settings.to_dict()

See Also:
    - powergenome.util: Utility functions for settings processing
    - powergenome.params: Legacy parameter definitions
"""

import copy
import logging
from contextvars import ContextVar

# Standard library imports
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Union

import pandas as pd
from flatten_dict import flatten
from ruamel.yaml import YAML

# Local imports
from powergenome.util import make_iterable, sort_nested_dict, update_dictionary

logger = logging.getLogger(__name__)

# Context variable for current settings
_current_settings: ContextVar = ContextVar("current_settings", default=None)

# Global settings variable (alternative to context-based approach)
_global_settings: "Settings" = None


class Settings:
    def __init__(self, config_path: Union[str, Path] = None, data: dict = None):
        """
        Initialize Settings instance.

        Parameters
        ----------
        config_path : str or Path, optional
            Path to settings file or directory. If a file, loads single YAML file.
            If a directory, loads all .yml files in the directory and merges them.
            File settings override data settings if both are provided.
        data : dict, optional
            Dictionary of settings data. If both config_path and data are provided,
            data is loaded first, then file settings are loaded and may override data.

        Examples
        --------
        >>> # Load from file
        >>> settings = Settings(config_path="settings.yml")

        >>> # Load from directory
        >>> settings = Settings(config_path="settings/")

        >>> # Create from dictionary
        >>> settings = Settings(data={"model_regions": ["CA", "TX"]})

        >>> # Combine data and file
        >>> settings = Settings(
        ...     data={"base_param": "value"},
        ...     config_path="settings.yml"
        ... )
        """
        self._data = {}
        if data:
            self._data.update(data)
        if config_path:
            logger.info(f"Loading settings from {config_path}")
            self.load_settings(config_path)

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a Settings instance from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing settings data.

        Returns
        -------
        Settings
            New Settings instance with the provided data.

        Examples
        --------
        >>> data = {"model_regions": ["CA", "TX"], "model_year": 2030}
        >>> settings = Settings.from_dict(data)
        >>> settings["model_regions"]
        ['CA', 'TX']
        """
        return cls(data=data)

    @classmethod
    def for_scenario(cls, base_settings: "Settings", scenario_data: dict):
        """
        Create a Settings instance specifically for a scenario.

        This method creates a new Settings instance by copying the base settings
        and then applying scenario-specific overrides. The base settings remain
        unchanged.

        Parameters
        ----------
        base_settings : Settings
            Base settings instance to copy from.
        scenario_data : dict
            Dictionary of scenario-specific settings that will override
            or add to the base settings.

        Returns
        -------
        Settings
            New Settings instance with base settings plus scenario overrides.

        Examples
        --------
        >>> base = Settings(data={"model_year": 2020, "regions": ["CA"]})
        >>> scenario = Settings.for_scenario(
        ...     base,
        ...     {"model_year": 2030, "carbon_tax": 50}
        ... )
        >>> scenario["model_year"]  # Overridden
        2030
        >>> scenario["regions"]     # Preserved from base
        ['CA']
        >>> scenario["carbon_tax"]  # Added from scenario
        50
        """
        new_settings = cls()
        new_settings._data = base_settings._data.copy()
        new_settings._data.update(scenario_data)
        return new_settings

    def __enter__(self):
        """
        Context manager entry - set this as the current settings.

        When entering a context, this settings instance becomes the current
        settings that can be accessed via get_current_settings().

        Returns
        -------
        Settings
            Self reference for use in the context.

        Examples
        --------
        >>> settings = Settings(data={"model_year": 2030})
        >>> with settings:
        ...     current = get_current_settings()
        ...     print(current["model_year"])
        2030
        """
        self._token = _current_settings.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - restore previous settings.

        When exiting a context, the previous settings are restored,
        allowing nested contexts to work correctly.

        Parameters
        ----------
        exc_type : type, optional
            Exception type if an exception occurred.
        exc_val : Exception, optional
            Exception value if an exception occurred.
        exc_tb : traceback, optional
            Exception traceback if an exception occurred.
        """
        _current_settings.reset(self._token)

    def load_settings(self, config_path: Union[str, Path]):
        """
        Load settings from a file or directory and update current data.

        This method loads settings from the specified path and merges them
        with the current settings data. New settings override existing ones.

        Parameters
        ----------
        config_path : str or Path
            Path to settings file or directory containing YAML files.

        Examples
        --------
        >>> settings = Settings()
        >>> settings.load_settings("path/to/settings.yml")
        >>> settings.load_settings("path/to/settings/directory")
        """
        self._data.update(load_settings(config_path))

    def get(self, key, default=None):
        """
        Get a setting value with optional default.

        Parameters
        ----------
        key : str
            The setting key to retrieve.
        default : any, optional
            Default value to return if key is not found.

        Returns
        -------
        any
            The setting value or default if not found.

        Examples
        --------
        >>> settings = Settings(data={"model_year": 2030})
        >>> settings.get("model_year")
        2030
        >>> settings.get("missing_key", "default_value")
        'default_value'
        >>> settings.get("missing_key")  # Returns None
        None
        """
        return self._data.get(key, default)

    def __getitem__(self, key):
        """
        Get a setting value using dictionary-style access.

        Parameters
        ----------
        key : str
            The setting key to retrieve.

        Returns
        -------
        any
            The setting value.

        Raises
        ------
        KeyError
            If the key is not found in settings.

        Examples
        --------
        >>> settings = Settings(data={"model_year": 2030})
        >>> settings["model_year"]
        2030
        >>> settings["missing_key"]  # Raises KeyError
        KeyError: Setting 'missing_key' not found in settings...
        """
        if key not in self._data:
            raise KeyError(
                f"Setting '{key}' not found in settings. Available keys: {list(self._data.keys())}"
            )
        return self._data[key]

    def __setitem__(self, key, value):
        """
        Set a setting value using dictionary-style access.

        Parameters
        ----------
        key : str
            The setting key to set.
        value : any
            The value to assign to the setting.

        Examples
        --------
        >>> settings = Settings()
        >>> settings["model_year"] = 2030
        >>> settings["model_year"]
        2030
        """
        self._data[key] = value

    def pop(self, key, default=None):
        """Remove and return the value for key if key is in the settings, else return default.

        Parameters
        ----------
        key : str
            The key to remove from settings.
        default : any, optional
            Value to return if key is not found.

        Returns
        -------
        any
            The value associated with the key, or default if key not found.
        """
        # Check if the key exists first
        if key not in self._data:
            if default is None:
                # No default provided and key doesn't exist, raise KeyError
                raise KeyError(key)
            else:
                # Default provided, return it
                return default
        # Key exists, remove and return it
        return self._data.pop(key)

    def __getattr__(self, key):
        """
        Get a setting value using attribute-style access.

        This method allows accessing settings as attributes (e.g., settings.model_year).
        Returns None if the attribute is not found (does not raise AttributeError).

        Parameters
        ----------
        key : str
            The setting key to retrieve.

        Returns
        -------
        any
            The setting value or None if not found.

        Examples
        --------
        >>> settings = Settings(data={"model_year": 2030})
        >>> settings.model_year
        2030
        >>> settings.missing_attr  # Returns None, doesn't raise AttributeError
        None
        """
        # During unpickling, _data may not exist yet. Access it via __dict__ to
        # avoid recursive __getattr__ calls.
        data = self.__dict__.get("_data")
        if data is None:
            raise AttributeError(key)
        return data.get(key, None)

    def __copy__(self):
        """
        Create a shallow copy of the Settings object.

        Returns
        -------
        Settings
            A new Settings instance with a shallow copy of the data.

        Examples
        --------
        >>> settings = Settings(data={"nested": {"key": "value"}})
        >>> copy_settings = copy.copy(settings)
        >>> copy_settings["nested"] is settings["nested"]  # Same reference
        True
        """
        new_settings = Settings()
        new_settings._data = self._data.copy()
        return new_settings

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the Settings object.

        Parameters
        ----------
        memo : dict
            Memoization dictionary for deep copy.

        Returns
        -------
        Settings
            A new Settings instance with a deep copy of the data.

        Examples
        --------
        >>> settings = Settings(data={"nested": {"key": "value"}})
        >>> deep_copy = copy.deepcopy(settings)
        >>> deep_copy["nested"] is settings["nested"]  # Different reference
        False
        """
        if id(self) in memo:
            return memo[id(self)]

        new_settings = Settings()
        memo[id(self)] = new_settings
        new_settings._data = copy.deepcopy(self._data, memo)
        return new_settings

    def to_dict(self) -> dict:
        """
        Convert the Settings object to a dictionary.

        Returns
        -------
        dict
            A copy of the settings data as a dictionary.

        Examples
        --------
        >>> settings = Settings(data={"model_year": 2030, "regions": ["CA"]})
        >>> settings_dict = settings.to_dict()
        >>> settings_dict
        {'model_year': 2030, 'regions': ['CA']}
        >>> settings_dict is not settings._data  # Returns a copy
        True
        """
        return self._data.copy()

    def update(self, updates: dict):
        """
        Update the settings with new values.

        This method merges the provided dictionary with the current settings.
        Existing keys are overwritten, new keys are added.

        Parameters
        ----------
        updates : dict
            Dictionary of settings to update.

        Examples
        --------
        >>> settings = Settings(data={"model_year": 2020})
        >>> settings.update({"model_year": 2030, "carbon_tax": 50})
        >>> settings["model_year"]  # Updated
        2030
        >>> settings["carbon_tax"]  # Added
        50
        """
        self._data.update(updates)

    def get_data(self) -> dict:
        """
        Get a reference to the internal data dictionary.

        Warning: This returns a reference to the internal data.
        Modifying it will affect the Settings object.

        Returns
        -------
        dict
            Reference to the internal data dictionary.

        Examples
        --------
        >>> settings = Settings(data={"model_year": 2030})
        >>> data = settings.get_data()
        >>> data["new_key"] = "new_value"  # Modifies the settings
        >>> settings["new_key"]
        'new_value'
        """
        return self._data

    @classmethod
    def set_global(cls, settings: "Settings"):
        """
        Set a Settings instance as the global default.

        This allows accessing settings without using a context manager.
        The settings can then be accessed via get_current_settings() or
        Settings.get_global() from anywhere in the code.

        Parameters
        ----------
        settings : Settings
            The Settings instance to set as global.

        Examples
        --------
        >>> settings = Settings(config_path="settings.yml")
        >>> Settings.set_global(settings)
        >>> current = get_current_settings()  # Works without "with" statement
        >>> current["model_year"]
        2030
        """
        global _global_settings
        _global_settings = settings

    @classmethod
    def get_global(cls) -> "Settings":
        """
        Get the global Settings instance.

        Returns
        -------
        Settings
            The global settings instance.

        Raises
        ------
        RuntimeError
            If no global settings have been set.

        Examples
        --------
        >>> settings = Settings(data={"model_year": 2030})
        >>> Settings.set_global(settings)
        >>> global_settings = Settings.get_global()
        >>> global_settings["model_year"]
        2030
        """
        if _global_settings is None:
            raise RuntimeError(
                "No global settings have been set. Use Settings.set_global(settings) first."
            )
        return _global_settings

    @classmethod
    def clear_global(cls):
        """
        Clear the global Settings instance.

        Examples
        --------
        >>> Settings.clear_global()
        >>> Settings.get_global()  # Raises RuntimeError
        """
        global _global_settings
        _global_settings = None


def get_current_settings() -> Settings:
    """
    Get the current settings from context or global settings.

    This function first tries to get settings from the current context (if using
    a "with" statement), then falls back to global settings if available.

    Returns
    -------
    Settings
        The current settings instance from context or global settings.

    Raises
    ------
    RuntimeError
        If no settings are available in either context or global scope.

    Examples
    --------
    # Using context (original approach)
    >>> settings = Settings(data={"model_year": 2030})
    >>> with settings:
    ...     current = get_current_settings()
    ...     print(current["model_year"])
    2030

    # Using global settings (new approach)
    >>> settings = Settings(data={"model_year": 2030})
    >>> Settings.set_global(settings)
    >>> current = get_current_settings()  # Works without "with"
    >>> current["model_year"]
    2030
    """
    # First try to get from context
    settings = _current_settings.get()
    if settings is not None:
        return settings

    # Fall back to global settings
    if _global_settings is not None:
        return _global_settings

    # No settings available
    raise RuntimeError(
        "No settings are currently available. Either use 'with settings:' context manager "
        "or call Settings.set_global(settings) to set global settings."
    )


def load_settings(path: Union[str, Path]) -> dict:
    """
    Load a YAML file or a directory of YAML files with settings parameters.

    This function loads settings from YAML files and performs post-processing
    including path resolution, tag processing, and parameter name fixes.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the settings file or directory. If a file, loads single YAML.
        If a directory, loads all .yml files and merges them.

    Returns
    -------
    dict
        All parameters from the YAML file(s) with post-processing applied.

    Raises
    ------
    FileNotFoundError
        If the path does not exist or is not a file/directory.
    """

    path = Path(path)
    if path.is_file():
        with open(path, "r") as f:
            #     settings = yaml.safe_load(f)
            yaml = YAML(typ="safe")
            settings = yaml.load(f)
    elif path.is_dir():
        settings = {}
        for sf in path.glob("*.yml"):
            yaml = YAML(typ="safe")
            s = yaml.load(sf)
            if s:
                settings.update(s)
    else:
        raise FileNotFoundError(
            "Path is not recognized. Check that your path is valid."
        )

    # Settings directories contain YAML files, while their relative data paths
    # are rooted at the directory containing the settings folder.
    settings_folder = path.parent

    data_locations = settings.get("data_location")
    if data_locations:
        values = (
            data_locations if isinstance(data_locations, list) else [data_locations]
        )
        resolved = [
            Path(value) if Path(value).is_absolute() else settings_folder / value
            for value in values
        ]
        settings["data_location"] = (
            resolved if isinstance(data_locations, list) else resolved[0]
        )

    input_folder = settings.get("input_folder")
    if input_folder and not isinstance(input_folder, list):
        input_folder_path = Path(input_folder)
        settings["input_folder"] = (
            input_folder_path
            if input_folder_path.is_absolute()
            else settings_folder / input_folder_path
        )

    if settings.get("generator_columns"):
        settings["generator_columns"] = add_model_tags_to_gen_columns(
            model_tag_values=settings.get("model_tag_values", {}),
            regional_tag_values=settings.get("regional_tag_values", {}),
            generator_columns=settings["generator_columns"],
        )

    settings = apply_all_tag_to_regions(settings)
    settings = expand_capacity_reserve_values(settings)
    settings = sort_nested_dict(settings)

    for key in [
        "EFS_DATA",
        "RESOURCE_GROUPS",
        "DISTRIBUTED_GEN_DATA",
        "RESOURCE_GROUP_PROFILES",
    ]:
        value = settings.get(key)
        if value:
            if isinstance(value, list):
                settings[key] = [Path(v) for v in value]
            else:
                settings[key] = Path(value)

    settings["model_regions"] = sorted(settings["model_regions"])
    zones = settings["model_regions"]
    logger.info(f"Sorted model regions are {', '.join(zones)}")
    zone_num_map = {
        zone: f"{number + 1}" for zone, number in zip(zones, range(len(zones)))
    }
    settings["zone_num_map"] = zone_num_map

    return fix_param_names(settings)


def add_model_tags_to_gen_columns(
    model_tag_values: Dict[str, Dict[str, int]],
    regional_tag_values: Dict[str, Dict[str, Dict[str, int]]],
    generator_columns: List[str],
) -> List[str]:
    """Add model resource tag keys to the list of columns that will be included in
    generator outputs.

    Parameters
    ----------
    model_tag_values : Dict[str, Dict[str, int]]
        Tags applied to resources in all regions. Top level is the tag name, which will
        become a column in the generators output. The next level is technology names
        and the value for each technology.
    regional_tag_values : Dict[str, Dict[str, Dict[str, int]]]
        Regional values applied to technologies. Top level is the region, then the tag
        name, then the technology name and value.
    generator_columns : List[str]
        List of columns to include in generator outputs from the settings.

    Returns
    -------
    List[str]
        Updated list of column names, now including any resource tags/columns.

    Example
    -------
    >>> model_tag_values = {'cost': {'solar': 100, 'wind': 150}}
    >>> regional_tag_values = {'NA': {'other_tag': {'solar': 20, 'wind': 25}}}
    >>> generator_columns = ['capacity', 'output']
    >>> add_model_tags_to_gen_columns(model_tag_values, regional_tag_values, generator_columns)
    ['capacity', 'output', 'cost', 'other_tag']

    """

    if not isinstance(generator_columns, list):
        logger.warning(
            "There is a parameter 'generator_columns' in your settings but it is not a "
            "list. This parameter will not have any effect in it's current form."
        )
        return generator_columns

    tag_keys = list((model_tag_values or {}).keys())
    regional_keys = []
    for region, regional_tags in (regional_tag_values or {}).items():
        regional_keys.extend(list(regional_tags.keys()))

    tag_keys = set(tag_keys + regional_keys)
    for tag in tag_keys:
        if tag not in generator_columns:
            generator_columns.append(tag)

    return generator_columns


def apply_all_tag_to_regions(settings: dict) -> dict:
    """Make copies of renewables_clusters dicts with region "all"

    If a renewables clustering object doesn't already existing for a region/technology
    then make a copy for use. This is helpful with large numbers of regions when
    the clustering parameters can be applied everywhere.

    Parameters
    ----------
    settings : dict
        All user-specified settings from YAML files

    Returns
    -------
    dict
        Copy of the input settings with renewables_clusters objects for all regions

    Raises
    ------
    KeyError
        The dictionary is missing the tag "region"
    KeyError
        The dictionary with region "all" is missing the tag "technology"
    """

    settings_all = dict()
    all_regions = settings["model_regions"]

    # Keeps a list of which regions should be modified by "all" (are not specifically tagged)
    techs_tagged_w_all = []
    techs_tagged_by_region = dict()

    i = 0
    to_delete = []

    # These are the keys in settings which will not be used to determine whether 'all' should apply to that region
    identifier_keys = ["technology", "pref_site", "turbine_type", "type"]

    for d in settings.get("renewables_clusters", []) or []:
        if "region" not in d:
            raise KeyError("Entry missing 'region' tag.")

        reg = d["region"]

        keys = sorted(d.keys())
        tech = ""
        for key in keys:
            if key in identifier_keys:
                if tech != "":
                    tech += "_"
                tech += str(d[key])

        # Update the dict stating that this technology is specified for this region
        if tech in techs_tagged_by_region:
            techs_tagged_by_region[tech].append(reg)
        elif reg.lower() == "all":
            techs_tagged_by_region[tech] = []
        else:
            techs_tagged_by_region[tech] = [reg]

        if reg.lower() == "all":
            settings_all[tech] = d

            if "technology" not in d:
                raise KeyError(f"""Entry for {reg} missing 'technology' tag.""")

            if tech in techs_tagged_w_all:
                s = f"""
                Multiple 'all' tags applied to technology {tech}. Only last one will be used.
                """
                logger.warning(s)

            else:
                techs_tagged_w_all.append(tech)

            to_delete.append(i)

        # Keeps track of the "all" tags so that they can be deleted later in the function
        i += 1

    for i in reversed(to_delete):
        del settings["renewables_clusters"][i]

    for tech in techs_tagged_w_all:
        for reg in all_regions:
            if reg not in techs_tagged_by_region[tech]:
                temp_entry = settings_all[tech].copy()
                temp_entry["region"] = reg

                settings["renewables_clusters"].append(temp_entry)

    return settings


def expand_capacity_reserve_values(settings: dict) -> dict:
    """
    Expand capacity_reserve_values and regional_capacity_reserves into regional_tag_values.

    This function provides a user-friendly shorthand for specifying capacity reserve credit
    values across multiple regions and constraints. It transforms two settings parameters:
    - capacity_reserve_values: Technology credit values (flat or nested by constraint)
    - regional_capacity_reserves: Regional assignments to capacity reserve constraints

    into the standard regional_tag_values structure.

    Parameters
    ----------
    settings : dict
        Settings dictionary containing optional keys:
        - capacity_reserve_values: Dict of technology values (flat or nested)
        - regional_capacity_reserves: Dict mapping constraints to regions

    Returns
    -------
    dict
        Modified settings with expanded regional_tag_values. Existing explicit entries
        in regional_tag_values take precedence over auto-generated ones.

    Notes
    -----
    Format auto-detection:
    - Flat format: All values in capacity_reserve_values are numbers
      Example: {Tech1: 0.9, Tech2: 0.95}
      These values are applied to all CapRes_* constraints in regional_capacity_reserves

    - Nested format: Values in capacity_reserve_values are dicts
      Example: {CapRes_1: {Tech1: 0.9, Tech2: 0.95}, CapRes_2: {...}}
      Each constraint's values are applied only to that constraint

    Examples
    --------
    # Flat format: single capacity reserve constraint
    settings = {
        'capacity_reserve_values': {
            'Conventional Steam Coal': 0.9,
            'Natural Gas Fired Combined Cycle': 0.9
        },
        'regional_capacity_reserves': {
            'CapRes_1': {'p1': 0.164, 'p2': 0.164}
        },
        'model_regions': ['p1', 'p2']
    }
    result = expand_capacity_reserve_values(settings)
    # result['regional_tag_values'] = {
    #     'p1': {'CapRes_1': {'Conventional Steam Coal': 0.9, ...}},
    #     'p2': {'CapRes_1': {'Conventional Steam Coal': 0.9, ...}}
    # }

    # Nested format: multiple capacity reserve constraints
    settings = {
        'capacity_reserve_values': {
            'CapRes_1': {
                'Conventional Steam Coal': 0.9,
                'Natural Gas Fired Combined Cycle': 0.9
            },
            'CapRes_2': {
                'Conventional Steam Coal': 0.95,
                'Natural Gas Fired Combined Cycle': 0.95
            }
        },
        'regional_capacity_reserves': {
            'CapRes_1': {'p8': 0.164, 'p9': 0.164},
            'CapRes_2': {'p1': 0.145, 'p2': 0.145}
        },
        'model_regions': ['p1', 'p2', 'p8', 'p9']
    }
    result = expand_capacity_reserve_values(settings)
    # result['regional_tag_values'] = {
    #     'p1': {'CapRes_2': {'Conventional Steam Coal': 0.95, ...}},
    #     'p2': {'CapRes_2': {'Conventional Steam Coal': 0.95, ...}},
    #     'p8': {'CapRes_1': {'Conventional Steam Coal': 0.9, ...}},
    #     'p9': {'CapRes_1': {'Conventional Steam Coal': 0.9, ...}}
    # }
    """

    # Get the capacity reserve settings, return early if not present
    capacity_reserve_values = settings.get("capacity_reserve_values")
    regional_capacity_reserves = settings.get("regional_capacity_reserves")

    if not capacity_reserve_values or not regional_capacity_reserves:
        return settings

    # Validate that all values are consistently dicts or non-dicts
    values = list(capacity_reserve_values.values())
    if not values:
        return settings
    all_dicts = all(isinstance(v, dict) for v in values)
    none_dicts = all(not isinstance(v, dict) for v in values)
    if not (all_dicts or none_dicts):
        raise ValueError(
            "All values in 'capacity_reserve_values' must be consistently either dicts (nested format) or non-dicts (flat format). "
            f"Found mixed types: {[type(v).__name__ for v in values]}"
        )
    # Auto-detect format: flat (all values are numbers) or nested (values are dicts)
    is_flat = none_dicts

    # Normalize capacity_reserve_values to nested format
    if is_flat:
        # Flat format: apply these values to all CapRes_* constraints
        constraint_names = list(regional_capacity_reserves.keys())
        normalized_values = {
            constraint: capacity_reserve_values.copy()
            for constraint in constraint_names
        }
    else:
        # Already nested
        normalized_values = capacity_reserve_values

    # Build regional_tag_values from the normalized structure
    expanded_regional_values = {}
    for constraint, tech_values in normalized_values.items():
        # Get regions assigned to this constraint
        regions_for_constraint = regional_capacity_reserves.get(constraint, {})

        for region in regions_for_constraint.keys():
            # Ensure region exists in the expanded dict
            if region not in expanded_regional_values:
                expanded_regional_values[region] = {}

            # Add this constraint and its technology values for this region
            expanded_regional_values[region][constraint] = tech_values.copy()

    # Merge with existing regional_tag_values, preserving explicit user entries
    existing_regional_values = settings.get("regional_tag_values", {})

    # Start with expanded values and layer existing values on top (for full precedence)
    merged_regional_values = copy.deepcopy(expanded_regional_values)

    # Layer existing values on top, so they take precedence
    for region, tags in existing_regional_values.items():
        if region not in merged_regional_values:
            merged_regional_values[region] = {}
        # For each constraint in existing values, merge its technology values
        for constraint, tech_values in tags.items():
            if constraint not in merged_regional_values[region]:
                merged_regional_values[region][constraint] = {}
            # Existing tech values override expanded ones
            merged_regional_values[region][constraint].update(tech_values)

    # Update settings with the merged regional_tag_values
    settings["regional_tag_values"] = merged_regional_values

    # Log what was populated
    regions_populated = set()
    for constraint, regions in regional_capacity_reserves.items():
        regions_populated.update(regions.keys())

    if regions_populated:
        logger.info(
            f"Expanded capacity_reserve_values into regional_tag_values for "
            f"regions: {sorted(regions_populated)}"
        )

    return settings


def fix_param_names(settings: dict) -> dict:
    fix_params = {
        "historical_load_region_maps": "historical_load_region_map",
        "demand_response_resources": "flexible_demand_resources",
        "data_years": "eia_data_years",
    }
    for k, v in fix_params.items():
        if k in settings:
            settings[v] = settings[k]
            s = f"""
            The settings parameter named {k} has been changed to {v}. Please correct it in
            your settings file.

            """
            logger.warning(s)
    return settings


def assign_model_planning_years(_settings: dict, year: int) -> dict:
    """Make sure "model_year" and "model_first_planning_year" appear as scalars.

    These can originally be set in any of these forms, in either the default
    settings or in the settings_management dictionary:

    model_year: 2040 and model_first_planning_year: 2031
    model_year: [2040, 2050] and model_first_planning_year: [2031, 2041]
    model_periods: (2031, 2040)
    model_periods: [(2031, 2040), (2041, 2050)]

    This function looks up the right values for the current year and assigns
    them as scalars (the first form above).

    Parameters
    ----------
    _settings : dict
        Model settings dictionary. Must have either "model_periods", "model_year"
        AND "model_first_planning_year", or "model_first_planning_year" as keys.
    year : int
        Model year.

    Returns
    -------
    dict
        Modified settings with scaler versions of "model_year" and "model_first_planning_year".

    Raises
    ------
    ValueError
        model_periods is not a series of tuples
    ValueError
        model_periods tuples are not all length 2
    ValueError
        model_year and model_first_planning_year must all be integer
    KeyError
        None of the required keys found
    ValueError
        The model year from scenario definitions is not in the settings
    """
    if "model_periods" in _settings:
        if not all(
            [
                isinstance(t, (tuple, list))
                for t in make_iterable(_settings["model_periods"])
            ]
        ):
            raise ValueError(
                "The settings parameter 'model_periods' must be a list of tuples or lists. "
                f"It is currently {_settings['model_periods']}"
            )
        if not all(len(t) == 2 for t in make_iterable(_settings["model_periods"])):
            raise ValueError(
                "The tuples in settings parameter 'model_periods' must all be 2 years. "
                f"The values found are {_settings['model_periods']}"
            )
        model_planning_period_dict = {
            year: (start_year, year)
            for (start_year, year) in make_iterable(_settings["model_periods"])
        }
    elif "model_year" in _settings and "model_first_planning_year" in _settings:
        model_year = make_iterable(_settings["model_year"])
        first_planning_year = make_iterable(_settings["model_first_planning_year"])
        if not all(isinstance(y, int) for y in model_year) and all(
            isinstance(y, int) for y in first_planning_year
        ):
            raise ValueError(
                "Both 'model_year' and 'model_first_planning_year' parameters must be "
                f"integers or lists of integers. The values found are {model_year} and "
                f"{first_planning_year}."
            )
        model_planning_period_dict = {
            year: (start_year, year)
            for year, start_year in zip(
                make_iterable(_settings["model_year"]),
                make_iterable(_settings["model_first_planning_year"]),
            )
        }
    elif "model_first_planning_year" in _settings:
        # we also allow leaving out the model_year tag and just specifying
        # model_first_planning_year
        model_planning_period_dict = {
            year: (
                _settings["model_first_planning_year"],
                _settings["model_first_planning_year"],
            )
        }
    else:
        raise KeyError(
            "To build a dictionary of scenario settings your settings file should include "
            "either the key 'model_periods' (a list of 2-element lists) or the keys "
            "'model_year' and 'model_first_planning_year' (each a list of years)."
        )

    # remove any model period data already there
    for key in ["model_periods", "model_year", "model_first_planning_year"]:
        try:
            del _settings[key]
        except KeyError:
            pass

    if year not in model_planning_period_dict:
        raise ValueError(
            f"The year {year} is in your scenario definition file for case {_settings.get('case_id')} "
            "but was not found in the 'model_year' or 'model_periods' settings parameters. "
            "Either it is missing in the main settings file or was removed in the "
            "'settings_management' section."
        )
    # assign the scalar values
    _settings["model_first_planning_year"] = model_planning_period_dict[year][0]
    _settings["model_year"] = model_planning_period_dict[year][1]

    return _settings


_YEAR_KEYED_CATCH_ALL_KEYS = frozenset({"default"})


def _is_year_keyed_dict(d: dict) -> bool:
    """Return True if *d* looks like a year-keyed settings dict.

    A dict is considered year-keyed when every key is either:

    - An integer in the range [1900, 2200], or
    - The special string key ``"default"``

    AND at least one integer year key is present.

    Parameters
    ----------
    d : dict
        Dictionary to test.

    Returns
    -------
    bool
        True when *d* qualifies as a year-keyed dict.
    """
    if not d:
        return False
    has_year_key = False
    for k in d.keys():
        if isinstance(k, int) and 1900 <= k <= 2200:
            has_year_key = True
        elif k in _YEAR_KEYED_CATCH_ALL_KEYS:
            pass  # special fallback key – OK
        else:
            return False
    return has_year_key


def _select_year_value(year_dict: dict, year: int, key_name: Optional[str] = None):
    """Return the value from a year-keyed dict for *year*.

    Selection priority:
    1. Exact integer year match.
    2. Special key ``"default"`` as a fallback value.

    If neither condition is met a :class:`ValueError` is raised.  Callers
    should ensure that all required planning years are covered (use
    ``_validate_year_coverage`` when the full set of planning years is known).

    Parameters
    ----------
    year_dict : dict
        Dictionary whose keys are integer years and/or ``"default"``.
    year : int
        Target planning year.
    key_name : str, optional
        Settings key name used in error messages.

    Returns
    -------
    any
        Value from *year_dict* selected for *year*.

    Raises
    ------
    ValueError
        If *year* is not found and no ``"default"`` key exists.
    """
    if year in year_dict:
        return year_dict[year]

    if "default" in year_dict:
        return year_dict["default"]

    available = sorted(k for k in year_dict if isinstance(k, int))
    key_info = f" for key '{key_name}'" if key_name else ""
    raise ValueError(
        f"Year {year}{key_info} not found in year-keyed dictionary. "
        f"Available years: {available}. "
        "Either add an explicit entry for this year or use the 'default' key "
        "to apply the same value to uncovered planning years."
    )


def _validate_year_coverage(
    year_dict: dict, all_years, key_name: Optional[str] = None
) -> None:
    """Raise :class:`ValueError` if *year_dict* does not cover every year in *all_years*.

    Coverage is satisfied when:

    - A ``"default"`` key is present, OR
    - Every year in *all_years* appears as an explicit integer key.

    Partial coverage (some but not all years present) raises an error.

    Parameters
    ----------
    year_dict : dict
        A year-keyed dictionary (already validated by :func:`_is_year_keyed_dict`).
    all_years : iterable of int
        The full set of planning years that must be covered.
    key_name : str, optional
        Settings key name used in error messages.

    Raises
    ------
    ValueError
        If coverage is incomplete and no ``"default"`` key is present.
    """
    if "default" in year_dict:
        return  # fully covered by catch-all

    missing = sorted(y for y in all_years if y not in year_dict)
    if missing:
        key_info = f" for key '{key_name}'" if key_name else ""
        available = sorted(k for k in year_dict if isinstance(k, int))
        raise ValueError(
            f"Year-keyed setting{key_info} is missing values for planning "
            f"year(s): {missing}. "
            f"Available years: {available}. "
            "Either add explicit entries for all planning years or use the "
            "'default' key to apply the same value to uncovered planning years."
        )


def simplify_settings_by_year(
    settings: dict,
    year: int,
    all_years: Optional[Iterable[int]] = None,
    _skip_keys: Optional[FrozenSet] = None,
) -> dict:
    """Replace year-keyed values in a settings dict with the value for *year*.

    Any value that is a dictionary whose every key is either an integer in the
    range [1900, 2200] or the special string ``"default"``
    (with at least one integer key) is treated as a *year-keyed dict* and
    resolved to the single value appropriate for *year*.  All other
    dictionaries are traversed recursively.  Non-dict values are returned
    unchanged.

    This allows settings parameters to vary across planning periods without
    requiring a full ``settings_management`` / scenario-definitions setup.

    Parameters
    ----------
    settings : dict
        Settings dictionary to process (not modified in place).
    year : int
        Target planning year used to select values from year-keyed dicts.
    all_years : iterable of int, optional
        The complete set of planning years for the current run.  When
        provided, every year-keyed dict is validated to ensure it contains
        an entry for every planning year (or uses ``"default"``).
        If a year-keyed dict covers only *some* years and lacks a catch-all
        key, a :class:`ValueError` is raised.
    _skip_keys : frozenset, optional
        Top-level keys that are skipped entirely during traversal.  Defaults
        to ``frozenset({"settings_management"})``.

    Returns
    -------
    dict
        New settings dictionary with all year-keyed values resolved.

    Raises
    ------
    ValueError
        If a year-keyed dict is missing a value for *year* (and no
        ``"default"`` catch-all is present).
    ValueError
        If *all_years* is provided and a year-keyed dict is missing entries
        for one or more planning years.

    Examples
    --------
    >>> settings = {
    ...     "resource_modifiers": {
    ...         "batteries": {
    ...             "technology": "Battery",
    ...             "tech_detail": "Lithium Ion",
    ...             "Var_OM_Cost_per_MWh": {2030: ["add", 0.15], 2040: ["add", 0.10]},
    ...         }
    ...     }
    ... }
    >>> result = simplify_settings_by_year(settings, 2030, all_years=[2030, 2040])
    >>> result["resource_modifiers"]["batteries"]["Var_OM_Cost_per_MWh"]
    ['add', 0.15]

    Notes
    -----
    The ``settings_management`` key is skipped by default to avoid interfering
    with scenario management logic.
    """
    if _skip_keys is None:
        _skip_keys = frozenset({"settings_management"})

    result = {}
    for key, value in settings.items():
        if key in _skip_keys:
            result[key] = value
        elif isinstance(value, dict):
            if _is_year_keyed_dict(value):
                if all_years is not None:
                    _validate_year_coverage(value, all_years, key)
                result[key] = _select_year_value(value, year, key)
            else:
                result[key] = simplify_settings_by_year(
                    value, year, all_years, _skip_keys
                )
        else:
            result[key] = value
    return result


def resolve_settings_to_year(
    settings: Union[dict, Settings],
    year: int,
    all_years: Optional[Iterable[int]] = None,
) -> dict:
    """
    Resolve a multi-year settings object into a single-year settings dictionary.

    This is the **default entry point** for per-year settings processing when you
    are running a single scenario (no scenario definitions file).  It applies
    all year-resolution steps in the correct order and returns a settings dict
    that is fully specialised for *year*:

    1. ``assign_model_planning_years`` — converts list-valued ``model_year`` /
       ``model_first_planning_year`` to scalars for this year.
    2. ``simplify_settings_by_year`` — recursively resolves year-keyed dicts
       (keys are ints 1900-2200 or ``"default"``).
    3. ``expand_capacity_reserve_values`` — expands capacity-reserve shorthand
       into ``regional_tag_values``.
    4. ``add_model_tags_to_gen_columns`` — (conditional) adds resource-tag
       column names to ``generator_columns`` when that key is present.

    Parameters
    ----------
    settings : Union[dict, Settings]
        The full, multi-year settings object.  A ``Settings`` instance is
        automatically converted to a plain ``dict`` before processing.
    year : int
        The planning year to resolve settings for.
    all_years : Iterable[int], optional
        All planning years in the study.  Used to validate that year-keyed
        dicts cover every year.  When *None*, only *year* itself is used for
        validation (safe for single-year studies).

    Returns
    -------
    dict
        A new settings dictionary fully resolved for *year*.  The input is
        deep-copied, so the original is never modified.

    Examples
    --------
    Single-year usage (the common case — no scenario file needed)::

        settings = Settings(config_path="my_settings/")
        year_settings = resolve_settings_to_year(settings, year=2030)

    Multi-year usage::

        settings = Settings(config_path="my_settings/")
        model_years = [2030, 2040]
        year_settings = {
            year: resolve_settings_to_year(settings, year, all_years=model_years)
            for year in model_years
        }

    See Also
    --------
    build_scenario_settings : Builds the full ``{year: {case_id: settings}}``
        structure when a scenario definitions file is used.
    """
    if isinstance(settings, Settings) and hasattr(settings, "to_dict"):
        settings = settings.to_dict()

    if all_years is None:
        all_years = {year}
    else:
        all_years = set(all_years)

    _settings = copy.deepcopy(settings)

    _settings = assign_model_planning_years(_settings, year)

    _settings = simplify_settings_by_year(_settings, year, all_years=all_years)

    _settings = expand_capacity_reserve_values(_settings)

    if _settings.get("generator_columns"):
        _settings["generator_columns"] = add_model_tags_to_gen_columns(
            model_tag_values=_settings.get("model_tag_values", {}),
            regional_tag_values=_settings.get("regional_tag_values", {}),
            generator_columns=_settings["generator_columns"],
        )

    return _settings


def build_scenario_settings(
    settings: Union[dict, Settings], scenario_definitions: pd.DataFrame
) -> Dict[int, Dict[Union[int, str], dict]]:
    """
    Build a nested dictionary of settings for each planning year/scenario.

    Parameters
    ----------
    settings : Union[dict, Settings]
        The full settings file (as dict or Settings object), including the
        "settings_management" section with alternate values for each scenario.
    scenario_definitions : pd.DataFrame
        DataFrame from the CSV file defined in the settings file
        "scenario_definitions_fn" parameter. Must have columns:
        - case_id: Unique identifier for each case
        - year: Planning year for the case
        - Additional columns corresponding to categories in settings_management

    Returns
    -------
    dict
        A nested dictionary with structure:
        {year: {case_id: settings_dict}}
        where each settings_dict contains the complete settings for that case/year.

    Raises
    ------
    ValueError
        If duplicate case/year combinations exist in scenario_definitions.
        If conflicting settings are defined for the same parameter.
    """

    # Convert Settings object to dict if needed
    if isinstance(settings, Settings) and hasattr(settings, "to_dict"):
        settings = settings.to_dict()

    # don't allow duplicate rows in the scenario definitions table, since they
    # could give unexpected results
    dups = scenario_definitions[["case_id", "year"]].duplicated()
    if dups.sum() > 0:
        raise ValueError(
            "The following cases and years are repeated in your scenario definitions file:\n\n"
            + scenario_definitions[dups].to_string(index=False)
        )

    # collect all unique planning years so we can validate year-keyed dicts
    all_planning_years = set(scenario_definitions["year"].unique())

    all_category_levels = set()
    active_category_levels = set()
    scenario_settings = {}
    missing_flag = object()
    case_period = {c: 1 for c in scenario_definitions["case_id"].unique()}
    for i, scenario_row in scenario_definitions.iterrows():
        year, case_id = scenario_row[["year", "case_id"]]

        _settings = copy.deepcopy(settings)
        _settings["case_id"] = case_id
        _settings["case_period"] = case_period[case_id]
        case_period[case_id] += 1

        # first apply any settings under "all_years", then any settings for this year
        for settings_year in ["all_years", year]:

            planning_year_settings_management = (
                settings.get("settings_management", {}).get(settings_year) or {}
            )

            # update settings from all_cases entry if available (these settings
            # are applied to all cases for this year, and don't use the category
            # names or levels from the scenario definitions table)
            if "all_cases" in planning_year_settings_management:
                new_parameter = planning_year_settings_management["all_cases"]
                _settings = update_dictionary(_settings, new_parameter)

            modified_settings = {}
            for category, level in scenario_row.drop(["case_id", "year"]).items():
                # category is a column from the scenario definitions table, e.g. ccs_capex
                # level is the selection for this category for this case/year, e.g., "mid" or "none"

                new_parameter = planning_year_settings_management.get(category, {}).get(
                    level, missing_flag
                )

                # Remember category/levels that were selected and that actually
                # had an effect.
                all_category_levels.add((case_id, year, category, level))
                if new_parameter is not missing_flag:
                    # note: user could set None or {} as the setting, to indicate
                    # this flag should use the default settings as-is
                    active_category_levels.add((case_id, year, category, level))
                if new_parameter in [missing_flag, None, {}]:
                    continue

                _settings = update_dictionary(_settings, new_parameter)

                # report any conflicts between these settings and previous ones
                for key in flatten(new_parameter).keys():
                    if key in modified_settings:
                        raise ValueError(
                            f"The setting {key} is modified by both the "
                            f"`{modified_settings[key]}` flag and the "
                            f"`{category}={level}` flag in the scenario "
                            f"definition for case {case_id}, {year}."
                        )
                    else:
                        # remember this setting for later
                        modified_settings[key] = f"{category}={level}"

        # resolve year-keyed parameters, expand capacity reserves, etc.
        _settings = resolve_settings_to_year(
            _settings, year, all_years=all_planning_years
        )

        scenario_settings.setdefault(year, {})[case_id] = _settings

    # Report any settings in the scenario definitions that had no effect. Values
    # can be changed via either the "all_years" key or a specific year, so we
    # have to wait till the end to decide which tags had no effect.
    missing_category_levels = all_category_levels - active_category_levels
    if missing_category_levels:
        missing = (
            pd.DataFrame(
                missing_category_levels,
                columns=["case_id", "year", "category", "level"],
            )
            .pivot(index=["case_id", "year"], columns="category", values="level")
            .fillna("")
            .reset_index()
        )
        logger.warning(
            "The following parameter value(s) in your scenario definitions file "
            "are not included in the 'settings_management' dictionary for the "
            "specified year(s). Settings will not be modified to reflect these "
            "entries:\n\n"
            + missing.to_string(index=False)
            + "\n\nYou can place empty entries (~) for these in the "
            "settings_management dictionary to avoid this message.\n"
        )

    return scenario_settings


def auto_fill_settings(**setting_mappings: str):
    """
    Decorator that automatically fills function arguments with values from Settings
    when None is passed or argument is not provided.

    Parameters
    ----------
    **setting_mappings : str
        Keyword arguments mapping function parameter names to settings keys.
        If a parameter name matches a settings key exactly, you can omit the mapping.

    Examples
    --------
    # Direct mapping (parameter name = settings key)
    @auto_fill_settings()
    def my_function(model_regions=None, model_year=None):
        pass

    # Custom mapping
    @auto_fill_settings(regions='model_regions', year='model_year')
    def my_function(regions=None, year=None):
        pass

    # Mixed approach
    @auto_fill_settings(pg_table='load_source_table_name')
    def my_function(settings=None, model_year=None, pg_table=None):
        pass
    """

    def decorator(func: Callable) -> Callable:
        func_sig = signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get current settings
            try:
                current_settings = get_current_settings()
            except RuntimeError:
                # No settings available, proceed with original function
                return func(*args, **kwargs)

            # Build a dictionary of all arguments (positional + keyword)
            bound_args = func_sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()

            # Fill in missing arguments from settings
            for param_name, param in func_sig.parameters.items():
                # Skip if argument was explicitly provided and is not None
                if (
                    param_name in bound_args.arguments
                    and bound_args.arguments[param_name] is not None
                ):
                    continue

                # Determine settings key to look up
                if param_name in setting_mappings:
                    settings_key = setting_mappings[param_name]
                elif param_name == "settings":
                    # Special case: pass the entire settings object
                    bound_args.arguments[param_name] = current_settings
                    continue
                else:
                    # Try to use parameter name as settings key
                    settings_key = param_name

                # Get value from settings if available
                settings_value = current_settings.get(settings_key)
                if settings_value is not None:
                    bound_args.arguments[param_name] = settings_value

            return func(*bound_args.args, **bound_args.kwargs)

        return wrapper

    return decorator
