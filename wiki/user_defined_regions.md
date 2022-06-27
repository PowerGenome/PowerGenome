# Configuring user defined regions

By default, PowerGenome uses [IPM Regions](https://github.com/PowerGenome/PowerGenome/wiki/Geospatial-Mappings#ipm-regions) to aggregate data.
However, a user can use custom regions by supplying the extra files and parameters listed below.

## Input Files

### `user_region_geodata_fn`

A geojson or shapefile with the boundaries of each custom region.
The name of each region is stored as a property in the geojson/shapefile under the key "region".
While a CRS (Coordinate Reference System) is needed, a specific CRS is not required.

```json
{
  "type": "FeatureCollection",
  "crs": {
    "type": "name",
    "properties": {
      "name": "..."
    }
  },
  "features": [
    {
      "type": "Feature",
      "properties": {
        "name": "user_region_a",
        "geometry": {
          "type": "Polygon",
          "coordiantes": [[[]]]
        }
      }
    },
    {
      "type": "Feature",
      "properties": {
        "name": "user_region_b",
        "geometry": {
          "type": "Polygon",
          "coordiantes": [[[]]]
        }
      }
    }
  ]
}
```

### `plant_region_map_fn`

A CSV file that maps user supplied regions to generators listed in the PUDL database or [EIA 860m](https://www.eia.gov/electricity/data/eia860m/) (stored in `PowerGenome/data/eia/860m/{month}_generator{year}.xlsx`).
The columns of the user supplied CSV file should be "plant_id_eia" and "region".
Map applicable Generators' EIA plant ID (called "Plant ID" in the EIA860m) to the corresponding region.

| plant_id_eia | region |
| -------------|--------|
|123           |user_region_a|
|456           |user_region_a|
|654           |user_region_b|
|321           |user_region_b|

### `user_regional_load_fn`

A CSV file that describes the hourly user_region_al load for each region.
Each column is the timeseries load for each combination of model year, electrification scenario and region.
The first row is the year corresponding to the column's load.
The second row is the "electrification scenario", which corresponds to the "electrification" column in `scenario_definitions_fn`.
(This allows the user to choose different loads profile for each scenario.)
The third row is the region name, and the subsequent 8760 rows are the load values for that year, electrification scenario, and region.

|2030|2030|2030|2030|2040|2040|2040|2040|
|----|----|----|----|----|----|----|----|
|electrification1|electrification1|electrification2|electrification2|electrification1|electrification1|electrification2|electrification2|
|user_region_a|user_region_b|user_region_a|user_region_b|user_region_a|user_region_b|user_region_a|user_region_b|
|load_t0|load_t0|load_t0|load_t0|load_t0|load_t0|load_t0|load_t0|
|...|...|...|...|...|...|...|...|
|load_t8759|load_t8759|load_t8759|load_t8759|load_t8759|load_t8759|load_t8759|load_t8759|

### `user_transmission_constraints_fn`

A CSV file that describes the transmission constraints between regions.
The columns are "region_from", "region_to" and "nonfirm_ttc_mw" (transmission capacity).
Transmission line distances are calculated by finding the distance between the centroids of each region.

|region_from|region_to|nonfirm_ttc_mw|
|---|---|---|
|user_region_a|user_region_b|1234|

## Settings

User defined regions must be incorporated into the settings file used for a run of PowerGenome.

- `model_regions`

  - List the desired user regions. These model regions can also be aggregated in `region_aggregations`.

- `distributed_gen_method`/`distributed_gen_values`

  - Specify distributed generation parameters.
    Distributed generation profiles for each user region can be described in `distributed_gen_profiles_fn`.

- `transmission_investment_cost`

  - Transmission financial information can be specified for each region under this key.

- `cost_multiplier_region_map`

  - Map the user regions to [AEO EMM regions](https://github.com/PowerGenome/PowerGenome/wiki/Geospatial-Mappings#nems-emm-regions) for cost multipliers.
  Use the names used by EIA's Open Data API (listed in PowerGenome settings file).
  - Alternate cost multipliers can be used by adjusting/making a new `cost_multiplier_fn` file in the [data/cost_multipliers/](https://github.com/PowerGenome/PowerGenome/tree/master/data/cost_multipliers) directory.

- `historical_load_region_map`/`future_load_region_map`

  - Map user defined regions to load regions, if applicable.
    See PowerGenome settings file for valid regions.

- `aeo_fuel_region_map`
  - Map user defined regions to AEO fuel regions.
    See PowerGenome settings file for valid regions.
