# Working with Renewable Resources

Turn the bundled test system's two wind cost tiers per region into four
cost-and-profile clusters, then verify that the change preserves available
capacity and writes matching generation profiles.

**Time:** about 20 minutes.
**Prerequisites:** complete the source installation in
[Getting Started](getting-started.md), activate that Python environment, and
have a matching checkout containing `tests/test_system/`. No external renewable
data download is needed.

This exercise uses the existing `p1_2`, `p3`, and `p4` geography and only bundled
land-based wind data. It generates GenX inputs; it does not run a capacity
expansion optimization or demonstrate calibrated wind economics. Solar in the
existing-generator table is not a new solar resource-group dataset.

## Step 1: Copy the bundled study

Use a new destination outside the source checkout so that edits, caches, and
outputs do not change the test fixture:

```bash
cp -R /path/to/PowerGenome/tests/test_system "$HOME/powergenome-wind-study"
cd "$HOME/powergenome-wind-study"
```

Replace `/path/to/PowerGenome` with your checkout path. The destination should
not already exist. Run every following command from this study directory.
Save the Python snippets as the named scripts inside this copied study and run
them with your activated environment.

Leave `settings/model_definition.yml` unchanged:

```yaml
model_regions: [p1_2, p3, p4]
region_aggregations:
  p1_2: [p1, p2]
```

The resource metadata already uses `p1_2`; PowerGenome includes both model-region
and aggregated base-region names when selecting sites. Do not rename the
metadata regions or introduce unrelated example geographies.

## Step 2: Inspect the actual wind inputs

Open `settings/data.yml`. It already points to the copied files:

```yaml
RESOURCE_GROUPS: test_data/resource_groups
data_location: test_data
```

The uppercase resource path resolves from your **current working directory**,
which is why you changed into the study. `data_location` and the existing
`input_folder: extra_inputs` resolve relative to the study containing `settings/`.

Save this as `inspect_wind.py` to inspect the definition and eligible supply:

```python
import json
from pathlib import Path

import pandas as pd

root = Path("test_data/resource_groups")
definition = json.loads((root / "onshorewind_group.json").read_text())
print(definition)
sites = pd.read_csv(root / definition["metadata"])
print(sites.columns.tolist())
eligible = sites.loc[(sites["capacity_mw"] > 0) & (sites["lcoe"] <= 75)]
print(eligible.groupby("metro_region")["capacity_mw"].sum().round(1))
```

```bash
python inspect_wind.py
```

The JSON contains these real filenames:

```json
{
  "technology": "landbasedwind",
  "metadata": "onshorewind_lcoe_ReEDS.csv",
  "profiles": "onshorewind_profiles_test_data_tidy.csv",
  "site_map": "site_map_test_data.csv"
}
```

The metadata's `CPA_ID` identifies candidate sites. `metro_region` identifies
location, `capacity_mw` is available MW, `cf` is a fractional capacity factor,
and `lcoe` is the input cost-screening feature. `interconnect_annuity` and
`interconnect_capex_mw` supply site-specific interconnection costs. PowerGenome
normalizes column names, so `CPA_ID` is available internally as `cpa_id`.

The profiles are tidy `weather_year,time_index,site_id,value` rows. The site map
connects metadata `CPA_ID` to profile `Site` IDs; keep it alongside the profile
file. The fixture has short profiles for 2012 and 2013, not two complete years.

The eligible supply is approximately:

| Region | Wind capacity (MW) |
| --- | ---: |
| `p1_2` | 54,045.6 |
| `p3` | 26,375.9 |
| `p4` | 24,066.3 |

The `lcoe <= 75` screen is an existing **test-system assumption**, not an economic
recommendation. Keep it unchanged here so you can isolate the effect of
aggregation. A different screening threshold changes candidate supply, not just
model size.

## Step 3: Run the unchanged baseline

In `settings/resources.yml`, confirm the existing `new_resources` list includes:

```yaml
new_resources:
  # Other bundled technologies remain in this list.
  - [LandbasedWind, Class3, Moderate, 1]
```

This is an excerpt, not a replacement list. Its four values are technology,
technology detail, cost case, and unit size in MW. Do not add solar or offshore
wind: their new-build resource groups are not bundled with this study.

The existing renewable definition filters at 75 and creates two capacity-weighted
LCOE quantile bins. With no additional clustering, each nonempty bin becomes one
new-build resource: two wind resources per region.

Run the baseline:

```bash
validate_powergenome --settings_file settings
run_powergenome --settings_file settings --results_folder wind_baseline
```

Validation and generation can finish with warnings about the bundled study,
including missing uranium/distillate fuel prices, unassigned resource tags, and
missing output columns for other resource types. These are not evidence of a
calibrated study; resolve them before using a real study for decisions. Stop if
you get an error rather than continuing with incomplete outputs.

The two planning periods are written under
`wind_baseline/Inputs/Inputs_p1/` (2030) and `Inputs_p2/` (2040).

## Step 4: Split each cost tier by generation profile

Edit the copied `settings/resources.yml`. Replace its existing
`renewables_clusters` list with:

```yaml
renewables_clusters:
  - region: all
    technology: landbasedwind
    filter:
      - feature: lcoe
        max: 75
    bin:
      - feature: lcoe
        q: 2
        weights: capacity_mw
    cluster:
      - feature: profile
        method: agglomerative
        n_clusters: 2
```

Only the `cluster` block is new. Sites are still screened and divided into the
same two LCOE tiers within each region; now each tier is split into two groups
with similar output timing. Profiles and supported cost columns are
capacity-weighted within each final cluster, and capacities are summed.

Keep both existing cache controls `false` for this exercise. Leave all other
technologies and both planning periods unchanged. With no `weather_year`
selection in the baseline, both bundled profile years are loaded in each run.

Edit the existing definition, **not** a new override file. A settings directory
loads only immediate `.yml` files with shallow top-level replacement and no
reliable ordering; another `renewables_clusters` key could replace the whole
list unexpectedly.

Run the modified study into a separate output folder:

```bash
validate_powergenome --settings_file settings
run_powergenome --settings_file settings --results_folder wind_profiles
```

## Step 5: Compare the outputs

Use `resources/Vre.csv`, not the legacy flat generator filename. The file also
contains existing wind and solar; select new-build land-based wind explicitly.
Save this complete check as `check_wind.py` in the study directory:

```python
from pathlib import Path

import pandas as pd


def new_wind(folder, period):
    path = Path(folder) / "Inputs" / f"Inputs_p{period}" / "resources/Vre.csv"
    resources = pd.read_csv(path)
    mask = resources["technology"].str.contains("landbasedwind", case=False, na=False)
    return resources.loc[mask & resources["New_Build"].eq(1)].copy()


sites = pd.read_csv("test_data/resource_groups/onshorewind_lcoe_ReEDS.csv")
eligible = sites.loc[(sites["capacity_mw"] > 0) & sites["lcoe"].le(75)]
expected_mw = eligible.groupby("metro_region")["capacity_mw"].sum().sort_index()

for period in (1, 2):
    before = new_wind("wind_baseline", period)
    after = new_wind("wind_profiles", period)
    counts = pd.DataFrame({
        "baseline": before.groupby("region").size(),
        "profile_clusters": after.groupby("region").size(),
    }).sort_index()
    assert counts.index.tolist() == ["p1_2", "p3", "p4"]
    assert counts["baseline"].eq(2).all()
    assert counts["profile_clusters"].eq(4).all()
    print(f"Period {period} resource counts:\n{counts}")

    for wind in (before, after):
        actual_mw = wind.groupby("region")["Max_Cap_MW"].sum().sort_index()
        # CSV output rounds capacities, so allow 1 MW per region.
        pd.testing.assert_series_equal(
            actual_mw, expected_mw, check_names=False,
            check_exact=False, rtol=0, atol=1,
        )

    period_dir = Path("wind_profiles/Inputs") / f"Inputs_p{period}"
    profiles = pd.read_csv(period_dir / "system/Generators_variability.csv")
    assert set(after["Resource"]).issubset(profiles.columns)
    wind_profiles = profiles.loc[:, after["Resource"].tolist()]
    assert wind_profiles.notna().all().all()
    assert wind_profiles.ge(0).all().all() and wind_profiles.le(1).all().all()
    print("Profile rows:", len(wind_profiles))
    print(after.groupby("region")["Inv_Cost_per_MWyr"].agg(["min", "max"]))
```

```bash
python check_wind.py
```

Expect **6 → 12 new wind resources** in each period, with the same total
available capacity in each region (apart from CSV rounding). The fixture writes
336 modeled profile rows when both short weather-year samples are loaded.
`Inv_Cost_per_MWyr` is an annual investment cost in $/MW-year; it is not the
screening metadata's `lcoe`.

A mean of these short sample profiles is not an annual capacity factor.

To inspect which sites were combined, open the three
`*site_cluster_assignments.csv` files under each period's `extra_outputs/`.
They contain `cpa_id` and `cluster`. Join `cpa_id` to the input metadata's
`CPA_ID` to inspect capacity and LCOE ranges; cluster labels need not stay
identical when you change settings.

## What you changed

You held geography, eligible supply, underlying technology costs, and time
settings fixed, and added temporal differentiation **within** each cost tier.
Cluster costs can change because different sites are averaged. More resources
are not automatically more accurate: this check establishes that the software
change behaved as intended, not that four clusters is the right study resolution.

For further work, compare model outcomes at different resolutions. Use
[Configure Renewable Resource Clusters](../how-to/configure-renewable-clusters.md)
for input schemas, method choices, caching, regional definitions, and optional
cost modifiers. See [New-Build Resources](../reference/settings/new-build.md) for
technology settings and [Output Format](../explanation/output-format.md) for
output layouts.
