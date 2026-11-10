# Working with Renewable Resources

Aggregate the published Texas example's individual wind candidates into four
cost-and-profile clusters per model region, then verify that the change
preserves available capacity and writes matching generation profiles.

**Time:** about 20 minutes.
**Prerequisites:** complete [Getting Started](getting-started.md), activate that
Python environment, and work from `powergenome-tutorial/`. The downloaded
example includes all required wind and solar data; no extra download is needed.

This exercise keeps the `TX1`, `TX2`, and `TX3` geography and the existing solar
configuration. It generates GenX inputs; it does not run an optimization or
demonstrate calibrated renewable economics.

## Step 1: Copy the settings

From `powergenome-tutorial/`, copy the Texas settings to a new sibling directory.
Choose an unused destination; do not overwrite an earlier exercise:

```bash
cp -R example_systems/TX-3-zone/settings example_systems/TX-3-zone/settings_wind
```

The two settings folders share the same parent, so their relative `data_location`
and `input_folder` still point to the downloaded data and extra inputs. The
resource-group paths also remain valid because you stay at the examples root.
Save the Python snippets below as the named scripts in `powergenome-tutorial/`
and run them with your activated environment.

Leave `settings_wind/model_definition.yml` unchanged:

```yaml
model_regions: [TX1, TX2, TX3]
region_aggregations:
  TX1: [p60, p61, p62]
  TX2: [p63, p64]
  TX3: [p65, p67]
```

The metadata uses the seven base-region names. PowerGenome selects sites from
each model region's base regions; do not rename metadata regions.

## Step 2: Inspect the actual wind inputs

Open `example_systems/TX-3-zone/settings_wind/data.yml`. It already contains:

```yaml
RESOURCE_GROUPS: example_systems/TX-3-zone/resource_groups
RESOURCE_GROUP_PROFILES: example_systems/TX-3-zone/resource_profiles
data_location: data
```

The uppercase paths resolve from your **current working directory**,
`powergenome-tutorial/`. `data_location` and `input_folder: extra_inputs` resolve
relative to the parent of `settings_wind/`, `example_systems/TX-3-zone/`.

Save this as `inspect_wind.py` to inspect the definition and eligible supply:

```python
import json
from pathlib import Path

import pandas as pd
import yaml

study = Path("example_systems/TX-3-zone")
root = study / "resource_groups"
definition = json.loads((root / "onshorewind_group.json").read_text())
print(definition)
sites = pd.read_csv(root / definition["metadata"])
print(sites.columns.tolist())
settings = yaml.safe_load((study / "settings_wind/model_definition.yml").read_text())
region_map = {
    base: region
    for region, bases in settings["region_aggregations"].items()
    for base in bases
}
sites["model_region"] = sites["region"].map(region_map)
assert sites["model_region"].notna().all()
eligible = sites.loc[(sites["capacity_mw"] > 0) & (sites["lcoe"] <= 75)]
print(eligible.groupby("model_region")["capacity_mw"].sum().round(1))
```

```bash
python inspect_wind.py
```

The JSON contains these filenames:

```json
{
  "technology": "landbasedwind",
  "metadata": "onshorewind_lcoe_tx_3_zone.csv",
  "profiles": "onshorewind_profiles_tx_3_zone.csv",
  "site_map": "onshorewind_site_map_tx_3_zone.csv"
}
```

The metadata's `cpa_id` identifies candidate sites, `region` identifies location,
`capacity_mw` is available MW, `cf` is a fractional capacity factor, and `lcoe`
is the input cost-screening feature. `interconnect_annuity` and
`interconnect_capex_mw` supply site-specific interconnection costs.

The profiles are tidy `site_id,time_index,value,weather_year` rows. The site map
connects metadata `cpa_id` to `profile_id`; it lives alongside the profiles in
`resource_profiles/`. The example has 240 synthetic time steps for weather year
2012, not a complete annual profile.

The eligible supply is approximately:

| Region | Wind candidates | Wind capacity (MW) |
| --- | ---: | ---: |
| `TX1` | 30 | 241,322.8 |
| `TX2` | 20 | 88,167.2 |
| `TX3` | 20 | 83,964.6 |

The `lcoe <= 75` screen is an existing **example assumption**, not an economic
recommendation. Keep it unchanged here so you can isolate the effect of
aggregation. A different screening threshold changes candidate supply, not just
model size.

## Step 3: Run the baseline

In `example_systems/TX-3-zone/settings_wind/resources.yml`, confirm that the
existing `new_resources` list includes:

```yaml
new_resources:
  # Other example technologies remain in this list.
  - [LandbasedWind, Class3, Moderate, 1]
  - [UtilityPV, Class1, Moderate, 1]
```

This is an excerpt, not a replacement list. Its four values are technology,
technology detail, cost case, and unit size in MW. Keep the solar entry and all
other technologies unchanged.

The published `renewables_clusters` list screens wind at 75 and solar at 60,
with no binning or clustering. Each retained candidate becomes a new-build
resource: 30 wind resources in `TX1` and 20 each in `TX2` and `TX3`.

```bash
validate_powergenome --settings_file example_systems/TX-3-zone/settings_wind
run_powergenome \
  --settings_file example_systems/TX-3-zone/settings_wind \
  --results_folder wind_baseline
```

Review warnings about resource tags and missing output columns. Stop if you get
an error rather than continuing with incomplete outputs. A completed run is not
evidence of a calibrated study.

The two planning periods are written under
`wind_baseline/base/Inputs/Inputs_p1/` (2030) and `Inputs_p2/` (2045).

## Step 4: Aggregate wind by cost and generation profile

Edit `example_systems/TX-3-zone/settings_wind/resources.yml`. Replace its existing
`renewables_clusters` list with this complete list, **including solar**:

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
  - region: all
    technology: utilitypv
    filter:
      - feature: lcoe
        max: 60
```

The wind `bin` and `cluster` blocks are new. Wind candidates are screened as
before, then divided into two capacity-weighted LCOE tiers within each region.
Each tier is split into two groups with similar output timing. Profiles and
supported cost columns are capacity-weighted within each final cluster, and
capacities are summed. Solar candidates are left unchanged.

Keep both existing cache controls `false` for this exercise. Leave all other
technologies, geography, and planning periods unchanged.

Edit the existing definition, **not** a new override file. A settings directory
loads only immediate `.yml` files with shallow top-level replacement and no
reliable ordering; another `renewables_clusters` key could replace the whole
list unexpectedly.

Run the modified study into a separate output folder:

```bash
validate_powergenome --settings_file example_systems/TX-3-zone/settings_wind
run_powergenome \
  --settings_file example_systems/TX-3-zone/settings_wind \
  --results_folder wind_profiles
```

## Step 5: Compare the outputs

Use `resources/Vre.csv`, not the legacy flat generator filename. Select
new-build land-based wind explicitly, and check that the solar rows are unchanged.
Save this complete check as `check_wind.py` in `powergenome-tutorial/`:

```python
from pathlib import Path

import pandas as pd
import yaml


def new_vre(folder, period, technology):
    path = Path(folder) / "base/Inputs" / f"Inputs_p{period}" / "resources/Vre.csv"
    resources = pd.read_csv(path)
    mask = resources["technology"].str.contains(technology, case=False, na=False)
    return resources.loc[mask & resources["New_Build"].eq(1)].copy()


study = Path("example_systems/TX-3-zone")
sites = pd.read_csv(study / "resource_groups/onshorewind_lcoe_tx_3_zone.csv")
settings = yaml.safe_load((study / "settings_wind/model_definition.yml").read_text())
region_map = {
    base: region
    for region, bases in settings["region_aggregations"].items()
    for base in bases
}
sites["model_region"] = sites["region"].map(region_map)
assert sites["model_region"].notna().all()
eligible = sites.loc[(sites["capacity_mw"] > 0) & sites["lcoe"].le(75)]
expected_mw = eligible.groupby("model_region")["capacity_mw"].sum().sort_index()
expected_counts = eligible.groupby("model_region").size().sort_index()

for period in (1, 2):
    before = new_vre("wind_baseline", period, "landbasedwind")
    after = new_vre("wind_profiles", period, "landbasedwind")
    counts = pd.DataFrame({
        "baseline": before.groupby("region").size(),
        "profile_clusters": after.groupby("region").size(),
    }).sort_index()
    assert counts.index.tolist() == ["TX1", "TX2", "TX3"]
    pd.testing.assert_series_equal(counts["baseline"], expected_counts, check_names=False)
    assert counts["profile_clusters"].eq(4).all()
    print(f"Period {period} resource counts:\n{counts}")

    for wind in (before, after):
        actual_mw = wind.groupby("region")["Max_Cap_MW"].sum().sort_index()
        # CSV output rounds each candidate, so allow 2 MW per region.
        pd.testing.assert_series_equal(
            actual_mw, expected_mw, check_names=False,
            check_exact=False, rtol=0, atol=2,
        )

    solar_before = new_vre("wind_baseline", period, "utilitypv").set_index("Resource")
    solar_after = new_vre("wind_profiles", period, "utilitypv").set_index("Resource")
    assert len(solar_before) == len(solar_after) == 70
    pd.testing.assert_frame_equal(solar_before.sort_index(), solar_after.sort_index())

    period_dir = Path("wind_profiles/base/Inputs") / f"Inputs_p{period}"
    profiles = pd.read_csv(period_dir / "system/Generators_variability.csv")
    assert set(after["Resource"]).issubset(profiles.columns)
    wind_profiles = profiles.loc[:, after["Resource"].tolist()]
    assert len(wind_profiles) == 240
    assert wind_profiles.notna().all().all()
    assert wind_profiles.ge(0).all().all() and wind_profiles.le(1).all().all()
    print("Profile rows:", len(wind_profiles))
    print(after.groupby("region")["Inv_Cost_per_MWyr"].agg(["min", "max"]))
```

```bash
python check_wind.py
```

Expect **70 → 12 new wind resources** in each period, with the same total
available capacity in each region apart from CSV rounding. All 70 new solar
resources remain unchanged. The example writes 240 modeled profile rows.
`Inv_Cost_per_MWyr` is an annual investment cost in $/MW-year; it is not the
screening metadata's `lcoe`.

A mean of these short synthetic profiles is not an annual capacity factor.

To inspect which wind candidates were combined, open the land-based-wind
`*site_cluster_assignments.csv` files under each period's `extra_outputs/`.
They contain `cpa_id` and `cluster`. Join `cpa_id` to the input metadata's
`cpa_id` to inspect capacity and LCOE ranges; cluster labels need not stay
identical when you change settings.

## What you changed

You held geography, eligible supply, underlying technology costs, and solar
configuration fixed while aggregating wind into cost tiers and profile clusters.
Cluster costs can change because different candidates are averaged. Fewer
resources can make optimization faster, but this check establishes software
behavior, not that four clusters is the right study resolution.

For further work, compare model outcomes at different resolutions. Use
[Configure Renewable Resource Clusters](../how-to/configure-renewable-clusters.md)
for input schemas, method choices, caching, regional definitions, and optional
cost modifiers. See [New-Build Resources](../reference/settings/new-build.md) for
technology settings and [Output Format](../explanation/output-format.md) for
output layouts.
