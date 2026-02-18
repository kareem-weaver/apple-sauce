# 2024 Data Challenge – Project Recap + Plan (for Santiago)

**Social Impact Topic:** Increase the **Mastercard Inclusive Growth Score (IGS)** for **strategic locations with IGS ≤ 45**  
**Core deliverables:** input datasets + commented code + combined submission CSV + **≤10 slide** exec-style presentation

---

## 1) What the challenge requires (high level)

We must use **IGS data + public data** to:
1. **Select a low-scoring community / census tracts (≤45)**
2. **Diagnose** gaps across IGS pillars (**Place, Economy, Community**)
3. **Augment** the provided data with additional public datasets (novelty matters)
4. Build an **analytical solution** (EDA + modeling) to identify drivers & forecast improvement
5. Propose **practical interventions** (policy/programs) and a measurement plan
6. Pitch it in a **10-slide** presentation aimed at C‑suite judges

**Constraints:** no paid/exclusive datasets; avoid PII; be careful with sensitive attributes (keep things aggregated at tract level).

---

## 2) Community scope & time window (current)

From the current notebook work, we are operating on **12 census tracts** (example list used in code):  
`4052, 4053.01, 4053.02, 4054.01, 4054.02, 4055, 4056, 4057, 4058, 4059.01, 4059.02, 4060`

Time range in the notebook: **2017–2022** (6 years).  
This yields **72 rows** when data is in *tract-year* format (12 tracts × 6 years).

---

## 3) What we have already completed (from the current Jupyter notebook export)

### A) Demographics metric ✅ (Race + Age/Sex)

**Race pipeline completed**
- Reads yearly CSVs matching: `acs_race_*.csv`
- Cleans:
  - sets `Label (Grouping)` as index
  - removes *Percent* columns
  - transposes so rows become tract observations
  - extracts `Estimate Year` from filename
  - extracts numeric **Tract ID** using regex (keeps `.01`, `.02`)
- Keeps key columns:
  - `Total:`
  - `White alone`
  - `Black or African American alone`
  - `Asian alone`

**Output produced**
- `filtered_race.csv`

**Age/Sex pipeline completed**
- Reads yearly CSVs matching: `acs_age_sex_*.csv`
- Cleans:
  - removes *Percent* columns
  - drops columns containing `Male|Female` (keeps combined totals)
  - transposes; adds `Estimate Year`
  - extracts **Tract ID** similarly
- Keeps key columns:
  - `Total population`
  - `Under 18 years`
  - `Median age (years)`
  - `18 years and over`

**Output produced**
- `final_age_sex_data.csv`

**Merged Demographics completed**
- Merge keys: `['Tract ID', 'Estimate Year']`
- Output produced:
- `merged_demographic_data.csv`

---

### B) Housing metric ⚠️ (in progress)

**Housing occupancy pipeline started**
- Reads yearly CSVs matching: `acs_housing_occupancy_*.csv`
- Sets `Label (Grouping)` as index, transposes, adds `Estimate Year`
- There is a “rows_to_remove” list (categories like race/ethnicity of householder, etc.) — defined but not fully shown as applied.
- The notebook shows column selection is not finalized (example keeps one tract column).

**Median Home Price pipeline completed**
- Reads `acs_median_home_price_*.csv`
- Renames tract columns to a consistent format using a tract list
- Builds a combined table by year
- Produces an in-memory dataframe `filtered_merged_median_price_df` with columns like:
  - `Estimate Year`
  - `Median Value (Dollars)`
  - `Tract 4052 Estimate`, `Tract 4053.01 Estimate`, …

**Housing merge currently errors**
- Attempted to `melt()` an occupancy dataframe named `filtered_merged_occupancy_df`
- Notebook throws:
  - `NameError: name 'filtered_merged_occupancy_df' is not defined`
- So the final merged housing-market dataset is **not yet produced**.

---

### C) Economic Stability ✅/⚠️ (Income done; Employment pending)

**Income stability combined dataset completed**
- Reads `income_stability_{year}.csv` for years 2017–2022
- Fixes a 2022 delimiter/column issue (replaces `;` with `,`)
- Drops non-tract summary rows (families, nonfamily households, etc.)
- Removes `Percent` columns
- Adds `Estimate Year` and saves combined output

**Output produced**
- `income_stability_combined.csv`

**Employment section**
- Present as a header in notebook, but no implementation yet.

---

### D) Inclusive Growth Score (IGS) data ✅ (loaded only)

- Notebook loads `Inclusive_Growth_Score_Data.csv` into `IG_data`
- No cleaning, keying, or merges shown yet

---

## 4) What we still need to do (core tasks)

### 4.1 Fix & complete Housing Metric (high priority)
Goal: a **tract-year** housing table that we can merge into the master dataset.

**To-do:**
- Ensure the housing occupancy dataframe is actually assigned to `filtered_merged_occupancy_df`
  - likely by defining `filtered_merged_occupancy_df = merged_occupancy_df.dropna(...)` or selecting the right columns first
- Reshape occupancy properly:
  - long format: `Estimate Year`, `Tract ID`, `Owner-occupied`, `Renter-occupied`, `Total occupied`
- Reshape median prices:
  - long format: `Estimate Year`, `Tract ID`, `Median Home Price`
- Merge occupancy + median prices on `Estimate Year` + `Tract ID`

**Deliverable output:**
- `merged_housing_market_data.csv` (or similarly named)

---

### 4.2 Build a single “master modeling table”
Goal: one dataset with **one row per tract-year** containing:
- Demographics (already merged)
- Housing (after fix)
- Income stability (needs alignment to tract-year format)
- Employment (to be added)
- IGS outcome/targets (from `IG_data`)

**Deliverable output:**
- `master_tract_year_dataset.csv`

---

### 4.3 Add “novel” public data augmentation (to score points)
Beyond basic Census, we should add at least **1–3** strong augmentations, tract-linked when possible.

High-value augmentation ideas (all public):
- **Education outcomes**
  - NCES school performance / graduation / enrollment (district/school → tract mapping if possible)
  - State/City open-data for reading proficiency (often school/district level)
- **Crime & safety**
  - City/county open-data crime incidents (aggregate to tract)
  - FBI/NIBRS where available (often agency-level; tract mapping may require geo join)
- **Health access/outcomes**
  - CDC PLACES (small area health estimates; often county/tract-ish depending on table)
  - locations of clinics/hospitals (HIFLD, OpenStreetMap) → distance / density metrics
- **Broadband quality**
  - FCC Broadband data / availability maps (tract-level in some releases)
- **Transit access / walkability**
  - GTFS feeds (transit stops density), OpenStreetMap features, commute connectivity proxies
- **Business activity**
  - County Business Patterns (CBP) / BDS (geography varies)
  - local business license open-data (if available)

---

### 4.4 Modeling & analysis plan
We need both **insights** and **forecasted improvement**.

**Modeling options:**
- **Regression** to identify drivers (interpretable)
- **Tree-based model** (Random Forest / XGBoost) for feature importance + non-linear patterns
- **Clustering** to group similar tracts and compare “peer tracts”

**Forecasting interventions:**
- Create “what-if” scenarios (increase broadband %, reduce rent burden %, etc.)
- Predict the resulting change in IGS (or pillars) using the model

---

## 5) Presentation plan (10 slides)
1. Title (team, community, topic)
2. Objective + why selected tract(s)
3. Community overview (demographics + baseline indicators)
4. Benchmarking vs state/national (and/or similar tracts)
5. Key findings (EDA visuals + model insights)
6. Root causes (drivers tied to pillars)
7. Proposed solutions (2–3 targeted interventions)
8. Implementation plan (stakeholders, timeline, cost categories)
9. Predicted outcomes (modeled score lift + assumptions)
10. Metrics for success + social impact conclusion

---

## 6) Santiago’s onboarding “first week” checklist

**Day 1–2: Understand the data + structure**
- Review how we’re building tract-year datasets (2017–2022, 12 tracts)
- Confirm what identifiers exist in `IG_data` (tract id format, year, pillar columns)

**Day 3–4: Fix housing pipeline**
- Resolve `filtered_merged_occupancy_df` NameError
- Produce long-format occupancy + long-format median price
- Merge them and export a clean housing metric CSV

**Day 5–7: Help build the master table**
- Join: demographics + housing + income (+ IGS)
- Validate row counts (should be 72 if all tracts/years match)
- Create 2–3 sanity-check plots (trends over time, missingness, distributions)

---

## 7) Known issues to fix (from notebook)
- `filtered_merged_occupancy_df` is referenced but never defined → breaks housing merge
- Housing occupancy column selection appears incomplete (only one tract column listed)
- Employment section not implemented
- IGS dataset is loaded but not cleaned/merged

---

## 8) Repo / folder structure suggestion (simple)
```
project/
  data_raw/
  data_intermediate/
  data_final/
  notebooks/
  src/
  outputs/
  slides/
  README.md
```

---

## 9) Next “milestone” goals
1. ✅ Demographics metric locked (done)
2. 🟨 Housing metric fixed + exported (next)
3. 🟨 Master dataset assembled (next)
4. 🟨 Add 1–3 augmentation datasets (next)
5. 🟨 Modeling + what‑if forecasts (next)
6. 🟨 Slide deck filled with visuals + narrative (final)

