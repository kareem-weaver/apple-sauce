# AUC Mastercard Challenge

This repository contains the end-to-end workflow used to identify low inclusive-growth communities, shortlist high-priority target areas, build Shelby County deep-dive visuals, and model the projected impact of the proposed **ShelbyFirst** intervention.

The project has four connected parts:

1. Build a tract-year analysis panel by combining Mastercard Inclusive Growth Score data with ACS-derived economic and demographic variables.
2. Construct spatially contiguous tract clusters, score them across time, and shortlist the most structurally distressed communities.
3. Produce presentation-ready Shelby County comparison maps, healthcare/context visuals, demographic overlays, and ZIP overlap files for the selected low-growth community.
4. Simulate a three-year ShelbyFirst intervention pathway and export outcome charts, tables, and interactive HTML views.

## Current project status

- The current processed shortlist places **`Tennessee | Shelby County | cluster_77` at rank `#6` out of `67` shortlisted clusters**.
- The updated Shelby demographic choropleth shows that the selected low-growth community overlaps the darker high-Black-share tract corridor.
- Recovery-period summary from the current processed artifacts:
  - Selected low-growth community: about **77% Black**
  - Rest of Shelby County: about **58% Black**
  - High-growth comparator: about **25% Black**

## Repository contents

### Core notebooks

- [final_model.ipynb](./final_model.ipynb)
  - Original national pipeline.
  - Builds the tract-year panel, adjacency-based clusters, shortlist tables, enrichment-ready outputs, and transport peer-model comparisons.

- [final_model2.ipynb](./final_model2.ipynb)
  - Updated and validated cluster-time workflow.
  - Rebuilds tract-to-cluster mappings, period summaries, shortlist scores, enrichment exports, Shelby healthcare comparisons, full-county Shelby community comparisons, Georgia candidate outputs, ShelbyCare scenario outputs, and presentation-ready files.

- [shelbyfirst_simulation.ipynb](./shelbyfirst_simulation.ipynb)
  - ShelbyFirst before/after simulation notebook.
  - Produces interactive HTML views for dental, obesity, diabetes, and a combined summary chart.

### Standalone plotting / export scripts

- [shelby_map_black_pct.py](./shelby_map_black_pct.py)
  - Builds a Shelby tract choropleth of percent Black population with low-growth and high-growth community outlines.

- [generate_shelby_black_population_overlay.py](./generate_shelby_black_population_overlay.py)
  - Current publication-style Shelby choropleth used in this repo.
  - Uses ACS tract averages for 2022-2024 and overlays the selected low-growth community plus the high-growth comparator as bold outlines.

- [shelbyfirst_outcomes_chart.py](./shelbyfirst_outcomes_chart.py)
  - Builds the full ShelbyFirst outcomes pathway chart.

- [shelbyfirst_table_chart.py](./shelbyfirst_table_chart.py)
  - Builds the standalone ShelbyFirst outcomes table chart.

- [generate_submission_source_pdfs.py](./generate_submission_source_pdfs.py)
  - Exports the main source files to readable PDF form and merges them into a submission bundle.

- [generate_readme_pdf.py](./generate_readme_pdf.py)
  - Exports this README to a formatted PDF for submission or appendix use.

## Data inputs

### Primary raw inputs

- `Inclusive_Growth_Score_Data_Export_25-02-2026_134202.csv`
- `Inclusive_Growth_Score_Data_Export_25-02-2026_134202.xlsx`

These are the main Mastercard Inclusive Growth Score exports used to build the tract-level panel and downstream clustering workflows.

### Additional data sources used inside the workflow

- **ACS / Census API**
  - Demographic, labor, housing, education, insurance, and transportation proxy fields.
- **TIGER/Line tract shapefiles**
  - Used to build tract geometry and tract adjacency.
- **CDC PLACES**
  - Used for tract-level healthcare proxy measures in the Shelby deep dive.
- **Optional transit data**
  - The original model includes transport/peer-model work using ACS vehicle data and optional transit inputs.

## Main output folders

### Processed data

- `data/processed`
  - Core tract panel, cluster mappings, period summaries, shortlist tables, enrichment tables, healthcare tables, ShelbyCare scenario outputs, and other intermediate artifacts.

### Presentation-ready deliverables

- `data/processed/presentation_ready`
  - Shelby maps, healthcare charts, demographic comparison files, ZIP overlap outputs, final presentation images, and other publishable figures.

### Submission documents

- `submission_pdfs`
  - PDF exports of the requested source files plus the merged submission bundle.

## Recommended run order

If you need to reproduce the project from the existing repo state, use this order:

1. Run `final_model.ipynb`
   - Produces the original national cluster shortlist and peer-model transport outputs.
   - Key outputs include:
     - `data/processed/tract_cluster_map.parquet`
     - `data/processed/cluster_year_panel.parquet`
     - `data/processed/cluster_period_summary.parquet`
     - `data/processed/cluster_shortlist.parquet`
     - `data/processed/enrichment_clusters.parquet`
     - `data/processed/peer_similarity_transport/*`

2. Run `final_model2.ipynb`
   - Produces the validated/current shortlist and the Shelby County deep-dive outputs.
   - Key outputs include:
     - `data/processed/seed_counties.parquet`
     - `data/processed/county_signal.parquet`
     - `data/processed/cluster_shortlist.parquet`
     - `data/processed/enrichment_tract_year_panel.parquet`
     - `data/processed/presentation_ready/shelby_full_tract_cluster_map.geojson`
     - `data/processed/presentation_ready/shelby_full_community_compare.csv`
     - `data/processed/presentation_ready/three_group_selected_vs_comparator_vs_rest.csv`
     - `data/processed/presentation_ready/selected_low_growth_96_tracts.geojson`
     - `data/processed/presentation_ready/selected_low_growth_community_overlapping_zctas.csv`
     - `data/processed/presentation_ready/selected_low_growth_tract_zcta_overlap_detail.csv`
     - `data/processed/presentation_ready/shelbycare_*`
     - `data/processed/presentation_ready/georgia_candidate_clusters_ranked.csv`

3. Run the Shelby map scripts
   - `python shelby_map_black_pct.py`
   - `python generate_shelby_black_population_overlay.py`

4. Run the ShelbyFirst exports
   - `python shelbyfirst_outcomes_chart.py`
   - `python shelbyfirst_table_chart.py`
   - Open and run `shelbyfirst_simulation.ipynb` if you need the interactive HTML simulations.

5. Run the submission/document exporters
   - `python generate_submission_source_pdfs.py`
   - `python generate_readme_pdf.py` (added in this repo update)

## How to run each major component

### 1. Original model

Open `final_model.ipynb` in Jupyter and run all cells.

This notebook:

- loads the validated tract-year panel
- builds adjacency-constrained tract clusters
- scores national candidate communities
- saves shortlist and enrichment-ready tables
- adds transport-oriented peer similarity outputs

### 2. Updated model

Open `final_model2.ipynb` in Jupyter and run all cells.

This notebook:

- reloads or rebuilds the tract-to-cluster mapping
- computes period-level trajectories (`pre_covid`, `covid`, `recovery`)
- applies the current shortlist scoring logic
- exports enrichment datasets
- performs Shelby County comparison-community analysis
- builds healthcare proxy comparisons from CDC PLACES
- writes presentation-ready maps/charts and ZIP overlap files
- exports ShelbyCare scenario tables and charts

### 3. Shelby demographic / comparator maps

Run:

```bash
python shelby_map_black_pct.py
python generate_shelby_black_population_overlay.py
```

Expected outputs:

- `data/processed/presentation_ready/shelby_map_black_pct.png`
- `data/processed/presentation_ready/shelby_black_share_choropleth.png`

### 4. ShelbyFirst outcome visuals

Run:

```bash
python shelbyfirst_outcomes_chart.py
python shelbyfirst_table_chart.py
```

Expected outputs:

- `data/processed/shelbyfirst_outcomes_chart.png`
- `data/processed/shelbyfirst_table_chart.png`

### 5. ShelbyFirst simulation notebook

Open `shelbyfirst_simulation.ipynb` and run all cells.

Expected outputs:

- `data/processed/shelbyfirst_dental_simulation.html`
- `data/processed/shelbyfirst_obesity_simulation.html`
- `data/processed/shelbyfirst_diabetes_simulation.html`
- `data/processed/shelbyfirst_summary_chart.html`

## Environment and dependencies

The project has been run in a Python 3.12 environment. The main packages used across the notebooks and scripts are:

- `pandas`
- `numpy`
- `pyarrow`
- `geopandas`
- `shapely`
- `pyogrio`
- `matplotlib`
- `plotly`
- `requests`
- `scipy`
- `scikit-learn`
- `reportlab`
- `pypdf`

Install a working environment with:

```bash
python -m pip install pandas numpy pyarrow geopandas pyogrio shapely matplotlib plotly requests scipy scikit-learn reportlab pypdf
```

## Notes on reproducibility

- Large processed `.parquet` files already exist in `data/processed`; many scripts and notebooks reuse them when present.
- Some scripts fetch ACS data directly from the Census API at runtime.
- Several workflows assume the tract shapefiles are available under `data/raw/tl_2024_state_tracts/`.
- `final_model2.ipynb` is the best source of truth for the **current** Shelby shortlist position and most recent presentation-ready outputs in this repository.

## Submission assets generated from this repo

The current submission-source PDFs are stored in:

- `submission_pdfs/final_model_source.pdf`
- `submission_pdfs/final_model2_source.pdf`
- `submission_pdfs/README_project.pdf`
- `submission_pdfs/shelby_map_black_pct_source.pdf`
- `submission_pdfs/shelbyfirst_outcomes_chart_source.pdf`
- `submission_pdfs/shelbyfirst_simulation_source.pdf`
- `submission_pdfs/shelbyfirst_table_chart_source.pdf`
- `submission_pdfs/submission_source_bundle.pdf`

## Contact / interpretation note

This README documents the repository as it exists in the current workspace, including the updated Shelby demographic choropleth and the current processed shortlist position for Shelby County `cluster_77`.
