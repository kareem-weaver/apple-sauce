# ShelbyFirst: A Data-Driven Inclusive Growth Strategy for Shelby County

ShelbyFirst is a portfolio project developed from a Mastercard Inclusive Growth Score (IGS) challenge submission. The work identifies a high-need community in Shelby County, Tennessee, benchmarks it against stronger peer communities, and proposes a measurable intervention built around mobility, health access, and nutrition.

## Start Here

- Pitch deck: [ShelbyFirst.pptx](./ShelbyFirst.pptx)
- PDF version: [ShelbyFirst.pdf](./ShelbyFirst.pdf)
- Main analysis notebook: [src/final_model2.ipynb](./src/final_model2.ipynb)
- Submission/package builder: [src/build_submission_package.py](./src/build_submission_package.py)

## Executive Summary

- Selected community: `Tennessee | Shelby County | cluster_77`
- Current shortlist rank: `#6 of 67` shortlisted low-growth recovery clusters
- Scale: `96` tracts and about `273K` residents in the recovery-period profile
- Analysis periods: `2017-2019` pre-COVID, `2020-2021` COVID shock, and `2022-2024` recovery
- Core data sources: Mastercard IGS, ACS/Census, CDC PLACES, tract geometry, and Shelby-specific healthcare/access enrichments

## What The Analysis Found

- The selected community is a large, contiguous low-growth cluster with a recovery-period IGS total of `35.8`.
- Economic underperformance is severe relative to a high-growth Shelby comparator.
- Median household income is about `$43.4K` versus `$119.4K`.
- Poverty is `30.0%` versus `5.4%`.
- Unemployment is `11.8%` versus `3.5%`.
- Labor-force participation is `58.2%` versus `66.9%`.
- The selected community is also far more racially concentrated.
- Selected low-growth community: `76.8%` Black
- Rest of Shelby County: `58.4%` Black
- High-growth comparator: `24.6%` Black
- Health-access stress overlaps with economic stress rather than appearing as a separate issue.
- Uninsured adults: `15.9%` versus `7.3%`
- Dental visit gap: `55.9%` versus `32.2%`
- Diabetes: `21.6%` versus `11.2%`
- Obesity: `46.2%` versus `31.5%`
- The central analytical story is consistent across the notebook outputs and the deck.
- Economic vulnerability, transportation friction, and care-access gaps reinforce each other and slow inclusive growth.

## Proposed Intervention

ShelbyFirst combines three connected ideas into one operating model:

- `Move`: lower-cost rides focused on healthcare access
- `Nourish`: community garden and healthy food programming
- `Connect`: referral, scheduling, and engagement support through trusted partners

The goal is to improve inclusive growth indirectly by reducing missed care, lowering access friction, and strengthening community participation.

## Scenario Modeling

The repo also includes a directional ShelbyCare scenario model used for pitch planning.

- In the `full` scenario, modeled unemployment improves by about `0.19` percentage points.
- Uninsured adults improve by about `0.96` percentage points.
- Obesity improves by about `2.70` percentage points.
- Diabetes improves by about `1.80` percentage points.
- These are scenario-based planning estimates, not causal treatment-effect claims.

## Repository Contents

- `ShelbyFirst.pptx` and `ShelbyFirst.pdf` contain the final presentation materials.
- `requirements.txt` lists the Python dependencies for the public repo.
- `src/final_model2.ipynb` is the current source of truth for the Shelby analysis and presentation-ready outputs.
- `src/final_model.ipynb` preserves the earlier national clustering and shortlist workflow.
- `src/shelbyfirst_simulation.ipynb` contains the scenario and simulation work.
- `src/*.py` contains the map builders, chart scripts, data fetchers, PDF helpers, and packaging utilities.

## Reproducing The Project

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Place the required raw and processed data under `src/data/`.
3. Rebuild the packaged deliverables:

```bash
python src/build_submission_package.py
```

4. If you need to refresh Shelby tract internet access from Census ACS:

```bash
python src/fetch_acs_internet_access.py
```

## What Is Intentionally Not Checked In

To keep the GitHub repo focused and lightweight, the public version excludes:

- Large raw and processed data folders under `src/data/`
- Generated package outputs under `src/outputs/`
- Temporary preview images and cache directories

The key public artifacts are the deck, the notebooks, the core scripts, and the documented analytical story.
