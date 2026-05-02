# ShelbyFirst: A Data-Driven Inclusive Growth Strategy for Shelby County

**ShelbyFirst** is a data-driven strategy proposal developed for the Mastercard Inclusive Growth Score (IGS) Challenge. The project identifies a high-need community in Shelby County, Tennessee, diagnoses overlapping health-access and economic barriers, and proposes a practical intervention built on mobility, nutrition, and care connectivity.

## Challenge Overview

The **Mastercard Inclusive Growth Score Challenge** asked teams to identify low-growth communities with structural economic and social barriers, explain the drivers of exclusion, and propose evidence-based interventions. ShelbyFirst ranked among the finalist submissions.

## Start Here

- **Presentation deck**: [ShelbyFirst.odp](./ShelbyFirst.odp)
- **Main analysis**: [src/final_model.ipynb](./src/final_model.ipynb)
- **Shelby deep-dive**: [src/final_model2.ipynb](./src/final_model2.ipynb)
- **Scenario modeling**: [src/shelbyfirst_simulation.ipynb](./src/shelbyfirst_simulation.ipynb)

## The Community

- **Location**: South Memphis / Whitehaven, Shelby County, Tennessee
- **Scale**: 96 census tracts, about 273,000 residents
- **IGS Score**: 35.8 for the 2022-2024 recovery period
- **Rank**: #6 of 67 shortlisted low-growth recovery clusters
- **Focus ZIP codes**: 38109, 38116, 38117

### Economic Underperformance

| Metric | Low-IGS Shelby | High-IGS Shelby | Gap |
|---|---:|---:|---:|
| Median household income | $43.4K | $119.4K | -64% |
| Poverty rate | 30.0% | 5.4% | +454% |
| Unemployment | 11.8% | 3.5% | +237% |
| Labor force participation | 58.2% | 66.9% | -13% |

### Health-Access Crisis

| Outcome | Low-IGS Shelby | High-IGS Shelby | Gap |
|---|---:|---:|---:|
| Uninsured adults | 15.9% | 7.3% | +118% |
| Dental visit gap | 55.6% | 32.1% | +73% |
| Diabetes | 21.6% | 11.3% | +91% |
| Obesity | 45.2% | 32.5% | +39% |

### Transportation Friction

Vehicle access varies sharply by ZIP code, suggesting transportation as a barrier to care:

- `38109` low-growth: roughly 30-45% of households with 0 or 1 vehicle
- `38117` high-growth: roughly 15-25% of households with 0 or 1 vehicle

## The Intervention

ShelbyFirst connects three programs into one platform:

- **Move**: affordable healthcare-focused rides
- **Nourish**: community garden and nutrition support
- **Connect**: app-driven coordination, reminders, and referrals

## Implementation Roadmap

### Phase 1 - Pilot

- Launch in the highest-need South Memphis service area
- Start one community garden site
- Build referral loops with ShelbyCares and trusted providers

### Phase 2 - Expand

- Add more ride destinations and health-event partners
- Expand produce distribution and awareness
- Grow the membership base and app usage

### Phase 3 - Scale

- Expand across more low-IGS tracts
- Formalize healthcare referral loops
- Transition toward blended nonprofit sustainability

## Projected Outcomes

### Base-Case Scenario

| Metric | Year 1 | Year 2 | Year 3 | High-IGS Baseline |
|---|---:|---:|---:|---:|
| IGS Score | 36.8 | 38.4 | 40.5 | 55.7 |
| Dental visit gap | 54.4% | 52.7% | 50.7% | 32.1% |
| Obesity | 44.8% | 44.2% | 43.5% | 32.5% |
| Diabetes | 21.3% | 21.0% | 20.5% | 11.3% |

## Repository Contents

- `ShelbyFirst.odp` - current presentation deck
- `src/final_model2.ipynb` - primary analysis notebook
- `src/final_model.ipynb` - earlier national clustering and shortlist workflow
- `src/shelbyfirst_simulation.ipynb` - scenario modeling and outcome projections
- `src/fetch_acs_internet_access.py` - Shelby tract internet-access fetcher
- `src/fetch_acs_vehicle_access.py` - Shelby tract vehicle-access fetcher
- `src/generate_shelby_slidedeck_visuals.py` - slide-deck chart and map renderer
- `src/project_paths.py` - shared path configuration
- `src/data/` - raw and processed analytical inputs and tables
- `src/outputs/` - generated PNGs and presentation/export artifacts

## How to Run

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Explore the analysis:

- Open `src/final_model2.ipynb` for the Shelby-specific analysis
- Open `src/shelbyfirst_simulation.ipynb` for outcome scenarios

3. Refresh data if needed:

```bash
python src/fetch_acs_internet_access.py
python src/fetch_acs_vehicle_access.py
```

4. Regenerate the slide-deck PNGs:

```bash
python src/generate_shelby_slidedeck_visuals.py
```

- Default output folder: `src/outputs/presentation_ready/`
- To render one asset only:

```bash
python src/generate_shelby_slidedeck_visuals.py --only black-map
```

- To override the destination:

```bash
python src/generate_shelby_slidedeck_visuals.py --output-dir <path>
```

- Legacy single-purpose entrypoints remain available:

```bash
python src/generate_shelby_black_population_overlay.py
python src/shelby_internet_access_bar_chart.py
```

5. Scenario-model exports:

- `src/outputs/ml_simulation/` holds the simulation PNGs and export CSVs
- `src/data/processed/ml_simulation/` now keeps only the processed parquet tables used for the simulation work

## Notes

- `src/data/processed/` is reserved for processed data inputs and analytical tables.
- `src/outputs/` is the home for generated charts, PNGs, and presentation/export artifacts.
- Large raw and processed data files are stored locally for reference and are not all tracked in version control.
- This project was AI-assisted. AI tools were used to help with coding, analysis support, documentation, and presentation asset generation, with project direction and decision-making led by the team.

## Credits

This project was developed by **Kareem Weaver** in collaboration with:

- **Kareem Weaver** - Computer Science major. Led data preprocessing, final modeling, core analysis, and technical implementation.
- **Santiago Soto** - Computer Science major. Led the machine learning work for predicted outcomes and scenario-oriented modeling.
- **Chace Cleveland** - Business major. Led storytelling, solution development, and community outreach.
