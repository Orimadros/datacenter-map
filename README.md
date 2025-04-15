# Data Center Mapping Project

This project aims to analyze and visualize the distribution of data centers and related infrastructure (cell towers, transmission lines) across different regions, with a focus on Brazil and the United States.

## Recent Updates

- **XGBoost Predictive Model**: Added a machine learning model that predicts data center locations across São Paulo state
  - Uses features like cell tower density, proximity to power infrastructure, and distance to existing hyperscalers
  - Includes anti-overfitting mechanisms (L1/L2 regularization, reduced tree complexity)
  - Generates predictions for all census tracts in the state
- **Enhanced Visualization**: Improved heatmap visualization of model predictions
  - Uses an elegant 'Reds' colormap for publication-quality output
  - Data center points are displayed with optimized size and white outline
  - Outputs are saved to standardized locations in the project structure
- **False Positive Analysis**: Added detailed analysis of high-probability false positives
  - Identifies census tracts predicted to have data centers but actually don't
  - Analyzes feature distributions across the top 300 false positives
  - Shows which infrastructure characteristics drive high predictions
  - Provides insights into potential future data center locations

## Project Structure

```
dc_map_project/
├── data/                # Data directory
│   ├── raw/            # Original, immutable data
│   │   ├── global/     # Global/shared datasets
│   │   │   ├── cell_towers.csv
│   │   │   ├── VIIRS/          # Satellite data
│   │   │   └── 110m_cultural/  # Geographic data
│   │   ├── brasil/     # Brazil-specific data
│   │   │   ├── infrastructure/
│   │   │   ├── population/
│   │   │   └── other_data/
│   │   └── us/         # US-specific data
│   │       ├── national_risk_index/
│   │       ├── 5G_data/
│   │       ├── zcta5_data/     # ZIP Code data
│   │       ├── transmission_line_data/
│   │       ├── substations/
│   │       ├── broadband/       # FTTP and cable data
│   │       └── population/
│   ├── processed/      # Cleaned and processed data
│   ├── model_input/    # Data ready for model consumption
│   ├── model_output/   # Model predictions and outputs
│   └── interim/        # Intermediate data
├── docs/               # Documentation
│   ├── api/           # API documentation
│   ├── notebooks/     # Rendered notebooks
│   └── reports/       # Generated reports
├── notebooks/          # Jupyter notebooks
│   ├── analysis/      # Analysis notebooks
│   │   ├── brasil/    # Brazil-specific analysis
│   │   └── us/        # US-specific analysis
│   └── exploration/   # Data exploration notebooks
├── src/               # Source code
│   ├── data/         # Data processing scripts
│   ├── visualization/# Plotting and visualization
│   ├── models/       # Model code
│   └── utils/        # Utility functions
├── models/            # Trained model artifacts
│   └── sao_paulo_xgboost/   # XGBoost model for São Paulo
├── tests/            # Test files
└── outputs/          # Generated outputs
    ├── figures/     # Generated plots and figures
    │   ├── brasil/  # Brazil-specific figures
    │   └── us/      # US-specific figures
    └── reports/     # Generated reports
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd dc_map_project
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Processing
- Raw data is organized into three main categories:
  - Global: Shared datasets used by both country analyses
  - Brazil: Country-specific data and infrastructure
  - US: Country-specific data and infrastructure
- Data processing scripts are in `src/data/`
- Data exploration notebooks are in `notebooks/exploration/`

#### Geocoding Challenges
- Brazilian address geocoding presented significant challenges:
  - Nominatim (the open-source geocoding service) failed to locate many data center addresses
  - This was resolved by implementing a hybrid approach:
    - Primary attempt using Nominatim with optimized address formatting
    - Fallback to a manually created dictionary of hardcoded coordinates
    - Hardcoded coordinates were obtained using Google Maps for precision
  - The implementation includes caching and validation to ensure coordinates fall within Brazil's boundaries
  - The complete solution improved geocoding success rate from ~10% to over 80%

### Visualization
- Visualization scripts are in `src/visualization/`
- Example: `python src/visualization/plot_cell_towers.py`
- Heatmap visualization: `src/visualization/plot_heatmap.py` provides a reusable function for creating probability heatmaps
- Country-specific visualizations are saved in their respective output directories:
  - Brazil: `outputs/figures/brasil/`
  - US: `outputs/figures/us/`

### Analysis
- Analysis notebooks are organized by country:
  - Brazil: `notebooks/analysis/brasil/`
  - US: `notebooks/analysis/us/`
- Reports are generated in `docs/reports/`

### Predictive Modeling
- XGBoost model for predicting data center locations: `src/models/sao_paulo_dc_prediction_xgboost.py`
- Model features:
  - Cell tower density and count
  - Distance to nearest electrical substation
  - Distance to nearest transmission line
  - Distance to nearest hyperscaler data center
  - Night light intensity (VIIRS)
  - Census tract area
  - Situational code of the census tract
- To run the model and generate predictions:
  ```bash
  python src/models/sao_paulo_dc_prediction_xgboost.py
  ```
- The model outputs:
  - Predictions for all census tracts as GeoJSON: `data/model_output/brasil/sao_paulo_census_tracts_predictions_xgb.geojson`
  - Heatmap visualization: `outputs/figures/sao_paulo_city_heatmap_xgb.png`
  - Model artifacts (saved model, scaler): `models/sao_paulo_xgboost/`
  - False positive analysis in the log output

## Data Sources

### Global Data
- Cell tower data (global coverage)
- VIIRS satellite data
- Natural Earth geographic data

### Brazil-specific Data
- Infrastructure data
- Population data
- Other Brazilian datasets
- Datacenter locations geocoded using a combination of Nominatim and manual geocoding
  - Due to limitations in Nominatim's ability to geocode all Brazilian addresses, many coordinates were manually hardcoded using Google Maps tool
  - The geocoding process is handled in `src/data/wrangle_brazil_data.py`

### US-specific Data
- National Risk Index
- 5G infrastructure data
- ZIP Code Tabulation Areas (ZCTA)
- Transmission line data
- Substation data
- Broadband infrastructure (FTTP and cable)
- Population data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
