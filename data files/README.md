# Bus Ridership Prediction ML Project

## Overview

This project predicts bus ridership at bus stops to help plan new stops and optimize routes. The model uses demand metrics as a proxy for actual ridership (since ridership data is not available) and prioritizes distance to schools and healthcare facilities as key features.

## ✅ Project Status: COMPLETE

- **Data Integration**: ✅ Complete (1,930 stops, 31 features)
- **Model Training**: ✅ Complete (99.7% R² score)
- **Prediction Tools**: ✅ Ready to use

## Quick Start

### 1. Predict Ridership for a New Location

```bash
python predict_ridership.py <latitude> <longitude>
```

**Example:**
```bash
python predict_ridership.py 48.4284 -123.3656
```

This will:
- Calculate all features for the location (distance to schools, healthcare, population, etc.)
- Predict ridership score (0-100 scale)
- Show key feature values

### 2. View Model Results

Check the following files for detailed results:
- `PROJECT_SUMMARY.md` - Complete project summary
- `models/model_results.csv` - Model comparison
- `models/feature_importance.csv` - Feature importance rankings

## Project Structure

```
data files/
├── processed_data/
│   └── bus_stops_with_features.csv    # Integrated dataset (1,930 stops)
├── models/
│   ├── ridership_predictor.pkl        # Trained model
│   ├── feature_list.txt               # List of features
│   ├── model_results.csv              # Model comparison
│   └── feature_importance.csv         # Feature rankings
├── integrate_data.py                  # Data integration (already run)
├── build_model.py                     # Model training (already run)
├── predict_ridership.py               # Prediction tool
├── PROJECT_SUMMARY.md                  # Detailed summary
└── README.md                           # This file
```

## Key Features

### Priority Features (Distance-based)
- Distance to nearest school
- Distance to nearest healthcare facility
- Count of schools/healthcare within 0.5km, 1km, 2km buffers

### Demand Features
- Overall demand from nearest census area
- Public transit commute demand
- Population density demand
- Route coverage
- And more...

### Network Features
- Route connectivity
- Bus stop density
- Distance to nearest census area

### Population Features
- Distance to high-population areas
- Population counts within buffers

## Model Performance

**Best Model**: Gradient Boosting Regressor
- **Test R²**: 0.9973 (99.7% variance explained)
- **Test RMSE**: 0.88 ridership units
- **Test MAE**: 0.64 ridership units

## Important Notes

### Limitations
1. **No actual ridership data**: Model uses demand metrics as a proxy
2. **Static predictions**: No temporal features (time of day, day of week)
3. **Victoria-specific**: Trained on Victoria, BC data

### For New Locations
When predicting for new locations without existing demand data, the model relies more on:
- Distance to schools/healthcare (priority features)
- Population density
- Network connectivity

## Usage Examples

### Example 1: Predict for a specific location
```bash
python predict_ridership.py 48.4284 -123.3656
```

### Example 2: Load and analyze the integrated data
```python
import pandas as pd
df = pd.read_csv('processed_data/bus_stops_with_features.csv')
print(df.describe())
```

### Example 3: Use the model programmatically
```python
import joblib
import numpy as np

model = joblib.load('models/ridership_predictor.pkl')
# Calculate features for your location, then:
prediction = model.predict([feature_vector])
```

## Files Description

- **integrate_data.py**: Combines all data sources into a single dataset
- **build_model.py**: Trains and evaluates ML models
- **predict_ridership.py**: Predicts ridership for new locations
- **PROJECT_SUMMARY.md**: Detailed project documentation
- **ML_Project_Analysis.md**: Initial analysis and questions

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install pandas numpy scikit-learn geopandas shapely
```

## Next Steps

1. **Use predictions** to identify high-ridership potential locations
2. **Analyze feature importance** to understand what drives ridership
3. **Collect actual ridership data** when available to improve the model
4. **Add temporal features** for time-based predictions (future enhancement)

## Support

For questions or issues, refer to:
- `PROJECT_SUMMARY.md` for detailed documentation
- `ML_Project_Analysis.md` for initial analysis
- Model files in `models/` directory for technical details

---

**Project Status**: ✅ Complete and ready to use
**Model Performance**: Excellent (99.7% R²)
**Priority Features**: Distance to schools/healthcare included and working

