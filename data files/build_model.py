"""
Machine Learning Model for Bus Ridership Prediction
Uses integrated features with focus on distance to schools/healthcare
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("BUS RIDERSHIP PREDICTION - MODEL BUILDING")
print("=" * 80)

# Load integrated data
data_file = Path("processed_data/bus_stops_with_features.csv")
print(f"\n1. Loading integrated data from {data_file}...")
df = pd.read_csv(data_file)
print(f"   Loaded {len(df)} bus stops with {len(df.columns)} features")

# ============================================================================
# 2. PREPARE FEATURES
# ============================================================================
print("\n2. Preparing features...")

# Priority features (distance to schools/healthcare)
priority_features = [
    'distance_to_nearest_school_km',
    'distance_to_nearest_healthcare_km',
    'schools_within_0.5km',
    'schools_within_1.0km',
    'schools_within_2.0km',
    'healthcare_within_0.5km',
    'healthcare_within_1.0km',
    'healthcare_within_2.0km',
]

# Demand features (from DA-level data)
demand_features = [col for col in df.columns if col.startswith('nearest_')]

# Network features
network_features = [
    'route_connections',
    'bus_stop_density_per_km2',
    'distance_to_nearest_da_km',
]

# Population features
population_features = [
    'distance_to_high_pop_area_km',
    'population_within_0.5km',
    'population_within_1.0km',
    'population_within_2.0km',
]

# Combine all features
all_features = priority_features + demand_features + network_features + population_features

# Remove any features that don't exist or have too many missing values
available_features = []
for feat in all_features:
    if feat in df.columns:
        missing_pct = df[feat].isna().sum() / len(df) * 100
        if missing_pct < 50:  # Keep if less than 50% missing
            available_features.append(feat)
        else:
            print(f"   [WARNING] Skipping {feat} - {missing_pct:.1f}% missing")
    else:
        print(f"   [WARNING] Feature {feat} not found in data")

print(f"   Selected {len(available_features)} features for modeling")
print(f"\n   Priority features (schools/healthcare): {len([f for f in available_features if f in priority_features])}")
print(f"   Demand features: {len([f for f in available_features if f in demand_features])}")
print(f"   Network features: {len([f for f in available_features if f in network_features])}")
print(f"   Population features: {len([f for f in available_features if f in population_features])}")

# Prepare feature matrix and target
X = df[available_features].copy()
y = df['ridership_proxy'].copy()

# Handle missing values (fill with median for numeric features)
for col in X.columns:
    if X[col].isna().any():
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        print(f"   Filled {X[col].isna().sum()} missing values in {col} with median {median_val:.2f}")

print(f"\n   Feature matrix shape: {X.shape}")
print(f"   Target variable range: {y.min():.2f} - {y.max():.2f}")

# ============================================================================
# 3. SPLIT DATA
# ============================================================================
print("\n3. Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")

# ============================================================================
# 4. TRAIN MODELS
# ============================================================================
print("\n4. Training models...")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    print(f"\n   Training {name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    results[name] = {
        'model': model,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
    }
    
    print(f"      Train R²: {train_r2:.4f}")
    print(f"      Test R²:  {test_r2:.4f}")
    print(f"      Test RMSE: {test_rmse:.2f}")
    print(f"      Test MAE:  {test_mae:.2f}")
    print(f"      CV R²:    {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# ============================================================================
# 5. FEATURE IMPORTANCE (for tree-based models)
# ============================================================================
print("\n5. Feature Importance Analysis...")

# Get feature importance from Random Forest (best tree-based model)
rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 15 Most Important Features:")
print("   " + "-" * 70)
for idx, row in feature_importance.head(15).iterrows():
    print(f"   {row['feature']:40s} {row['importance']:.4f}")

# Check priority features ranking
print("\n   Priority Features (Schools/Healthcare) Ranking:")
priority_ranking = feature_importance[feature_importance['feature'].isin(priority_features)]
for idx, row in priority_ranking.iterrows():
    rank = feature_importance.index.get_loc(idx) + 1
    print(f"   Rank {rank:2d}: {row['feature']:40s} {row['importance']:.4f}")

# ============================================================================
# 6. SELECT BEST MODEL AND SAVE
# ============================================================================
print("\n6. Selecting best model...")

# Select based on test R²
best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
best_model = results[best_model_name]['model']

print(f"   Best model: {best_model_name}")
print(f"   Test R²: {results[best_model_name]['test_r2']:.4f}")
print(f"   Test RMSE: {results[best_model_name]['test_rmse']:.2f}")

# Save model
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)
model_file = model_dir / "ridership_predictor.pkl"
joblib.dump(best_model, model_file)
print(f"\n   [OK] Saved model to {model_file}")

# Save feature list
feature_file = model_dir / "feature_list.txt"
with open(feature_file, 'w') as f:
    for feat in available_features:
        f.write(f"{feat}\n")
print(f"   [OK] Saved feature list to {feature_file}")

# Save results summary
results_file = model_dir / "model_results.csv"
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train_R2': [r['train_r2'] for r in results.values()],
    'Test_R2': [r['test_r2'] for r in results.values()],
    'Test_RMSE': [r['test_rmse'] for r in results.values()],
    'Test_MAE': [r['test_mae'] for r in results.values()],
    'CV_R2_Mean': [r['cv_r2_mean'] for r in results.values()],
    'CV_R2_Std': [r['cv_r2_std'] for r in results.values()],
})
results_df.to_csv(results_file, index=False)
print(f"   [OK] Saved results to {results_file}")

# Save feature importance
importance_file = model_dir / "feature_importance.csv"
feature_importance.to_csv(importance_file, index=False)
print(f"   [OK] Saved feature importance to {importance_file}")

# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("MODEL BUILDING COMPLETE")
print("=" * 80)
print(f"\nBest Model: {best_model_name}")
print(f"  Test R² Score: {results[best_model_name]['test_r2']:.4f}")
print(f"  Test RMSE: {results[best_model_name]['test_rmse']:.2f} ridership units")
print(f"  Test MAE: {results[best_model_name]['test_mae']:.2f} ridership units")
print(f"\nModel explains {results[best_model_name]['test_r2']*100:.1f}% of variance in ridership proxy")

print("\nTop 5 Features for Prediction:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

print("\n" + "=" * 80)

