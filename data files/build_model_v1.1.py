"""
Machine Learning Models for Bus Ridership Prediction
Linear Regressor - baseline model to fit data for comparison.
RandomForest Regressor - advanced model to capture complex patterns and make predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, r2_score
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("BUS RIDERSHIP PREDICTION - MODEL BUILDING")
print("=" * 80)

# ============================================================================
# 1. LOAD INTEGRATED DATA
# ============================================================================

df = pd.read_csv(Path("processed_data/bus_stops_with_features.csv"))

# print(df.head())

# ============================================================================
# 2. PREPARE AND LOAD DATA
# ============================================================================

priority_features = [
    'nearest_Overall_Accessibility',
    'nearest_Commute_PT_Demand',
    'nearest_Population_Density_Demand',
    'nearest_Bus_Stop_Proximity',
    'population_within_1.0km',
    'schools_within_0.5km',
    'healthcare_within_0.5km',
    'route_connections'
]

target_feature = 'ridership_proxy'

# Seperate features and target
X = df[priority_features]
y = df[target_feature]

# ============================================================================
# 3. SPLIT DATA
# ============================================================================

#   80% randomly sampled data for training
#   20% hold-out data for final testing
#   For both the baseline and advanced model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================================
# 4. TRAIN BASELINE MODEL - LINEAR REGRESSION
# ============================================================================

print("\nTraining baseline model - Linear Regression...")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred_linear = linear_model.predict(X_train_scaled)
y_test_pred_linear = linear_model.predict(X_test_scaled)

# Calculate metrics
linear_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_linear))
linear_train_mae = mean_absolute_error(y_train, y_train_pred_linear)
linear_train_r2 = r2_score(y_train, y_train_pred_linear)
linear_train_mape = mean_absolute_percentage_error(y_train, y_train_pred_linear) * 100

linear_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_linear))
linear_test_mae = mean_absolute_error(y_test, y_test_pred_linear)
linear_test_r2 = r2_score(y_test, y_test_pred_linear)
linear_test_mape = mean_absolute_percentage_error(y_test, y_test_pred_linear) * 100

# Cross-validation
linear_cv_scores = cross_val_score(linear_model, X_train_scaled, y_train, 
                                   cv=5, scoring='r2', n_jobs=-1)
print(f"\n5-Fold CV R² Score: {linear_cv_scores.mean():.4f} (+/- {linear_cv_scores.std():.4f})")

print("\nFinished training.")

# ============================================================================
# 5. TRAIN ADVANCED MODEL - RANDOM FOREST REGRESSION
# ============================================================================

print("\nTraining advanced model - Random Forest Regression...")

# Define expanded parameter grid - 9600 Folds
param_grid = {
    'n_estimators': [100, 200, 300, 500],        # 4 options
    'max_depth': [10, 15, 20, 30, None],         # 5 options
    'min_samples_split': [2, 5, 10, 15],         # 4 options
    'min_samples_leaf': [1, 2, 4, 8],            # 4 options
    'max_features': ['sqrt', 'log2', 0.5],       # 3 options
    'bootstrap': [True, False]                   # 2 options
}

# Base RFR model
rfr = RandomForestRegressor(random_state=42, n_jobs=-1)

# Grid Search with Cross-Validation
grid_search = GridSearchCV(
    estimator=rfr,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='r2',               # Optimize for R² score - not sensitive to outliers
    verbose=1
)

# Fit grid search
grid_search.fit(X_train, y_train)
    
# Get best model
rf_model = grid_search.best_estimator_

# Predictions
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

# Calculate metrics
rf_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
rf_train_mae = mean_absolute_error(y_train, y_train_pred_rf)
rf_train_r2 = r2_score(y_train, y_train_pred_rf)
rf_train_mape = mean_absolute_percentage_error(y_train, y_train_pred_rf) * 100

rf_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
rf_test_mae = mean_absolute_error(y_test, y_test_pred_rf)
rf_test_r2 = r2_score(y_test, y_test_pred_rf)
rf_test_mape = mean_absolute_percentage_error(y_test, y_test_pred_rf) * 100

# Cross-validation
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, 
                               cv=5, scoring='r2', n_jobs=-1)
print(f"\n5-Fold CV R² Score: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")

print("\nFinished training.")

# ============================================================================
# 6. SAVE MODELS
# ============================================================================

print("\nSAVING MODELS")

# Save Linear Regression (baseline model)
linear_package = {
    'model': linear_model,
    'scaler': scaler,
    'feature_names': priority_features
}

# Save Linear Regression
joblib.dump(linear_package, 'models/linear_regression_model.pkl')
print("Saved: models/linear_regression_model.pkl")

# Save Random Forest (advanced model)
rf_package = {
    'model': rf_model,
    'feature_names': priority_features,
    'target_name': target_feature
}
joblib.dump(rf_package, 'models/random_forest_model.pkl')
print("Saved: models/random_forest_model.pkl")

# ============================================================================
# 7. MODEL COMPARISON
# ============================================================================

print("\nMODEL COMPARISON")

comparison_df = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R²', 'MAPE (%)'],
    'Linear Regression': [
        linear_test_rmse,
        linear_test_mae,
        linear_test_r2,
        linear_test_mape
    ],
    'Random Forest': [
        rf_test_rmse,
        rf_test_mae,
        rf_test_r2,
        rf_test_mape
    ]
})

# Calculate improvement
comparison_df['Improvement'] = [
    f"{((linear_test_rmse - rf_test_rmse) / linear_test_rmse * 100):.1f}%",
    f"{((linear_test_mae - rf_test_mae) / linear_test_mae * 100):.1f}%",
    f"{((rf_test_r2 - linear_test_r2) / linear_test_r2 * 100):.1f}%",
    f"{((linear_test_mape - rf_test_mape) / linear_test_mape * 100):.1f}%"
]

print("\n" + comparison_df.to_string(index=False))

# Determine winner
print("\n" + "-" * 70)
if rf_test_r2 > linear_test_r2:
    print("Winner: Random Forest Regression")
    print(f"Random Forest has {comparison_df.iloc[2, 3]} better R² score")
else:
    print("Winner: Linear Regression")
    print(f"Linear Regression has better R² score")
print("-" * 70)

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

#
# Plot: Predictions comparison
#

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Linear Regression
axes[0].scatter(y_test, y_test_pred_linear, alpha=0.6, edgecolors='k', linewidth=0.5)
axes[0].plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Ridership', fontsize=12)
axes[0].set_ylabel('Predicted Ridership', fontsize=12)
axes[0].set_title(f'Linear Regression (R²={linear_test_r2:.3f})', 
                  fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Random Forest Regression
axes[1].scatter(y_test, y_test_pred_rf, alpha=0.6, edgecolors='k', 
                linewidth=0.5, color='green')
axes[1].plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Ridership', fontsize=12)
axes[1].set_ylabel('Predicted Ridership', fontsize=12)
axes[1].set_title(f'Random Forest (R²={rf_test_r2:.3f})', 
                  fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_predictions_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: model_predictions_comparison.png")
plt.close()

#
# Plot: Feature Importance for Random Forest Regression model
#
# This bar chart visualizes the importance of each feature used in the Random Forest model.
# Feature importance indicates how much each feature contributes to reducing prediction error.

importances = rf_model.feature_importances_     
feature_names = priority_features               

feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
plt.title('Feature Importance - Random Forest Model', fontsize=16, fontweight='bold')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('random_forest_feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: random_forest_feature_importance.png")
plt.close()

#
# Plot Grid Search Hyperparameter Heatmap
#
# This heatmap visualizes the mean test scores from the grid search
# across different combinations of 'n_estimators' and 'max_depth'.
# The color intensity indicates the performance of the model with those hyperparameters.
# The optimal combination was identified to be n_estimators=300 and max_depth=20.

grid_results = pd.DataFrame(grid_search.cv_results_)
pivot_table = grid_results.pivot_table(
    values='mean_test_score',
    index='param_n_estimators',
    columns='param_max_depth'
)
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap='YlGnBu')
plt.title('Grid Search CV Mean Test Scores', fontsize=16, fontweight='bold')
plt.xlabel('Max Depth', fontsize=12)
plt.ylabel('Number of Estimators', fontsize=12)
plt.tight_layout()
plt.savefig('grid_search_cv_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: grid_search_cv_heatmap.png")
plt.close()