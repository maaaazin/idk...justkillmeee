# =============================================================================
# CELL 1: Import necessary libraries
# =============================================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Set plot style
plt.style.use('seaborn')
sns.set_palette('coolwarm')

# =============================================================================
# CELL 2: Data Loading and Preprocessing
# =============================================================================
# Load Data
df = pd.read_csv("Total_Data (1).csv")
df = df.loc[:, ~df.columns.str.contains('Unnamed')]
df_numeric = df.select_dtypes(include=[np.number])

# Drop missing target values
df_cleaned = df.dropna(subset=["Infant mortality rate (per 1000 live births)"])

# Get top features
top_features = df_numeric.corr()["Infant mortality rate (per 1000 live births)"].abs().sort_values(ascending=False).index[1:11]

print("Top 10 features:")
for i, feature in enumerate(top_features, 1):
    print(f"{i}. {feature}")

# =============================================================================
# CELL 3: Histogram for Feature Distributions
# =============================================================================
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 20), constrained_layout=True)
axes = axes.flatten()

for i, feature in enumerate(top_features):
    sns.histplot(df_cleaned[feature], bins=20, ax=axes[i], color='skyblue', edgecolor='black')
    axes[i].set_title(feature, fontsize=12)
    axes[i].tick_params(axis='x', rotation=45)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.show()

# =============================================================================
# CELL 4: Correlation Heatmap
# =============================================================================
plt.figure(figsize=(14, 10))
sns.heatmap(df_cleaned[top_features.tolist() + ["Infant mortality rate (per 1000 live births)"]].corr(), 
            annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar=True, 
            annot_kws={"size": 10}, square=True)

plt.title("Correlation Heatmap (Top Features)", fontsize=14)
plt.xticks(fontsize=10, rotation=45, ha='right')
plt.yticks(fontsize=10)
plt.show()

# =============================================================================
# CELL 5: Feature Engineering
# =============================================================================
# Define Selected Features
selected_features = [
    "Women (age 15-49 years) having a mobile phone that they themselves use (%)",
    "Women (age 15-49)  with 10 or more years of schooling (%)",
    "Men (age 15-49)  with 10 or more years of schooling (%)",
    "Population below age 15 years (%)",
    "Children under age 3 years breastfed within one hour of birth15 (%)",
    "Births delivered by caesarean section (in the 5 years before the survey) (%)",
    "Total children age 6-23 months receiving an adequate diet16, 17  (%)",
    "Children who received postnatal care from a doctor/nurse/LHV/ANM/midwife/ other health personnel within 2 days of delivery (for last birth in the 5 years before the survey) (%)",
    "Mothers who had at least 4 antenatal care visits  (for last birth in the 5 years before the survey) (%)",
    "Institutional births (in the 5 years before the survey) (%)",
    "Children age 12-23 months who have received 3 doses of penta or DPT vaccine (%)",
    "Women (age 15-49) who are literate4 (%)",
    "Mothers who consumed iron folic acid for 100 days or more when they were pregnant (for last birth in the 5 years before the survey) (%)",
    "Children age 12-23 months who have received BCG (%)",
    "Children age 12-23 months who have received the first dose of measles-containing vaccine (MCV) (%)",
    "Children age 24-35 months who have received a second dose of measles-containing vaccine (MCV) (%)",
    "Children age 12-23 months who have received 3 doses of penta or hepatitis B vaccine (%)",
    "Children age 9-35 months who received a vitamin A dose in the last 6 months (%)",
    "Children age 12-23 months who received most of their vaccinations in a public health facility (%)",
    "Children age 12-23 months who received most of their vaccinations in a private health facility (%)",
    "Births in a private health facility that were delivered by caesarean section (in the 5 years before the survey) (%)",
    "Children under 5 years who are underweight (weight-for-age)18 (%)",
    "Children under 5 years who are stunted (height-for-age)18 (%)",
    "Children under 5 years who are overweight (weight-for-height)20 (%)",
    "Currently married women (age 15-49 years) who usually participate in three household decisions25 (%)"
]

# Feature Engineering
df['Literacy_gap'] = df["Men (age 15-49) who are literate4 (%)"] - df["Women (age 15-49) who are literate4 (%)"]
df['Nutrition_score'] = (df['Children under 5 years who are stunted (height-for-age)18 (%)'] + 
                         df['Children under 5 years who are underweight (weight-for-age)18 (%)']) / 2

# Ensure these features exist in df_cleaned
df_cleaned['Literacy_gap'] = df['Literacy_gap']
df_cleaned['Nutrition_score'] = df['Nutrition_score']

selected_features.extend(["Literacy_gap", "Nutrition_score"])

print("Total number of features:", len(selected_features))

# =============================================================================
# CELL 6: Model Training and Evaluation - Data Preparation
# =============================================================================
# Define Features and Target
X = df[selected_features]
y = df["Infant mortality rate (per 1000 live births)"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection
model = RandomForestRegressor(n_estimators=100, random_state=42)
rfe = RFE(model, n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)
selected_indices = rfe.get_support(indices=True)
final_selected_features = [selected_features[i] for i in selected_indices]

print("Selected features:")
for i, feature in enumerate(final_selected_features, 1):
    print(f"{i}. {feature}")

# =============================================================================
# CELL 7: Train Random Forest Model
# =============================================================================
# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10, min_samples_split=3, min_samples_leaf=2)
rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)

# Evaluation Metrics
print("Random Forest Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# =============================================================================
# CELL 8: Train SVR Model
# =============================================================================
# Train SVR Model
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)
svr_pred = svr_model.predict(X_test_scaled)

# Evaluate SVR model
print("\nSVR Performance:")
print(f"MAE: {mean_absolute_error(y_test, svr_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, svr_pred)):.2f}")
print(f"R²: {r2_score(y_test, svr_pred):.2f}")

# =============================================================================
# CELL 9: Train Gradient Boosting Model
# =============================================================================
# Train Gradient Boosting Model
gbr = GradientBoostingRegressor(n_estimators=50, learning_rate=0.02, max_depth=3, random_state=42)
gbr.fit(X_selected, y)
print("Gradient Boosting R²:", gbr.score(X_selected, y))

# =============================================================================
# CELL 10: Hyperparameter Tuning for SVR
# =============================================================================
# Hyperparameter Tuning for SVR
svr_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1],
    'epsilon': [0.01, 0.1, 0.2]
}
svr_grid_search = GridSearchCV(SVR(kernel='rbf'), svr_param_grid, cv=5, n_jobs=-1, scoring='r2')
svr_grid_search.fit(X_train_scaled, y_train)

print("Best SVR Parameters:", svr_grid_search.best_params_)
print("Best SVR Score:", svr_grid_search.best_score_)

# =============================================================================
# CELL 11: Hyperparameter Tuning for Gradient Boosting
# =============================================================================
# Hyperparameter Tuning for Gradient Boosting
param_grid = {
    'n_estimators': [100, 200, 400],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.02, 0.1]
}
grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(X_selected, y)

print("Best GB Parameters:", grid_search.best_params_)

# =============================================================================
# CELL 12: Save Models and Scaler
# =============================================================================
import joblib

# Save models and scaler
joblib.dump(rf_model, 'rf_model.joblib')
joblib.dump(svr_model, 'svr_model.joblib')
joblib.dump(gbr, 'gbr_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(selected_features, 'selected_features.joblib')
joblib.dump(top_features, 'top_features.joblib')

print("Models and scaler saved successfully!") 