# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR  # Added SVR import
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load Data
df = pd.read_csv("C:\\Users\\Shivam\\Desktop\\Total_Data (1).csv")
df = df.loc[:, ~df.columns.str.contains('Unnamed')]
df_numeric = df.select_dtypes(include=[np.number])

# Drop missing target values
df_cleaned = df.dropna(subset=["Infant mortality rate (per 1000 live births)"])

top_features = df_numeric.corr()["Infant mortality rate (per 1000 live births)"].abs().sort_values(ascending=False).index[1:11]


# =============================
# 🔥 Enhanced Exploratory Data Analysis (EDA)
# =============================

# 🔹 Histogram for Feature Distributions
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 20), constrained_layout=True)  # Adjusted to maintain spacing
axes = axes.flatten()

for i, feature in enumerate(top_features):
    sns.histplot(df_cleaned[feature], bins=20, ax=axes[i], color='skyblue', edgecolor='black')
    axes[i].set_title(feature, fontsize=12)
    axes[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.show()

# 🔹 Correlation Heatmap (Improved Visibility)
plt.figure(figsize=(14, 10))
sns.heatmap(df_cleaned[top_features.tolist() + ["Infant mortality rate (per 1000 live births)"]].corr(), 
            annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar=True, 
            annot_kws={"size": 10}, square=True)

plt.title("Correlation Heatmap (Top Features)", fontsize=14)
plt.xticks(fontsize=10, rotation=45, ha='right')  # Rotate and align x-axis labels
plt.yticks(fontsize=10)
plt.show()

# 🔹 Define Selected Features
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

# 🔹 Feature Engineering (Adding Derived Features)
df['Literacy_gap'] = df["Men (age 15-49) who are literate4 (%)"] - df["Women (age 15-49) who are literate4 (%)"]
df['Nutrition_score'] = (df['Children under 5 years who are stunted (height-for-age)18 (%)'] + 
                         df['Children under 5 years who are underweight (weight-for-age)18 (%)']) / 2

# Ensure these features exist in df_cleaned
df_cleaned['Literacy_gap'] = df['Literacy_gap']
df_cleaned['Nutrition_score'] = df['Nutrition_score']

selected_features.extend(["Literacy_gap", "Nutrition_score"])

# Check for missing columns before plotting
missing_columns = [col for col in selected_features if col not in df_cleaned.columns]
if missing_columns:
    print("Missing columns:", missing_columns)

# 🔹 Parallel Coordinates Plot
plt.figure(figsize=(14, 8))
parallel_coordinates(df_cleaned[top_features.tolist() + ["Infant mortality rate (per 1000 live births)"]], 
                     class_column="Infant mortality rate (per 1000 live births)", colormap=plt.get_cmap("coolwarm"))
plt.xticks(fontsize=10, rotation=45, ha='right')  # Align x-axis labels properly
plt.title("Parallel Coordinates Plot for Top 10 Important Features")
plt.gca().legend_.remove()  # Remove the legend inside the plot
plt.show()

# 🔹 Feature Importance Bar Chart
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(df_cleaned[top_features], df_cleaned["Infant mortality rate (per 1000 live births)"])
feature_importances = rf_model.feature_importances_

plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances, y=top_features, hue=top_features, palette="coolwarm", legend=False)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("The Effect On Infant Mortality Rate")
plt.show()

# 🔹 Scatter Plot: Vaccination vs Infant Mortality
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_cleaned, 
                x="Children age 12-23 months who have received 3 doses of penta or DPT vaccine (%)",
                y="Infant mortality rate (per 1000 live births)", color='b')
plt.title("Vaccination Rate vs. Infant Mortality Rate", fontsize=14)
plt.xlabel("Fully Vaccinated Children (%)", fontsize=12)
plt.ylabel("Infant Mortality Rate", fontsize=12)
plt.show()

# ================================
# 🔥 Machine Learning Model Training
# ================================

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

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10, min_samples_split=3, min_samples_leaf=2)
rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)

# Evaluation Metrics
print("Random Forest Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
print(f"R²: {r2_score(y_test, y_pred)}")

# Train SVR Model
# Adding SVR model
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)
svr_pred = svr_model.predict(X_test_scaled)

# Evaluate SVR model
print("\nSVR Performance:")
print(f"MAE: {mean_absolute_error(y_test, svr_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, svr_pred))}")
print(f"R²: {r2_score(y_test, svr_pred)}")

# Train Gradient Boosting Model
gbr = GradientBoostingRegressor(n_estimators=50, learning_rate=0.02, max_depth=3, random_state=42)
gbr.fit(X_selected, y)
print("Gradient Boosting R²:", gbr.score(X_selected, y))

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

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 400],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.02, 0.1]
}
grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(X_selected, y)

print("Best GB Parameters:", grid_search.best_params_)

# Save the final model for prediction
final_model = rf_model

# ================================
# 🔥 GUI Application for Prediction
# ================================

class InfantMortalityPredictorApp:
    def _init_(self, root):
        self.root = root
        self.root.title("Infant Mortality Rate Predictor")
        self.root.geometry("1000x800")
        self.root.configure(bg="#f0f0f0")
        
        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.prediction_tab = ttk.Frame(self.notebook)
        self.visualization_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.prediction_tab, text="Prediction")
        self.notebook.add(self.visualization_tab, text="Visualizations")
        
        # Setup the prediction tab
        self.setup_prediction_tab()
        
        # Setup the visualization tab
        self.setup_visualization_tab()
        
    def setup_prediction_tab(self):
        # Main frame
        main_frame = ttk.Frame(self.prediction_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="Infant Mortality Rate Prediction", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Create a canvas with scrollbar for the feature inputs
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=1, column=0, sticky="nsew")
        scrollbar.grid(row=1, column=1, sticky="ns")
        
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Feature entry fields
        self.feature_entries = {}
        
        # Get median values from the dataset for default values
        median_values = X.median().to_dict()
        
        # Create input fields for each feature
        for i, feature in enumerate(selected_features):
            # Create a shorter display name
            display_name = feature
            if len(display_name) > 60:
                display_name = display_name[:57] + "..."
            
            label = ttk.Label(scrollable_frame, text=display_name, wraplength=400)
            label.grid(row=i, column=0, sticky="w", padx=10, pady=5)
            
            default_value = round(median_values.get(feature, 0), 2)
            entry = ttk.Entry(scrollable_frame, width=10)
            entry.insert(0, str(default_value))
            entry.grid(row=i, column=1, padx=10, pady=5)
            
            # Store the entry widget with the full feature name as key
            self.feature_entries[feature] = entry
        
        # Create tooltip for features
        tooltip_frame = ttk.Frame(main_frame)
        tooltip_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)
        
        ttk.Label(tooltip_frame, text="Hover over feature names for details").pack()
        
        # Add model selection dropdown
        model_frame = ttk.Frame(main_frame)
        model_frame.grid(row=3, column=0, columnspan=2, pady=5)
        
        ttk.Label(model_frame, text="Select Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="Random Forest")
        model_dropdown = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                      values=["Random Forest", "Gradient Boosting", "SVR"])
        model_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Add predict button and result display
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        predict_button = ttk.Button(button_frame, text="Predict Infant Mortality Rate", command=self.predict)
        predict_button.pack(pady=10)
        
        result_frame = ttk.LabelFrame(main_frame, text="Prediction Result")
        result_frame.grid(row=5, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        
        self.result_label = ttk.Label(result_frame, text="", font=("Arial", 12))
        self.result_label.pack(pady=10)
        
    def setup_visualization_tab(self):
        # Create a frame for the visualizations
        viz_frame = ttk.Frame(self.visualization_tab)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create buttons for different visualizations
        btn_frame = ttk.Frame(viz_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        correlation_btn = ttk.Button(btn_frame, text="Correlation Heatmap", 
                                     command=lambda: self.show_visualization("correlation"))
        correlation_btn.pack(side=tk.LEFT, padx=5)
        
        importance_btn = ttk.Button(btn_frame, text="Feature Importance", 
                                   command=lambda: self.show_visualization("importance"))
        importance_btn.pack(side=tk.LEFT, padx=5)
        
        scatter_btn = ttk.Button(btn_frame, text="Vaccination vs Mortality", 
                                command=lambda: self.show_visualization("scatter"))
        scatter_btn.pack(side=tk.LEFT, padx=5)
        
        # Frame for the plot
        self.plot_frame = ttk.Frame(viz_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
    def show_visualization(self, viz_type):
        # Clear previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        fig = plt.Figure(figsize=(10, 8), dpi=100)
        
        if viz_type == "correlation":
            ax = fig.add_subplot(111)
            sns.heatmap(df_cleaned[top_features.tolist() + ["Infant mortality rate (per 1000 live births)"]].corr(), 
                       annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title("Correlation Heatmap (Top Features)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
        elif viz_type == "importance":
            ax = fig.add_subplot(111)
            importance_sorted = sorted(zip(rf_model.feature_importances_, top_features))
            sns.barplot(x=[x[0] for x in importance_sorted], y=[x[1] for x in importance_sorted], ax=ax)
            ax.set_title("Feature Importance")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Features")
            
        elif viz_type == "scatter":
            ax = fig.add_subplot(111)
            sns.scatterplot(data=df_cleaned, 
                           x="Children age 12-23 months who have received 3 doses of penta or DPT vaccine (%)",
                           y="Infant mortality rate (per 1000 live births)", ax=ax)
            ax.set_title("Vaccination Rate vs. Infant Mortality Rate")
            ax.set_xlabel("Fully Vaccinated Children (%)")
            ax.set_ylabel("Infant Mortality Rate")
        
        # Create canvas for the plot
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def predict(self):
        try:
            # Get values from entry fields
            input_data = {}
            for feature, entry in self.feature_entries.items():
                try:
                    value = float(entry.get())
                    input_data[feature] = value
                except ValueError:
                    messagebox.showerror("Invalid Input", f"Please enter a valid number for {feature}")
                    return
            
            # Create input dataframe
            input_df = pd.DataFrame([input_data])
            
            # Scale the input data
            input_scaled = scaler.transform(input_df)
            
            # Get selected model
            selected_model = self.model_var.get()
            
            # Make prediction based on selected model
            if selected_model == "Random Forest":
                prediction = rf_model.predict(input_scaled)[0]
                model_info = f"Model used: Random Forest Regressor\nModel performance (R²): {r2_score(y_test, y_pred):.3f}"
            elif selected_model == "SVR":
                prediction = svr_model.predict(input_scaled)[0]
                model_info = f"Model used: Support Vector Regression\nModel performance (R²): {r2_score(y_test, svr_pred):.3f}"
            else:  # Gradient Boosting
                prediction = gbr.predict(input_scaled)[0]
                model_info = f"Model used: Gradient Boosting Regressor\nModel performance (R²): {gbr.score(X_selected, y):.3f}"
            
            # Display result
            self.result_label.config(text=f"Predicted Infant Mortality Rate: {prediction:.2f} per 1000 live births")
            
            # Show additional information
            info_text = (
                f"Prediction details:\n"
                f"- {model_info}\n"
                f"- Important factors: {', '.join(top_features[:3])}"
            )
            messagebox.showinfo("Prediction Details", info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {str(e)}")
            print(f"Error details: {e}")

if _name_ == "_main_":
    root = tk.Tk()
    app = InfantMortalityPredictorApp(root)
    root.mainloop()