import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ttkthemes import ThemedTk
import pandas as pd
import joblib

# Set plot style
plt.style.use('seaborn-v0_8')  # Updated style name
sns.set_theme()  # Use seaborn's default theme

class InfantMortalityPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Infant Mortality Rate Predictor")
        self.root.geometry("1200x800")
        
        # Set theme
        style = ttk.Style()
        style.configure("TNotebook", background="#f0f0f0")
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Helvetica", 10))
        style.configure("Title.TLabel", font=("Helvetica", 16, "bold"), foreground="#2c3e50")
        style.configure("TButton", font=("Helvetica", 10), padding=10)
        style.configure("Predict.TButton", font=("Helvetica", 12, "bold"), padding=15)
        style.configure("TCombobox", font=("Helvetica", 10), padding=5)
        
        # Load models and data
        self.load_models()
        
        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create tabs
        self.prediction_tab = ttk.Frame(self.notebook)
        self.visualization_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.prediction_tab, text="Prediction")
        self.notebook.add(self.visualization_tab, text="Visualizations")
        
        # Setup the prediction tab
        self.setup_prediction_tab()
        
        # Setup the visualization tab
        self.setup_visualization_tab()
    
    def load_models(self):
        try:
            # Load models and scaler
            self.rf_model = joblib.load('rf_model.joblib')
            self.svr_model = joblib.load('svr_model.joblib')
            self.gbr_model = joblib.load('gbr_model.joblib')
            self.scaler = joblib.load('scaler.joblib')
            self.selected_features = joblib.load('selected_features.joblib')
            self.top_features = joblib.load('top_features.joblib')
            self.final_selected_features = joblib.load('final_selected_features.joblib')
            
            # Load data for visualizations
            self.df = pd.read_csv("Total_Data (1).csv")
            self.df = self.df.loc[:, ~self.df.columns.str.contains('Unnamed')]
            
            # Calculate derived features
            self.df['Literacy_gap'] = self.df["Men (age 15-49) who are literate4 (%)"] - self.df["Women (age 15-49) who are literate4 (%)"]
            self.df['Nutrition_score'] = (self.df['Children under 5 years who are stunted (height-for-age)18 (%)'] + 
                                        self.df['Children under 5 years who are underweight (weight-for-age)18 (%)']) / 2
            
            # Drop missing target values
            self.df_cleaned = self.df.dropna(subset=["Infant mortality rate (per 1000 live births)"])
            
            print("Models and data loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
            self.root.destroy()
    
    def setup_prediction_tab(self):
        # Main frame
        main_frame = ttk.Frame(self.prediction_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title with decorative line
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(title_frame, text="Infant Mortality Rate Prediction", style="Title.TLabel")
        title_label.pack(side=tk.LEFT)
        
        # Create a canvas with scrollbar for the feature inputs
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(canvas_frame, bg="#f0f0f0", highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Feature entry fields
        self.feature_entries = {}
        
        # Get median values from the dataset for default values
        median_values = self.df[self.selected_features].median().to_dict()
        
        # Create input fields for each feature
        for i, feature in enumerate(self.selected_features):
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, pady=5, padx=10)
            
            # Create a shorter display name
            display_name = feature
            if len(display_name) > 60:
                display_name = display_name[:57] + "..."
            
            label = ttk.Label(frame, text=display_name, wraplength=400)
            label.pack(side=tk.LEFT, padx=(0, 10))
            
            default_value = round(median_values.get(feature, 0), 2)
            entry = ttk.Entry(frame, width=10)
            entry.insert(0, str(default_value))
            entry.pack(side=tk.RIGHT)
            
            self.feature_entries[feature] = entry
        
        # Add model selection dropdown
        model_frame = ttk.Frame(main_frame)
        model_frame.pack(fill=tk.X, pady=20)
        
        ttk.Label(model_frame, text="Select Model:", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="Random Forest")
        model_dropdown = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                    values=["Random Forest", "Gradient Boosting", "SVR"],
                                    state="readonly", width=20)
        model_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Add predict button and result display
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=20)
        
        predict_button = ttk.Button(button_frame, text="Predict Infant Mortality Rate", 
                                  command=self.predict, style="Predict.TButton")
        predict_button.pack(pady=10)
        
        result_frame = ttk.LabelFrame(main_frame, text="Prediction Result", padding=20)
        result_frame.pack(fill=tk.X, pady=20)
        
        self.result_label = ttk.Label(result_frame, text="", font=("Helvetica", 12))
        self.result_label.pack(pady=10)
    
    def setup_visualization_tab(self):
        # Create a frame for the visualizations
        viz_frame = ttk.Frame(self.visualization_tab)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(viz_frame, text="Data Visualizations", style="Title.TLabel")
        title_label.pack(pady=(0, 20))
        
        # Create buttons for different visualizations
        btn_frame = ttk.Frame(viz_frame)
        btn_frame.pack(fill=tk.X, pady=20)
        
        # Style for visualization buttons
        style = ttk.Style()
        style.configure("Viz.TButton", font=("Helvetica", 10), padding=10)
        
        correlation_btn = ttk.Button(btn_frame, text="Correlation Heatmap", 
                                   command=lambda: self.show_visualization("correlation"),
                                   style="Viz.TButton")
        correlation_btn.pack(side=tk.LEFT, padx=5)
        
        importance_btn = ttk.Button(btn_frame, text="Feature Importance", 
                                 command=lambda: self.show_visualization("importance"),
                                 style="Viz.TButton")
        importance_btn.pack(side=tk.LEFT, padx=5)
        
        scatter_btn = ttk.Button(btn_frame, text="Vaccination vs Mortality", 
                              command=lambda: self.show_visualization("scatter"),
                              style="Viz.TButton")
        scatter_btn.pack(side=tk.LEFT, padx=5)
        
        # Frame for the plot
        self.plot_frame = ttk.Frame(viz_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=20)
    
    def show_visualization(self, viz_type):
        # Clear previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        fig = plt.Figure(figsize=(10, 8), dpi=100)
        fig.patch.set_facecolor('#f0f0f0')
        
        if viz_type == "correlation":
            ax = fig.add_subplot(111)
            sns.heatmap(self.df_cleaned[self.top_features.tolist() + ["Infant mortality rate (per 1000 live births)"]].corr(), 
                       annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title("Correlation Heatmap (Top Features)", pad=20)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
        elif viz_type == "importance":
            ax = fig.add_subplot(111)
            importance_sorted = sorted(zip(self.rf_model.feature_importances_, self.top_features))
            sns.barplot(x=[x[0] for x in importance_sorted], y=[x[1] for x in importance_sorted], ax=ax)
            ax.set_title("Feature Importance", pad=20)
            ax.set_xlabel("Importance")
            ax.set_ylabel("Features")
            
        elif viz_type == "scatter":
            ax = fig.add_subplot(111)
            sns.scatterplot(data=self.df_cleaned, 
                           x="Children age 12-23 months who have received 3 doses of penta or DPT vaccine (%)",
                           y="Infant mortality rate (per 1000 live births)", ax=ax)
            ax.set_title("Vaccination Rate vs. Infant Mortality Rate", pad=20)
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
            
            # Get selected model
            selected_model = self.model_var.get()
            
            # Make prediction based on selected model
            if selected_model == "Random Forest":
                # Scale all features for RF
                input_scaled = self.scaler.transform(input_df)
                prediction = self.rf_model.predict(input_scaled)[0]
                model_info = "Model used: Random Forest Regressor"
            elif selected_model == "SVR":
                # Scale all features for SVR
                input_scaled = self.scaler.transform(input_df)
                prediction = self.svr_model.predict(input_scaled)[0]
                model_info = "Model used: Support Vector Regression"
            else:  # Gradient Boosting
                # Select only the 10 features GBR was trained on (no scaling needed as per analysis script)
                input_df_selected = input_df[self.final_selected_features]
                prediction = self.gbr_model.predict(input_df_selected)[0]
                model_info = "Model used: Gradient Boosting Regressor"
            
            # Display result
            self.result_label.config(text=f"Predicted Infant Mortality Rate: {prediction:.2f} per 1000 live births")
            
            # Show additional information
            info_text = (
                f"Prediction details:\n"
                f"- {model_info}\n"
                f"- Important factors: {', '.join(self.top_features[:3])}"
            )
            messagebox.showinfo("Prediction Details", info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {str(e)}")
            print(f"Error details: {e}")

if __name__ == "__main__":
    root = ThemedTk(theme="arc")  # Using a modern theme
    app = InfantMortalityPredictorApp(root)
    root.mainloop() 