import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib

class InfantMortalityPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Infant Mortality Rate Predictor")
        self.root.geometry("1000x800")
        self.root.configure(bg="#f0f0f0")
        
        # Load models and data
        self.load_models()
        
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
    
    def load_models(self):
        try:
            self.rf_model = joblib.load('rf_model.joblib')
            self.svr_model = joblib.load('svr_model.joblib')
            self.gbr_model = joblib.load('gbr_model.joblib')
            self.scaler = joblib.load('scaler.joblib')
            self.selected_features = joblib.load('selected_features.joblib')
            self.top_features = joblib.load('top_features.joblib')
            
            # Load data for visualizations
            self.df = pd.read_csv("Total_Data (1).csv")
            self.df = self.df.loc[:, ~self.df.columns.str.contains('Unnamed')]
            self.df_cleaned = self.df.dropna(subset=["Infant mortality rate (per 1000 live births)"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
            self.root.destroy()
    
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
        median_values = self.df[self.selected_features].median().to_dict()
        
        # Create input fields for each feature
        for i, feature in enumerate(self.selected_features):
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
            sns.heatmap(self.df_cleaned[self.top_features.tolist() + ["Infant mortality rate (per 1000 live births)"]].corr(), 
                       annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title("Correlation Heatmap (Top Features)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
        elif viz_type == "importance":
            ax = fig.add_subplot(111)
            importance_sorted = sorted(zip(self.rf_model.feature_importances_, self.top_features))
            sns.barplot(x=[x[0] for x in importance_sorted], y=[x[1] for x in importance_sorted], ax=ax)
            ax.set_title("Feature Importance")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Features")
            
        elif viz_type == "scatter":
            ax = fig.add_subplot(111)
            sns.scatterplot(data=self.df_cleaned, 
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
            input_scaled = self.scaler.transform(input_df)
            
            # Get selected model
            selected_model = self.model_var.get()
            
            # Make prediction based on selected model
            if selected_model == "Random Forest":
                prediction = self.rf_model.predict(input_scaled)[0]
                model_info = "Model used: Random Forest Regressor"
            elif selected_model == "SVR":
                prediction = self.svr_model.predict(input_scaled)[0]
                model_info = "Model used: Support Vector Regression"
            else:  # Gradient Boosting
                prediction = self.gbr_model.predict(input_scaled)[0]
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
    root = tk.Tk()
    app = InfantMortalityPredictorApp(root)
    root.mainloop() 