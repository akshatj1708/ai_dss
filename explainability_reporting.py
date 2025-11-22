from dataclasses import dataclass, asdict
from typing import Dict, List, Union, Optional, Any
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import base64
from io import BytesIO
import logging
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

@dataclass
class ExplainabilityReport:
    """Stores explainability analysis results."""
    global_shap_values: Optional[Dict[str, float]] = None
    local_shap_values: Optional[Dict[int, Dict[str, float]]] = None
    lime_explanations: Optional[Dict[int, Dict[str, float]]] = None
    plots: Dict[str, str] = None  # Store base64 encoded plots
    feature_importance: Dict[str, float] = None
    model_type: str = None
    timestamp: str = None
    metadata: Dict[str, Any] = None
    llm_analysis: Optional[str] = None  # Added field for LLM analysis

    def to_dict(self, include_plots: bool = False) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        data = asdict(self)
        # Remove plots from the dictionary if include_plots is False
        if not include_plots and 'plots' in data:
            del data['plots']
        return data

class ExplainabilityReporting:
    def __init__(self, model, X, feature_names: List[str], model_type: str = 'classifier', preprocessor=None):
        """
        Initialize explainability reporter.
        
        Args:
            model: Trained model (must implement predict or predict_proba)
            X: Feature matrix (pandas DataFrame or numpy array)
            feature_names: List of feature names (must match model expectation)
            model_type: 'classifier' or 'regressor'
            preprocessor: Fitted ColumnTransformer or Pipeline used during training
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.preprocessor = preprocessor
        self.report = ExplainabilityReport(model_type=model_type)
        
        # --- Validate and Align Input Dimensions ---
        # Ensure X only contains the features the model expects
        if hasattr(X, 'columns'):
            try:
                self.X_original = X[feature_names].copy()
            except KeyError as e:
                logger.warning(f"Could not filter X by feature_names: {e}. Using X as is.")
                self.X_original = X.copy()
        else:
            # If numpy array, verify shape
            if X.shape[1] != len(feature_names):
                logger.warning(f"Input X has {X.shape[1]} columns but feature_names has {len(feature_names)}. "
                               "Attempting to slice X to match feature_names length.")
                self.X_original = pd.DataFrame(X[:, :len(feature_names)], columns=feature_names)
            else:
                self.X_original = pd.DataFrame(X, columns=feature_names)

        self.categorical_features = []
        self.categories_ = {}
        self.label_encoders = {}
        
        # Process the input data (Create numeric version for SHAP/LIME)
        self.X_encoded = self._preprocess_data(self.X_original)
        self._validate_inputs()
        
    def _preprocess_data(self, X_df):
        """
        Preprocess input data to handle categorical features.
        Returns a numeric numpy array suitable for SHAP/LIME generation.
        """
        from sklearn.preprocessing import LabelEncoder
        
        X_encoded = X_df.copy()
        
        # Identify categorical columns
        self.categorical_features = X_encoded.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Encode categorical features
        for col in self.categorical_features:
            le = LabelEncoder()
            # Convert to string to handle mixed types or Nones securely
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            self.label_encoders[col] = le
            self.categories_[col] = le.classes_.tolist()
            
        return X_encoded.values.astype(float)
        
    def _validate_inputs(self):
        """Validate input parameters."""
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model must implement predict() method")
            
        if len(self.feature_names) != self.X_encoded.shape[1]:
            raise ValueError(f"Number of feature names ({len(self.feature_names)}) "
                             f"must match number of columns in processed X ({self.X_encoded.shape[1]})")
            
        if self.model_type not in ['classifier', 'regressor']:
            raise ValueError("model_type must be either 'classifier' or 'regressor'")

    def _reconstruct_input(self, X_numeric):
        """
        Helper to convert SHAP's numeric input back to the format the Model expects.
        """
        # Ensure 2D array
        if len(X_numeric.shape) == 1:
            X_numeric = X_numeric.reshape(1, -1)
            
        # 1. Convert to DataFrame with correct column names
        X_df = pd.DataFrame(X_numeric, columns=self.feature_names)
        
        # 2. Decode categorical features back to original strings
        for col, le in self.label_encoders.items():
            try:
                # Round to nearest integer (SHAP might add small noise) and cast to int
                col_indices = X_df[col].round().astype(int)
                
                # Clip indices to be within valid range of the encoder
                n_classes = len(le.classes_)
                col_indices = col_indices.clip(0, n_classes - 1)
                
                # Inverse transform
                X_df[col] = le.inverse_transform(col_indices)
            except Exception as e:
                logger.warning(f"Failed to inverse transform column {col}: {e}")
        
        return X_df

    def compute_feature_importance(self, sample_size: int = 100) -> Dict[str, float]:
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")
        
        try:
            # Sample data if needed
            if sample_size and len(self.X_encoded) > sample_size:
                np.random.seed(42)
                idx = np.random.choice(len(self.X_encoded), sample_size, replace=False)
                X_sample = self.X_encoded[idx]
            else:
                X_sample = self.X_encoded
            
            def predict_fn(X):
                X_df_reconstructed = self._reconstruct_input(X)
                
                # If external preprocessor provided
                if self.preprocessor is not None:
                    try:
                        X_final = self.preprocessor.transform(X_df_reconstructed)
                        if hasattr(self.model, 'predict_proba'):
                            return self.model.predict_proba(X_final)[:, 1] 
                        return self.model.predict(X_final)
                    except:
                        pass 

                # Standard prediction
                if hasattr(self.model, 'predict_proba'):
                    probs = self.model.predict_proba(X_df_reconstructed)
                    if probs.shape[1] == 2:
                        return probs[:, 1]
                    # Handle single column probability output
                    if probs.ndim == 1 or probs.shape[1] == 1:
                        return probs.flatten()
                    return probs
                
                return self.model.predict(X_df_reconstructed)
            
            # Initialize SHAP explainer
            explainer = shap.KernelExplainer(predict_fn, X_sample)
            
            # Calculate SHAP values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_values = explainer.shap_values(X_sample, nsamples=100)
            
            # Process SHAP values
            if isinstance(shap_values, list): 
                global_importance = np.abs(np.array(shap_values)).mean(0).mean(0)
            else:
                global_importance = np.abs(shap_values).mean(0)
            
            feature_importance = dict(zip(self.feature_names, global_importance))
            
            self.report.feature_importance = feature_importance
            self.report.global_shap_values = feature_importance
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error in SHAP calculation: {str(e)}", exc_info=True)
            # Use empty dict on failure to allow process to continue
            return {}
            
    def explain_instance(self, instance_idx: int, num_features: int = 5) -> Dict[str, float]:
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not installed. Install with: pip install lime")
        
        if instance_idx >= len(self.X_encoded):
            raise ValueError(f"Instance index {instance_idx} out of bounds")
        
        try:
            # --- FIX: Robust Prediction Wrapper for LIME ---
            def predict_fn(X):
                X_df_reconstructed = self._reconstruct_input(X)
                
                # Handle Classification vs Regression specifically
                if self.model_type == 'classifier':
                    if hasattr(self.model, 'predict_proba'):
                        probs = self.model.predict_proba(X_df_reconstructed)
                        
                        # ENSURE 2D OUTPUT (N, 2) for binary classification
                        # LIME expects [prob_0, prob_1], but some models return just [prob_1]
                        if probs.ndim == 1:
                            return np.vstack([1 - probs, probs]).T
                        elif probs.shape[1] == 1:
                            return np.hstack([1 - probs, probs])
                        return probs
                    else:
                        # Fallback if model is classifier but lacks predict_proba
                        # Convert hard predictions (0/1) to pseudo-probabilities
                        preds = self.model.predict(X_df_reconstructed)
                        return np.vstack([1 - preds, preds]).T
                else:
                    # Regression case
                    return self.model.predict(X_df_reconstructed)
            
            # Get the instance to explain
            instance = self.X_encoded[instance_idx]
            
            # Initialize LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.X_encoded,
                feature_names=self.feature_names,
                categorical_features=[self.X_original.columns.get_loc(c) for c in self.categorical_features],
                categorical_names=self.categories_,
                class_names=['class_0', 'class_1'] if self.model_type == 'classifier' else None,
                mode='classification' if self.model_type == 'classifier' else 'regression',
                verbose=False
            )
            
            # Generate explanation
            exp = explainer.explain_instance(
                data_row=instance,
                predict_fn=predict_fn,
                num_features=num_features
            )
            
            # Convert to dictionary
            explanation = dict(exp.as_list())
            
            # Store in report
            if self.report.lime_explanations is None:
                self.report.lime_explanations = {}
            self.report.lime_explanations[instance_idx] = explanation
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error in LIME explanation for index {instance_idx}: {str(e)}", exc_info=True)
            return {}
            
    def generate_plots(self) -> Dict[str, str]:
        """Generate and store explainability plots."""
        plots = {}
        
        # Generate SHAP summary plot
        if self.report.global_shap_values:
            plt.figure(figsize=(10, 6))
            sorted_features = sorted(self.report.global_shap_values.items(), key=lambda x: x[1], reverse=True)
            
            sns.barplot(
                x=[x[1] for x in sorted_features],
                y=[x[0] for x in sorted_features]
            )
            plt.title('Global Feature Importance (SHAP)')
            plt.xlabel('mean(|SHAP value|)')
            plt.tight_layout()
            plots['shap_summary'] = self._plot_to_base64()
            plt.close()
            
        # Generate LIME explanation plot
        if hasattr(self.report, 'lime_explanations') and self.report.lime_explanations:
            for idx, exp in list(self.report.lime_explanations.items())[:1]:  # Just first explanation
                plt.figure(figsize=(10, 6))
                features = list(exp.keys())[:10]  # Top 10 features
                values = list(exp.values())[:10]
                
                colors = ['green' if v > 0 else 'red' for v in values]
                
                sns.barplot(x=values, y=features, palette=colors)
                plt.title(f'LIME Explanation for Instance {idx}')
                plt.tight_layout()
                plots[f'lime_instance_{idx}'] = self._plot_to_base64()
                plt.close()
                
        self.report.plots = plots
        return plots

    def _plot_to_base64(self) -> str:
        """Convert current matplotlib figure to base64 string."""
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def analyze_with_llm(self, llm_interface) -> str:
        """
        Send the generated explanation report to the LLM for detailed analysis.
        """
        # Construct the analysis prompt
        prompt = f"""
        You are an expert AI Explainability Analyst. Analyze the following model explanation report.
        
        Model Type: {self.model_type}
        
        Global Feature Importance (SHAP):
        {json.dumps(self.report.global_shap_values, indent=2) if self.report.global_shap_values else "Not available"}
        
        Local Explanations (LIME) for specific instances:
        {json.dumps(self.report.lime_explanations, indent=2) if self.report.lime_explanations else "Not available"}
        
        Please provide a detailed analysis covering:
        1. **Global Drivers**: What are the most important features driving the model's decisions overall?
        2. **Local Insights**: For the specific instances explained, why did the model make those predictions? 
           Highlight any conflicting features (e.g., a feature that usually pushes positive but pushed negative here).
        3. **Trust & Bias**: Are there any features being used that seem counterintuitive or potential sources of bias?
        4. **Summary**: A brief executive summary of the model's behavior.
        
        Format the response in Markdown.
        """
        
        try:
            system_prompt = "You are a helpful data science assistant specializing in interpreting machine learning models."
            analysis = llm_interface.generate_response(prompt, system_prompt)
            self.report.llm_analysis = analysis
            return analysis
        except Exception as e:
            logger.error(f"Error generating LLM analysis: {str(e)}")
            return "LLM Analysis unavailable due to an error."

    def generate_explanation_report(self, output_dir: str = None, format: str = 'json', include_plots: bool = False) -> Dict[str, Any]:
        """Generate a comprehensive explanation report."""
        import os
        from datetime import datetime
        
        self.report.timestamp = datetime.now().isoformat()
        
        if hasattr(self, 'categorical_features') and self.categorical_features:
            self.report.metadata = {
                'categorical_features': self.categorical_features,
                'categories': self.categories_
            }
        
        # Generate plots (generate internally regardless of return flag, to ensure existence)
        self.generate_plots()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format in ['json', 'all']:
                report_path = os.path.join(output_dir, f'explanation_report_{timestamp}.json')
                with open(report_path, 'w') as f:
                    # Save file WITH plots so the record is complete on disk
                    json.dump(self.report.to_dict(include_plots=True), f, indent=2)
            
        # Return dict, filtering plots based on request
        return self.report.to_dict(include_plots=include_plots)