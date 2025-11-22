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

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return asdict(self)

class ExplainabilityReporting:
    def __init__(self, model, X: pd.DataFrame, feature_names: List[str], model_type: str = 'classifier'):
        """
        Initialize explainability reporter.
        
        Args:
            model: Trained model (must implement predict or predict_proba)
            X: Feature matrix (pandas DataFrame or numpy array)
            feature_names: List of feature names
            model_type: 'classifier' or 'regressor'
        """
        self.model = model
        self.X = X.values if hasattr(X, 'values') else X
        self.feature_names = feature_names
        self.model_type = model_type
        self.report = ExplainabilityReport(model_type=model_type)
        self._validate_inputs()
        
    def _validate_inputs(self):
        """Validate input parameters."""
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model must implement predict() method")
            
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError("Number of feature names must match number of columns in X")
            
        if self.model_type not in ['classifier', 'regressor']:
            raise ValueError("model_type must be either 'classifier' or 'regressor'")

    def compute_feature_importance(self, sample_size: int = 100) -> Dict[str, float]:
        """
        Compute global feature importance using SHAP.
        
        Args:
            sample_size: Number of samples to use for SHAP (None for all)
            
        Returns:
            Dictionary of {feature_name: importance_score}
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")
            
        # Sample data if large dataset
        if sample_size and len(self.X) > sample_size:
            np.random.seed(42)
            idx = np.random.choice(len(self.X), sample_size, replace=False)
            X_sample = self.X[idx]
        else:
            X_sample = self.X

        # Initialize SHAP explainer based on model type
        if hasattr(self.model, 'predict_proba'):
            explainer = shap.Explainer(self.model.predict_proba, X_sample)
        else:
            explainer = shap.Explainer(self.model.predict, X_sample)
            
        # Calculate SHAP values
        shap_values = explainer(X_sample)
        
        # Store global feature importance
        if len(shap_values.shape) > 2:  # For multi-class
            global_importance = np.abs(shap_values.values).mean(0).mean(0)
        else:
            global_importance = np.abs(shap_values.values).mean(0)
            
        self.report.feature_importance = dict(zip(self.feature_names, global_importance))
        self.report.global_shap_values = self.report.feature_importance
        
        return self.report.feature_importance

    def explain_instance(self, instance_idx: int, num_features: int = 5) -> Dict[str, float]:
        """
        Generate local explanation for a single instance using LIME.
        
        Args:
            instance_idx: Index of instance to explain
            num_features: Number of top features to include in explanation
            
        Returns:
            Dictionary of {feature_name: importance_score} for the instance
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not installed. Install with: pip install lime")
            
        if instance_idx >= len(self.X):
            raise ValueError(f"Instance index {instance_idx} out of bounds")
            
        # Initialize LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X,
            feature_names=self.feature_names,
            class_names=['class_0', 'class_1'] if self.model_type == 'classifier' else None,
            mode='classification' if self.model_type == 'classifier' else 'regression'
        )
        
        # Get prediction function
        predict_fn = (self.model.predict_proba if hasattr(self.model, 'predict_proba') 
                     else self.model.predict)
        
        # Generate explanation
        exp = explainer.explain_instance(
            self.X[instance_idx],
            predict_fn,
            num_features=num_features
        )
        
        # Store explanation
        explanation = dict(exp.as_list())
        if not hasattr(self.report, 'lime_explanations'):
            self.report.lime_explanations = {}
        self.report.lime_explanations[instance_idx] = explanation
        
        return explanation

    def generate_plots(self) -> Dict[str, str]:
        """Generate and store explainability plots."""
        plots = {}
        
        # Generate SHAP summary plot
        if self.report.global_shap_values:
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=list(self.report.global_shap_values.values()),
                y=list(self.report.global_shap_values.keys())
            )
            plt.title('Global Feature Importance (SHAP)')
            plt.tight_layout()
            plots['shap_summary'] = self._plot_to_base64()
            plt.close()
            
        # Generate LIME explanation plot for first instance if available
        if hasattr(self.report, 'lime_explanations') and self.report.lime_explanations:
            for idx, exp in list(self.report.lime_explanations.items())[:1]:  # Just first explanation
                plt.figure(figsize=(10, 6))
                features = list(exp.keys())[:10]  # Top 10 features
                values = list(exp.values())[:10]
                sns.barplot(x=values, y=features)
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

    def generate_explanation_report(self, output_dir: str = None, 
                                  format: str = 'json') -> Dict[str, Any]:
        """
        Generate a comprehensive explanation report.
        
        Args:
            output_dir: Directory to save reports (if None, only return dict)
            format: Output format ('json', 'pdf', or 'all')
            
        Returns:
            Dictionary containing report data
        """
        import time
        from datetime import datetime
        
        # Add timestamp
        self.report.timestamp = datetime.now().isoformat()
        
        # Generate plots
        self.generate_plots()
        
        # Prepare report data
        report_data = self.report.to_dict()
        
        # Save to files if output directory is provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format in ['json', 'all']:
                json_path = output_dir / f'explanation_report_{timestamp}.json'
                with open(json_path, 'w') as f:
                    json.dump(report_data, f, indent=2)
                    
            if format in ['pdf', 'all'] and self.report.plots:
                from matplotlib.backends.backend_pdf import PdfPages
                pdf_path = output_dir / f'explanation_report_{timestamp}.pdf'
                
                with PdfPages(pdf_path) as pdf:
                    for plot_name, plot_data in self.report.plots.items():
                        # Create a figure for each plot
                        fig = plt.figure(figsize=(10, 6))
                        # Add title
                        plt.title(plot_name.replace('_', ' ').title())
                        # Add the image
                        img = base64.b64decode(plot_data)
                        img = plt.imread(BytesIO(img))
                        plt.imshow(img)
                        plt.axis('off')
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close()
        
        return report_data