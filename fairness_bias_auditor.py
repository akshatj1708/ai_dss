"""
Fairness and Bias Auditing Module

This module provides functionality to audit machine learning models for fairness and bias
across different demographic subgroups using Fairlearn.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

@dataclass
class FairnessReport:
    """Container for fairness audit results.
    
    Attributes:
        overall_metrics: Dictionary of fairness metrics
        subgroup_analysis: Detailed metrics per subgroup
        sensitive_features: Names of sensitive features analyzed
        fairness_threshold: Threshold used for fairness evaluation
        recommendations: Suggestions for addressing any fairness issues
    """
    overall_metrics: Dict[str, float]
    subgroup_analysis: Dict[str, Any]
    sensitive_features: List[str]
    fairness_threshold: float
    recommendations: str = ""

class FairnessAndBiasAuditing:
    """Audits machine learning models for fairness and bias across demographic subgroups.
    
    This class implements fairness metrics including Demographic Parity and Equalized Odds
    to evaluate potential biases in model predictions across different demographic groups.
    """
    
    def __init__(
        self, 
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y_true: Union[pd.Series, np.ndarray],
        sensitive_features: Union[pd.DataFrame, np.ndarray],
        sensitive_feature_names: List[str] = None
    ):
        """Initialize the fairness auditor.
        
        Args:
            model: Trained model with predict() method
            X: Feature matrix used for predictions
            y_true: True labels
            sensitive_features: Sensitive attributes to analyze for fairness
            sensitive_feature_names: Optional names for sensitive features
        """
        self.model = model
        self.X = X
        self.y_true = y_true
        self.sensitive_features = sensitive_features
        
        # Generate predictions if not provided
        self.y_pred = self.model.predict(X)
        
        # Set feature names or generate defaults
        if sensitive_feature_names is not None:
            if len(sensitive_feature_names) != sensitive_features.shape[1]:
                raise ValueError("Length of sensitive_feature_names must match number of sensitive features")
            self.sensitive_feature_names = sensitive_feature_names
        else:
            self.sensitive_feature_names = [f"sensitive_feature_{i}" for i in range(sensitive_features.shape[1])]
    
    def calculate_bias_metrics(self, fairness_threshold: float = 0.8) -> Dict[str, Any]:
        """Calculate key fairness metrics.
        
        Args:
            fairness_threshold: Threshold for determining fairness (0-1)
            
        Returns:
            Dictionary containing fairness metrics and pass/fail status
        """
        try:
            from fairlearn.metrics import (
                demographic_parity_difference,
                equalized_odds_difference
            )
            
            # Calculate fairness metrics
            metrics = {
                'demographic_parity': demographic_parity_difference(
                    self.y_true, 
                    self.y_pred, 
                    sensitive_features=self.sensitive_features
                ),
                'equalized_odds': equalized_odds_difference(
                    self.y_true, 
                    self.y_pred,
                    sensitive_features=self.sensitive_features
                )
            }
            
            # Check against thresholds (closer to 0 is better)
            metrics['is_fair'] = all(
                abs(m) <= (1 - fairness_threshold) 
                for m in metrics.values()
            )
            
            return metrics
            
        except ImportError:
            raise ImportError(
                "Fairlearn is required for bias metrics. "
                "Install with: pip install fairlearn"
            )
    
    def audit_fairness(self, fairness_threshold: float = 0.8) -> FairnessReport:
        """Perform comprehensive fairness audit.
        
        Args:
            fairness_threshold: Threshold for fairness metrics (0-1)
            
        Returns:
            FairnessReport object containing detailed analysis
        """
        # Calculate overall metrics
        metrics = self.calculate_bias_metrics(fairness_threshold)
        
        # Generate subgroup analysis
        subgroup_analysis = self._analyze_subgroups()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, fairness_threshold)
        
        return FairnessReport(
            overall_metrics=metrics,
            subgroup_analysis=subgroup_analysis,
            sensitive_features=self.sensitive_feature_names,
            fairness_threshold=fairness_threshold,
            recommendations=recommendations
        )
    
    def _analyze_subgroups(self) -> Dict[str, Dict[str, float]]:
        """Analyze metrics across different subgroups.
        
        Returns:
            Dictionary with metrics for each subgroup
        """
        if isinstance(self.sensitive_features, pd.DataFrame):
            sensitive_df = self.sensitive_features.copy()
            sensitive_df.columns = self.sensitive_feature_names
        else:
            sensitive_df = pd.DataFrame(
                self.sensitive_features,
                columns=self.sensitive_feature_names
            )
        
        # Add predictions and true labels
        sensitive_df['y_true'] = self.y_true
        sensitive_df['y_pred'] = self.y_pred
        
        subgroup_metrics = {}
        
        # Calculate metrics for each sensitive feature
        for feature in self.sensitive_feature_names:
            metrics = {}
            for group in sensitive_df[feature].unique():
                mask = sensitive_df[feature] == group
                y_true_group = sensitive_df.loc[mask, 'y_true']
                y_pred_group = sensitive_df.loc[mask, 'y_pred']
                
                if len(y_true_group) > 0:  # Only calculate if group has samples
                    metrics[group] = {
                        'accuracy': accuracy_score(y_true_group, y_pred_group),
                        'precision': precision_score(y_true_group, y_pred_group, average='weighted', zero_division=0),
                        'recall': recall_score(y_true_group, y_pred_group, average='weighted', zero_division=0),
                        'sample_size': len(y_true_group)
                    }
            
            subgroup_metrics[feature] = metrics
        
        return subgroup_metrics
    
    def _generate_recommendations(
        self, 
        metrics: Dict[str, float], 
        threshold: float
    ) -> str:
        """Generate recommendations based on fairness metrics.
        
        Args:
            metrics: Dictionary of fairness metrics
            threshold: Fairness threshold used
            
        Returns:
            String containing recommendations
        """
        recommendations = []
        
        # Check demographic parity
        dp_diff = abs(metrics.get('demographic_parity', 0))
        if dp_diff > (1 - threshold):
            recommendations.append(
                f"Significant demographic parity difference detected ({dp_diff:.3f}). "
                "Consider using techniques like reweighing or adversarial debiasing."
            )
        
        # Check equalized odds
        eo_diff = abs(metrics.get('equalized_odds', 0))
        if eo_diff > (1 - threshold):
            recommendations.append(
                f"Significant equalized odds difference detected ({eo_diff:.3f}). "
                "Consider using techniques like equalized odds postprocessing."
            )
        
        # Add general recommendations if no specific issues found
        if not recommendations:
            recommendations.append(
                "No significant fairness issues detected. "
                "Continue monitoring as new data becomes available."
            )
        
        return " ".join(recommendations)

# Example usage
if __name__ == "__main__":
    # Example with synthetic data
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Sensitive feature (e.g., gender)
    gender = np.random.choice([0, 1], size=n_samples)  # 0: female, 1: male
    
    # Create synthetic features with some bias
    X = np.random.randn(n_samples, 5)
    X[:, 0] += gender * 0.5  # Introduce some bias
    
    # Generate labels with some bias
    y = ((X[:, 0] + X[:, 1] * 0.5) > 0).astype(int)
    
    # Train a simple model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Create auditor instance
    auditor = FairnessAndBiasAuditing(
        model=model,
        X=X,
        y_true=y,
        sensitive_features=gender.reshape(-1, 1),
        sensitive_feature_names=['gender']
    )
    
    # Run audit
    report = auditor.audit_fairness(fairness_threshold=0.8)
    
    # Print results
    print("\nFairness Audit Results:")
    print("-" * 30)
    print(f"Demographic Parity: {report.overall_metrics['demographic_parity']:.4f}")
    print(f"Equalized Odds: {report.overall_metrics['equalized_odds']:.4f}")
    print(f"Is Fair: {report.overall_metrics['is_fair']}")
    print("\nSubgroup Analysis:")
    for feature, groups in report.subgroup_analysis.items():
        print(f"\n{feature}:")
        for group, metrics in groups.items():
            print(f"  {group}: Accuracy={metrics['accuracy']:.3f}, "
                  f"Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, "
                  f"n={metrics['sample_size']}")
    
    print("\nRecommendations:")
    print(report.recommendations)
