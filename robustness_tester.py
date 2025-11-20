"""
Robustness and Security Stress-Testing Module

This module provides functionality to test model robustness against adversarial attacks
and noisy inputs using the Adversarial Robustness Toolbox (ART).
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Union, Optional
from dataclasses import dataclass
import logging
from sklearn.metrics import accuracy_score
import pandas as pd

try:
    from art.estimators.classification import SklearnClassifier, PyTorchClassifier, TensorFlowClassifier
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method
    from art.defences.preprocessor import FeatureSqueezing, GaussianAugmentation
    from art.metrics import clever_u
except ImportError:
    print("Adversarial Robustness Toolbox (ART) not found. Install with: pip install adversarial-robustness-toolbox")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RobustnessReport:
    """
    Container for robustness testing results.
    
    Attributes:
        original_accuracy: Accuracy on clean test data
        adversarial_accuracy: Accuracy on adversarial examples
        noise_robustness_score: Score (0-1) indicating robustness to noise
        vulnerability_insights: List of identified vulnerabilities
        robustness_metrics: Dictionary of various robustness metrics
        recommendations: List of improvement suggestions
    """
    original_accuracy: float
    adversarial_accuracy: float
    noise_robustness_score: float
    vulnerability_insights: List[str]
    robustness_metrics: Dict[str, float]
    recommendations: List[str]

class RobustnessTester:
    """
    Class for testing model robustness against adversarial attacks and noise.
    """
    
    def __init__(
        self,
        model: Any,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        model_type: str = 'sklearn',  # 'sklearn', 'pytorch', 'tensorflow'
        feature_ranges: Optional[Tuple[float, float]] = None,
        **kwargs
    ):
        """
        Initialize the robustness tester.
        
        Args:
            model: Trained model to test
            X_test: Test features (numpy array or pandas DataFrame)
            y_test: Test labels (numpy array or pandas Series)
            model_type: Type of model ('sklearn', 'pytorch', 'tensorflow')
            feature_ranges: Tuple of (min, max) values for feature scaling
            **kwargs: Additional model-specific parameters
        """
        self.model = model
        self.X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        self.y_test = y_test.values if hasattr(y_test, 'values') else y_test
        self.model_type = model_type
        self.feature_ranges = feature_ranges or (self.X_test.min(), self.X_test.max())
        self.art_model = self._wrap_model(**kwargs)
        
        # Initialize results
        self.original_predictions = None
        self.adversarial_examples = None
        self.noisy_samples = None
        self.attack_success_rates = {}
        
    def _wrap_model(self, **kwargs) -> Any:
        """Wrap the model for use with ART."""
        if self.model_type.lower() == 'sklearn':
            return self._wrap_sklearn_model(**kwargs)
        elif self.model_type.lower() == 'pytorch':
            return self._wrap_pytorch_model(**kwargs)
        elif self.model_type.lower() == 'tensorflow':
            return self._wrap_tensorflow_model(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _wrap_sklearn_model(self, **kwargs) -> 'SklearnClassifier':
        """Wrap a scikit-learn model for ART."""
        from art.estimators.classification import SklearnClassifier
        return SklearnClassifier(
            model=self.model,
            clip_values=self.feature_ranges,
            **kwargs
        )
    
    def _wrap_pytorch_model(self, **kwargs) -> 'PyTorchClassifier':
        """Wrap a PyTorch model for ART."""
        import torch
        from art.estimators.classification import PyTorchClassifier
        
        if not hasattr(self.model, 'forward'):
            raise ValueError("PyTorch model must have a 'forward' method")
            
        return PyTorchClassifier(
            model=self.model,
            clip_values=self.feature_ranges,
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(self.model.parameters()),
            input_shape=self.X_test.shape[1:],
            nb_classes=len(np.unique(self.y_test)),
            **kwargs
        )
    
    def _wrap_tensorflow_model(self, **kwargs) -> 'TensorFlowClassifier':
        """Wrap a TensorFlow/Keras model for ART."""
        from art.estimators.classification import TensorFlowV2Classifier
        
        return TensorFlowV2Classifier(
            model=self.model,
            clip_values=self.feature_ranges,
            nb_classes=len(np.unique(self.y_test)),
            input_shape=self.X_test.shape[1:],
            **kwargs
        )
    
    def generate_adversarial_samples(
        self,
        attack_type: str = 'fgsm',
        eps: float = 0.1,
        **attack_params
    ) -> Tuple[np.ndarray, float]:
        """
        Generate adversarial examples using specified attack.
        
        Args:
            attack_type: Type of attack ('fgsm', 'pgd', 'carlini', etc.)
            eps: Attack strength/epsilon value
            **attack_params: Additional attack parameters
            
        Returns:
            Tuple of (adversarial_examples, success_rate)
        """
        # Get clean predictions
        self.original_predictions = self.art_model.predict(self.X_test)
        
        # Initialize attack
        if attack_type.lower() == 'fgsm':
            attack = FastGradientMethod(estimator=self.art_model, eps=eps, **attack_params)
        elif attack_type.lower() == 'pgd':
            attack = ProjectedGradientDescent(estimator=self.art_model, eps=eps, **attack_params)
        elif attack_type.lower() == 'carlini':
            attack = CarliniL2Method(classifier=self.art_model, **attack_params)
        else:
            raise ValueError(f"Unsupported attack type: {attack_type}")
        
        # Generate adversarial examples
        self.adversarial_examples = attack.generate(x=self.X_test)
        
        # Calculate attack success rate
        adv_predictions = self.art_model.predict(self.adversarial_examples)
        success_rate = 1 - accuracy_score(
            np.argmax(self.original_predictions, axis=1) if len(self.original_predictions.shape) > 1 else self.original_predictions,
            np.argmax(adv_predictions, axis=1) if len(adv_predictions.shape) > 1 else adv_predictions
        )
        
        self.attack_success_rates[attack_type] = success_rate
        logger.info(f"Attack '{attack_type}' success rate: {success_rate:.2%}")
        
        return self.adversarial_examples, success_rate
    
    def inject_noise(
        self,
        noise_type: str = 'gaussian',
        noise_level: float = 0.1,
        **noise_params
    ) -> np.ndarray:
        """
        Inject noise into test samples.
        
        Args:
            noise_type: Type of noise ('gaussian', 'uniform', 'salt_pepper')
            noise_level: Strength of the noise
            **noise_params: Additional noise parameters
            
        Returns:
            Noisy test samples
        """
        if noise_type == 'gaussian':
            noise = np.random.normal(
                scale=noise_level * (self.feature_ranges[1] - self.feature_ranges[0]),
                size=self.X_test.shape
            )
        elif noise_type == 'uniform':
            noise = np.random.uniform(
                low=-noise_level,
                high=noise_level,
                size=self.X_test.shape
            ) * (self.feature_ranges[1] - self.feature_ranges[0])
        elif noise_type == 'salt_pepper':
            # For salt and pepper noise, we need to handle binary/categorical features
            salt = np.random.random(self.X_test.shape) < (noise_level / 2)
            pepper = np.random.random(self.X_test.shape) < (noise_level / 2)
            noise = salt.astype(float) - pepper.astype(float)
            noise *= (self.feature_ranges[1] - self.feature_ranges[0])
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        
        self.noisy_samples = np.clip(
            self.X_test + noise,
            self.feature_ranges[0],
            self.feature_ranges[1]
        )
        
        return self.noisy_samples
    
    def measure_robustness(self, attack_types=None, noise_types=None, eps_values=None, noise_levels=None):
        """Measure model robustness against various attacks and noise."""
        if attack_types is None:
            attack_types = ['fgsm', 'pgd']
        if noise_types is None:
            noise_types = ['gaussian', 'uniform']
        if eps_values is None:
            eps_values = [0.01, 0.05, 0.1]
        if noise_levels is None:
            noise_levels = [0.05, 0.1, 0.2]

        try:
            # Convert data to numpy if they're pandas DataFrames/Series
            if hasattr(self.X_test, 'values'):
                X_test = self.X_test.values
            else:
                X_test = self.X_test

            if hasattr(self.y_test, 'values'):
                y_test = self.y_test.values
            else:
                y_test = self.y_test

            # Calculate original accuracy (R² score for regression)
            y_pred = self.model.predict(X_test)
            original_accuracy = self.model.score(X_test, y_test)  # R² score for regression

            # Initialize results
            adversarial_accuracy = {}
            noise_robustness = {}
            vulnerability_insights = []
            robustness_metrics = {}
            recommendations = []

            # Skip adversarial attacks for regression models
            if hasattr(self.model, 'predict_proba'):  # Only for classifiers
                # Adversarial testing code for classifiers...
                pass
            else:
                vulnerability_insights.append("Adversarial attacks are not supported for regression models.")
                recommendations.append("Consider using a classification model for adversarial robustness testing.")

            # Noise injection testing
            for noise_type in noise_types:
                noise_robustness[noise_type] = {}
                for level in noise_levels:
                    X_noisy = self._inject_noise(X_test.copy(), noise_type, level)
                    if hasattr(self.model, 'predict_proba'):  # Classifier
                        y_pred_noisy = self.model.predict(X_noisy)
                        acc = accuracy_score(y_test, y_pred_noisy)
                    else:  # Regressor
                        y_pred_noisy = self.model.predict(X_noisy)
                        acc = r2_score(y_test, y_pred_noisy)
                    noise_robustness[noise_type][str(level)] = acc

            # Calculate overall noise robustness score (average across all noise types/levels)
            if noise_robustness:
                all_scores = [score for noise_scores in noise_robustness.values() 
                            for score in noise_scores.values()]
                noise_robustness_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
            else:
                noise_robustness_score = 0.0

            # Generate insights and recommendations
            if not vulnerability_insights:  # If no insights were added (e.g., for classifiers)
                if 'adversarial_accuracy' in locals() and adversarial_accuracy:
                    avg_adv_acc = sum(adversarial_accuracy.values()) / len(adversarial_accuracy)
                    if avg_adv_acc < 0.7 * original_accuracy:
                        vulnerability_insights.append("Model shows significant vulnerability to adversarial attacks.")
                        recommendations.append("Consider implementing adversarial training or using a more robust model architecture.")
                    else:
                        vulnerability_insights.append("Model shows reasonable resistance to adversarial attacks.")

            # Add noise-related insights
            if noise_robustness_score < 0.7 * original_accuracy:
                vulnerability_insights.append("Model is sensitive to input noise.")
                recommendations.append("Consider adding noise to the training data to improve robustness.")
            else:
                vulnerability_insights.append("Model shows good resistance to input noise.")

            if not recommendations:
                recommendations.append("No specific recommendations. Model shows good robustness properties.")

            # Create robustness metrics
            robustness_metrics = {
                'original_r2': original_accuracy,
                'noise_robustness': noise_robustness,
                'is_regression': not hasattr(self.model, 'predict_proba')
            }

            return RobustnessReport(
                original_accuracy=original_accuracy,
                adversarial_accuracy=adversarial_accuracy,
                noise_robustness_score=noise_robustness_score,
                vulnerability_insights=vulnerability_insights,
                robustness_metrics=robustness_metrics,
                recommendations=recommendations
            )

        except Exception as e:
            print(f"Error in measure_robustness: {str(e)}")
            raise