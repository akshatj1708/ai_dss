"""
Robustness and Security Stress-Testing Module

This module provides functionality to test model robustness against adversarial attacks
and noisy inputs using the Adversarial Robustness Toolbox (ART).
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Union, Optional
from dataclasses import dataclass, field
import logging
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import pandas as pd

# Remove the entire try-except block and replace with this:
try:
    from art.estimators.classification import SklearnClassifier, PyTorchClassifier, TensorFlowV2Classifier
    from art.estimators.regression import ScikitlearnRegressor, PyTorchRegressor
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method
    from art.defences.preprocessor import FeatureSqueezing, GaussianAugmentation
    from art.metrics import clever_u
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False

# Configure logging
import logging
logger = logging.getLogger(__name__)

@dataclass
class RobustnessReport:
    """
    Container for robustness testing results.
    """
    original_accuracy: float
    adversarial_accuracy: Dict[str, float] = field(default_factory=dict)
    noise_robustness_score: float = 0.0
    vulnerability_insights: List[str] = field(default_factory=list)
    robustness_metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

class RobustnessTester:
    """
    Class for testing model robustness against adversarial attacks and noise.
    Supports both classification and regression models.
    """
    
    def __init__(
        self,
        model: Any,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        model_type: str = 'sklearn',  # 'sklearn', 'pytorch', 'tensorflow'
        task_type: str = 'classification',  # 'classification' or 'regression'
        feature_ranges: Optional[Tuple[float, float]] = None,
        **kwargs
    ):
        if not ART_AVAILABLE:
            raise ImportError(
                "Adversarial Robustness Toolbox (ART) is required but not installed. "
                "Install with: pip install adversarial-robustness-toolbox"
            )
        """
        Initialize the robustness tester.
        
        Args:
            model: Trained model to test
            X_test: Test features (numpy array or pandas DataFrame)
            y_test: Test labels (numpy array or pandas Series)
            model_type: Type of model ('sklearn', 'pytorch', 'tensorflow')
            task_type: Type of task ('classification' or 'regression')
            feature_ranges: Tuple of (min, max) values for feature scaling
            **kwargs: Additional model-specific parameters
        """
        self.model = model
        self.X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        self.y_test = y_test.values if hasattr(y_test, 'values') else y_test
        self.model_type = model_type.lower()
        self.task_type = task_type.lower()
        self.feature_ranges = feature_ranges or (float(self.X_test.min()), float(self.X_test.max()))
        self.art_model = self._wrap_model(**kwargs)
        
        # Initialize results
        self.original_predictions = None
        self.adversarial_examples = None
        self.noisy_samples = None
        self.attack_success_rates = {}
        
    def _wrap_model(self, **kwargs) -> Any:
        """Wrap the model for use with ART based on model and task type."""
        if self.task_type == 'classification':
            return self._wrap_classification_model(**kwargs)
        elif self.task_type == 'regression':
            return self._wrap_regression_model(**kwargs)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}. Use 'classification' or 'regression'")
    
    def _wrap_classification_model(self, **kwargs) -> Any:
        """Wrap a classification model for ART."""
        if self.model_type == 'sklearn':
            from art.estimators.classification import SklearnClassifier
            return SklearnClassifier(
                model=self.model,
                clip_values=self.feature_ranges,
                **kwargs
            )
        elif self.model_type == 'pytorch':
            import torch
            from art.estimators.classification import PyTorchClassifier
            return PyTorchClassifier(
                model=self.model,
                clip_values=self.feature_ranges,
                loss=torch.nn.CrossEntropyLoss(),
                optimizer=torch.optim.Adam(self.model.parameters()),
                input_shape=self.X_test.shape[1:],
                nb_classes=len(np.unique(self.y_test)),
                **kwargs
            )
        elif self.model_type == 'tensorflow':
            from art.estimators.classification import TensorFlowV2Classifier
            return TensorFlowV2Classifier(
                model=self.model,
                clip_values=self.feature_ranges,
                nb_classes=len(np.unique(self.y_test)),
                input_shape=self.X_test.shape[1:],
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type for classification: {self.model_type}")

    def _wrap_regression_model(self, **kwargs) -> Any:
        """Wrap a regression model for ART."""
        if self.model_type == 'sklearn':
            from art.estimators.regression import ScikitlearnRegressor
            return ScikitlearnRegressor(
                model=self.model,
                clip_values=self.feature_ranges,
                **kwargs
            )
        elif self.model_type == 'pytorch':
            import torch
            from art.estimators.regression import PyTorchRegressor
            return PyTorchRegressor(
                model=self.model,
                clip_values=self.feature_ranges,
                loss=torch.nn.MSELoss(),
                optimizer=torch.optim.Adam(self.model.parameters()),
                input_shape=self.X_test.shape[1:],
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type for regression: {self.model_type}")
            
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
        if self.task_type == 'regression':
            logger.warning("Adversarial attacks are not well-defined for regression models. Using noise injection instead.")
            return self.inject_noise('gaussian', eps), 0.0

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
    
    def measure_robustness(
        self,
        attack_types: List[str] = None,
        noise_types: List[str] = None,
        eps_values: List[float] = None,
        noise_levels: List[float] = None
    ) -> RobustnessReport:
        """
        Measure model robustness against various attacks and noise.
        
        Args:
            attack_types: List of attack types to test
            noise_types: List of noise types to test
            eps_values: List of epsilon values for attacks
            noise_levels: List of noise levels for noise injection
            
        Returns:
            RobustnessReport containing test results
        """
        # Set default values if not provided
        if attack_types is None:
            attack_types = ['fgsm', 'pgd'] if self.task_type == 'classification' else []
        if noise_types is None:
            noise_types = ['gaussian', 'uniform']
        if eps_values is None:
            eps_values = [0.01, 0.05, 0.1]
        if noise_levels is None:
            noise_levels = [0.05, 0.1, 0.2]

        try:
            # Calculate original score
            y_pred = self.model.predict(self.X_test)
            if self.task_type == 'classification':
                original_score = accuracy_score(self.y_test, y_pred)
            else:  # regression
                original_score = r2_score(self.y_test, y_pred)

            # Initialize results
            adversarial_scores = {}
            noise_robustness = {noise_type: {} for noise_type in noise_types}
            vulnerability_insights = []
            robustness_metrics = {
                'task_type': self.task_type,
                'feature_ranges': self.feature_ranges
            }
            recommendations = []

            # Adversarial testing (only for classification)
            if self.task_type == 'classification' and attack_types:
                for attack_type in attack_types:
                    for eps in eps_values:
                        try:
                            _, success_rate = self.generate_adversarial_samples(
                                attack_type=attack_type,
                                eps=eps
                            )
                            adversarial_scores[f"{attack_type}_eps{eps}"] = 1 - success_rate
                        except Exception as e:
                            logger.warning(f"Failed to generate {attack_type} attack with eps={eps}: {str(e)}")
                            adversarial_scores[f"{attack_type}_eps{eps}"] = 0.0
            else:
                vulnerability_insights.append(
                    "Adversarial attacks are not performed (only supported for classification models)."
                )

            # Noise injection testing
            for noise_type in noise_types:
                for level in noise_levels:
                    try:
                        X_noisy = self.inject_noise(noise_type, level)
                        y_pred_noisy = self.model.predict(X_noisy)
                        
                        if self.task_type == 'classification':
                            score = accuracy_score(self.y_test, y_pred_noisy)
                        else:  # regression
                            score = r2_score(self.y_test, y_pred_noisy)
                            
                        noise_robustness[noise_type][str(level)] = score
                    except Exception as e:
                        logger.warning(f"Failed to test with {noise_type} noise (level={level}): {str(e)}")
                        noise_robustness[noise_type][str(level)] = 0.0

            # Calculate overall noise robustness score
            if noise_robustness:
                all_scores = [score for noise_scores in noise_robustness.values() 
                            for score in noise_scores.values()]
                noise_robustness_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
            else:
                noise_robustness_score = 0.0

            # Generate insights and recommendations
            if self.task_type == 'classification':
                if adversarial_scores:
                    avg_adv_score = sum(adversarial_scores.values()) / len(adversarial_scores)
                    if avg_adv_score < 0.7 * original_score:
                        vulnerability_insights.append("Model shows significant vulnerability to adversarial attacks.")
                        recommendations.append("Consider implementing adversarial training or using a more robust model architecture.")
                    else:
                        vulnerability_insights.append("Model shows reasonable resistance to adversarial attacks.")

            # Add noise-related insights
            if noise_robustness_score < 0.7 * original_score:
                vulnerability_insights.append("Model is sensitive to input noise.")
                recommendations.append("Consider adding noise to the training data to improve robustness.")
            else:
                vulnerability_insights.append("Model shows good resistance to input noise.")

            if not recommendations:
                recommendations.append("No specific recommendations. Model shows good robustness properties.")

            # Update robustness metrics
            robustness_metrics.update({
                'original_score': original_score,
                'noise_robustness': noise_robustness,
                'adversarial_scores': adversarial_scores if self.task_type == 'classification' else None
            })

            return RobustnessReport(
                original_accuracy=float(original_score),
                adversarial_accuracy=adversarial_scores,
                noise_robustness_score=float(noise_robustness_score),
                vulnerability_insights=vulnerability_insights,
                robustness_metrics=robustness_metrics,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Error in measure_robustness: {str(e)}", exc_info=True)
            raise