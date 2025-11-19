import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
from pydantic import BaseModel
from typing import Dict, Any
from agent_core import LLMInterface

class PerformanceReport(BaseModel):
    model_version: str
    metrics: Dict[str, float]
    benchmarks: Dict[str, bool]
    passed_all_benchmarks: bool
    summary: str

class ModelPerformanceEvaluator:
    def __init__(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, llm_interface: LLMInterface):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.llm_interface = llm_interface
        self.y_pred = model.predict(X_test)
        
        # Check if it's a classification or regression model
        self.is_classification = hasattr(model, 'predict_proba')
        if self.is_classification:
            try:
                self.y_pred_proba = model.predict_proba(X_test)[:, 1]
            except:
                # If predict_proba fails, treat as regression
                self.is_classification = False

    def evaluateModel(self) -> Dict[str, float]:
        """
        Calculates metrics based on model type (classification or regression).
        """
        if self.is_classification:
            metrics = {
                'accuracy': accuracy_score(self.y_test, self.y_pred),
                'precision': precision_score(self.y_test, self.y_pred, average='binary', zero_division=0),
                'recall': recall_score(self.y_test, self.y_pred, average='binary', zero_division=0),
                'f1_score': f1_score(self.y_test, self.y_pred, average='binary', zero_division=0),
                'auc_roc': roc_auc_score(self.y_test, self.y_pred_proba)
            }
        else:
            # Regression metrics
            metrics = {
                'mse': mean_squared_error(self.y_test, self.y_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_test, self.y_pred)),
                'mae': mean_absolute_error(self.y_test, self.y_pred),
                'r2_score': r2_score(self.y_test, self.y_pred)
            }
        return metrics

    def benchmarkPerformance(self, metrics: Dict[str, float], thresholds: Dict[str, float]) -> Dict[str, bool]:
        """
        Compares calculated metrics against configurable performance thresholds.
        For regression: lower error metrics (MSE, RMSE, MAE) are better, higher R² is better
        For classification: higher metrics are better
        """
        benchmark_results = {}
        for metric, threshold_value in thresholds.items():
            if metric in metrics:
                if metric in ['mse', 'rmse', 'mae']:
                    # For error metrics, lower is better
                    benchmark_results[metric] = metrics[metric] <= threshold_value
                else:
                    # For accuracy, R², etc., higher is better
                    benchmark_results[metric] = metrics[metric] >= threshold_value
        return benchmark_results

    def generate_performance_report(self, model_version: str, thresholds: Dict[str, float]) -> PerformanceReport:
        """
        Generates a complete performance report by evaluating and benchmarking the model.
        """
        metrics = self.evaluateModel()
        benchmarks = self.benchmarkPerformance(metrics, thresholds)
        passed_all = all(benchmarks.values())
        llm_summary = self._get_llm_interpretation(metrics, benchmarks, thresholds)

        report = PerformanceReport(
            model_version=model_version,
            metrics=metrics,
            benchmarks=benchmarks,
            passed_all_benchmarks=passed_all,
            summary=llm_summary
        )
        return report

    def _get_llm_interpretation(self, metrics: Dict[str, float], benchmarks: Dict[str, bool], thresholds: Dict[str, float]) -> str:
        """
        Uses the LLM to generate a human-readable interpretation of the performance metrics.
        """
        if self.is_classification:
            prompt = f"""
            You are an expert Machine Learning model evaluator. Analyze the following classification model performance report and provide a concise, insightful summary.

            **Model Performance Metrics:**
            - Accuracy: {metrics.get('accuracy', 'N/A'):.4f}
            - Precision: {metrics.get('precision', 'N/A'):.4f}
            - Recall: {metrics.get('recall', 'N/A'):.4f}
            - F1-Score: {metrics.get('f1_score', 'N/A'):.4f}
            - AUC-ROC: {metrics.get('auc_roc', 'N/A'):.4f}

            **Performance Benchmarks:**
            The model's performance was benchmarked against the following thresholds:
            - Accuracy > {thresholds.get('accuracy', 'N/A')}: {'Passed' if benchmarks.get('accuracy') else 'Failed'}
            - Precision > {thresholds.get('precision', 'N/A')}: {'Passed' if benchmarks.get('precision') else 'Failed'}
            - Recall > {thresholds.get('recall', 'N/A')}: {'Passed' if benchmarks.get('recall') else 'Failed'}

            **Your Task:**
            1.  **Overall Assessment:** Start with a one-sentence summary of the model's performance. Did it meet the required benchmarks?
            2.  **Metric Interpretation:** Briefly explain what the key metrics (like Precision and Recall) mean in this context.
            3.  **Actionable Recommendations:** Based on the results, suggest 1-2 concrete next steps.

            Provide a professional and clear summary.
            """
        else:
            # Regression prompt
            prompt = f"""
            You are an expert Machine Learning model evaluator. Analyze the following regression model performance report and provide a concise, insightful summary.

            **Model Performance Metrics:**
            - Mean Squared Error (MSE): {metrics.get('mse', 'N/A'):.4f}
            - Root Mean Squared Error (RMSE): {metrics.get('rmse', 'N/A'):.4f}
            - Mean Absolute Error (MAE): {metrics.get('mae', 'N/A'):.4f}
            - R² Score: {metrics.get('r2_score', 'N/A'):.4f}

            **Performance Benchmarks:**
            The model's performance was benchmarked against the following thresholds:
            - MSE < {thresholds.get('mse', 'N/A')}: {'Passed' if benchmarks.get('mse') else 'Failed'}
            - RMSE < {thresholds.get('rmse', 'N/A')}: {'Passed' if benchmarks.get('rmse') else 'Failed'}
            - MAE < {thresholds.get('mae', 'N/A')}: {'Passed' if benchmarks.get('mae') else 'Failed'}
            - R² Score > {thresholds.get('r2_score', 'N/A')}: {'Passed' if benchmarks.get('r2_score') else 'Failed'}

            **Your Task:**
            1.  **Overall Assessment:** Start with a one-sentence summary of the model's performance. Did it meet the required benchmarks?
            2.  **Metric Interpretation:** Briefly explain what the key metrics mean. For example, RMSE represents the typical prediction error in the same units as the target variable, and R² indicates the proportion of variance explained by the model.
            3.  **Actionable Recommendations:** Based on the results, suggest 1-2 concrete next steps for improving the model or proceeding to deployment.

            Provide a professional and clear summary.
            """
        system_prompt = "You are a helpful AI assistant specializing in machine learning model evaluation."
        interpretation = self.llm_interface.generate_response(prompt, system_prompt)
        return interpretation
