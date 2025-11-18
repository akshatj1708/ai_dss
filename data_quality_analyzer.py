import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency

class DataQualityAnalyzer:
    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame = None):
        self.df_train = df_train
        self.df_test = df_test
        self.report = {}

    def analyze(self):
        """Run all data quality checks."""
        self.report['schema_validation'] = self._validate_schema()
        self.report['missing_values'] = self._check_missing_values()
        self.report['outlier_detection'] = self._detect_outliers()
        if self.df_test is not None:
            self.report['distribution_drift'] = self._check_distribution_drift()
        return self.report

    def _validate_schema(self):
        """Validate that the test schema matches the training schema."""
        if self.df_test is None:
            return "Skipping schema validation: No test dataset provided."
        
        train_cols = set(self.df_train.columns)
        test_cols = set(self.df_test.columns)

        if train_cols != test_cols:
            return {
                'status': 'failed',
                'missing_in_test': list(train_cols - test_cols),
                'extra_in_test': list(test_cols - train_cols)
            }
        return {'status': 'passed'}

    def _check_missing_values(self):
        """Check for missing values in the dataset(s)."""
        results = {'training': self.df_train.isnull().sum().to_dict() }
        if self.df_test is not None:
            results['testing'] = self.df_test.isnull().sum().to_dict()
        return results

    def _detect_outliers(self):
        """Detect outliers using the IQR method."""
        outliers = {}
        for col in self.df_train.select_dtypes(include=np.number).columns:
            Q1 = self.df_train[col].quantile(0.25)
            Q3 = self.df_train[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = self.df_train[(self.df_train[col] < lower_bound) | (self.df_train[col] > upper_bound)].shape[0]
            if outlier_count > 0:
                outliers[col] = outlier_count
        return outliers

    def _check_distribution_drift(self):
        """Check for distribution drift between training and testing datasets."""
        if self.df_test is None:
            return "Skipping distribution drift: No test dataset provided."

        drift_results = {}
        for col in self.df_train.columns:
            if pd.api.types.is_numeric_dtype(self.df_train[col]):
                stat, p_value = ks_2samp(self.df_train[col], self.df_test[col])
                drift_results[col] = {'type': 'numeric', 'p_value': p_value, 'drift': bool(p_value < 0.05)}
            else:
                contingency_table = pd.crosstab(self.df_train[col], self.df_test[col])
                chi2, p_value, _, _ = chi2_contingency(contingency_table)
                drift_results[col] = {'type': 'categorical', 'p_value': p_value, 'drift': bool(p_value < 0.05)}
        return drift_results
