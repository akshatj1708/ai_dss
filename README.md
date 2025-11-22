# LuminaML 🔍

**The Transparent AI Assurance & Evaluation Framework**

LuminaML is a sophisticated, modular backend system designed to perform a "360-degree audit" of Machine Learning models and datasets. Unlike standard analysis tools, it integrates **Mistral AI** to provide qualitative insights alongside quantitative metrics, ensuring your AI systems are not only accurate but also fair, robust, and explainable.

---

## 🌟 Core Framework Capabilities

The core of the Framework is its ability to perform specific, automated verification tasks. These functions aim to provide an end-to-end workflow for AI assurance.



[Image of AI model evaluation workflow diagram]


### ● Data Quality Analysis
Performs deep statistical analysis on datasets to detect quality issues. It validates data schemas, identifies outliers, and measures distribution drift between training and testing sets to flag potential data integrity problems before they affect the model.

### ● Model Performance Evaluation
Calculates and benchmarks a suite of standard performance metrics. For classification models, this includes **AUC-ROC**, **Precision**, **Recall**, **F1-Score**, and **Accuracy**, providing a clear picture of predictive power.

### ● Fairness and Bias Auditing
Leverages libraries like **Fairlearn** to conduct a quantitative audit for hidden biases. It calculates key fairness metrics such as **Demographic Parity** and **Equalized Odds** across user-defined demographic subgroups to uncover inequitable model behavior.

### ● Robustness and Security Stress-Testing
Proactively tests model resilience using frameworks like the **Adversarial Robustness Toolbox (ART)**. It generates adversarial examples to probe for vulnerabilities and performs data corruption tests (e.g., injecting noise) to measure how gracefully performance degrades.

### ● Explainability Reporting
Addresses the "black box" problem by using techniques like **SHAP** and **LIME**. It generates feature importance plots and reports that explain which input features most influenced the model's predictions, providing crucial transparency for debugging and user trust.

---

## 🛠️ System Architecture

LuminaML is built as a modular **FastAPI** application that orchestrates various specialized agents.

| Module | Description |
| :--- | :--- |
| **`app.py`** | The REST API Gateway. Manages user sessions and routes requests to specific audit modules. |
| **`agent_core.py`** | The Intelligence Layer. Manages context and interfaces with **Mistral AI** to interpret complex metrics into human-readable reports. |
| **`file_processor.py`** | Handles ingestion of CSV, Excel, PDF, DOCX, and Images (via OCR). |
| **`data_preprocessor.py`** | Cleans data, imputes missing values, and uses LLM logic to infer correct data types. |
| **`fairness_bias_auditor.py`** | Dedicated module for calculating disparity metrics across sensitive groups. |
| **`robustness_tester.py`** | Security module that simulates attacks (FGSM, PGD) and noise injection. |
| **`explainability_reporting.py`** | Wrapper for SHAP/LIME to generate local and global explanations. |

---

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (required for image data processing)
* Mistral AI API Key

### Installation

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/your-org/lumina-ml.git](https://github.com/your-org/lumina-ml.git)
   cd lumina-ml
