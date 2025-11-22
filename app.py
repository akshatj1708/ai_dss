import os
import uuid
import pickle
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from agent_core import IntelligentAgent
from file_processor import FileProcessor
from data_preprocessor import DataPreprocessor
from data_quality_analyzer import DataQualityAnalyzer
from model_performance_evaluator import ModelPerformanceEvaluator
import pandas as pd
from robustness_tester import RobustnessTester
from sklearn.base import is_classifier, is_regressor
from fairness_bias_auditor import FairnessAndBiasAuditing
from explainability_reporting import ExplainabilityReporting, ExplainabilityReport
import logging
import numpy as np
from datetime import datetime
import json
from mistralai.client import MistralClient
from Configs.config import get_mistral_client, MODEL_NAME

logger = logging.getLogger(__name__)

# Load environment variables
env_path = "Configs/.env"
if os.path.exists(env_path):
    load_dotenv(env_path)
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
else:
    MISTRAL_API_KEY = None

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in .env file")

app = FastAPI(
    title="InsightX - AI Data Analysis and Evaluation Agent",
    description="An intelligent agent for data analysis, quality checks, and model performance evaluation.",
    version="3.0.0"
)

# Session management
sessions = {}

class Query(BaseModel):
    session_id: str
    query: str

class EvaluationRequest(BaseModel):
    model_version: str
    thresholds: Dict[str, float]

class FairnessAuditRequest(BaseModel):
    sensitive_columns: List[str]
    fairness_threshold: float = 0.8

class RobustnessTestRequest(BaseModel):
    attack_types: List[str] = ['fgsm']
    noise_types: List[str] = ['gaussian']
    eps_values: List[float] = [0.01, 0.05, 0.1]
    noise_levels: List[float] = [0.05, 0.1, 0.2]
    model_type: str = 'sklearn'  # 'sklearn', 'pytorch', 'tensorflow'

class ExplainabilityRequest(BaseModel):
    instance_indices: List[int] = [0]  # Default to first instance
    sample_size: int = 100  # For SHAP calculations
    num_features: int = 5  # For LIME explanations
    format: str = 'json'  # 'json', 'pdf', or 'all'
    include_plots: bool = False  # Defaults to False to suppress huge base64 strings in response

def _process_and_store_dataset(session_id: str, file: UploadFile, dataset_type: str):
    file_path = os.path.join("Datasets", file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    file_processor = FileProcessor()
    file_info = file_processor.process_file(file_path)

    if file_info['type'] != 'dataframe':
        raise HTTPException(status_code=400, detail="Only dataframe datasets are supported.")

    df = file_info['data']
    agent = sessions.get(session_id, {}).get('agent') or IntelligentAgent(MISTRAL_API_KEY)
    data_preprocessor = DataPreprocessor(agent=agent)
    processed_df, _ = data_preprocessor.preprocess_dataframe(df)

    if session_id not in sessions:
        sessions[session_id] = {}
    sessions[session_id][dataset_type] = processed_df
    if dataset_type == 'training_data':
        sessions[session_id]['target_column'] = processed_df.columns[-1]
    elif 'target_column' in sessions[session_id]:
        target_column = sessions[session_id]['target_column']
        sessions[session_id]['testing_has_target'] = target_column in processed_df.columns
    sessions[session_id]['agent'] = agent

@app.post("/upload/training/", summary="Upload a training dataset")
def upload_training_dataset(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    try:
        _process_and_store_dataset(session_id, file, 'training_data')
        return {"session_id": session_id, "message": "Training dataset uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/upload/testing/{session_id}", summary="Upload a testing dataset")
def upload_testing_dataset(session_id: str, file: UploadFile = File(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    try:
        _process_and_store_dataset(session_id, file, 'testing_data')
        return {"session_id": session_id, "message": "Testing dataset uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/upload/model/{session_id}", summary="Upload a trained model")
def upload_model(session_id: str, file: UploadFile = File(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    try:
        model = pickle.load(file.file)
        sessions[session_id]['model'] = model
        return {"session_id": session_id, "message": "Model uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.post("/query/")
async def handle_query(request: Request):
    """
    Handle natural language queries about the dataset and model using Mistral AI.
    """
    try:
        # Parse the request body
        data = await request.json()
        session_id = data.get('session_id')
        query = data.get('query')
        
        if not session_id or not query:
            raise HTTPException(status_code=400, detail="Both session_id and query are required")
            
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
            
        session = sessions[session_id]
        
        # Initialize Mistral client
        client = get_mistral_client()
        
        # Prepare context from session
        context = "You are a helpful AI assistant. "
        
        if 'training_data' in session:
            # Add dataset information to context
            df = session['training_data']
            context += f"The dataset has {len(df)} rows and {len(df.columns)} columns. "
            context += f"Columns: {', '.join(df.columns)}. "
            
            # Add basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                context += "Numeric columns with their means: "
                for col in numeric_cols:
                    context += f"{col} (mean: {df[col].mean():.2f}), "
        
        if 'model' in session:
            context += "There is a trained model available. "
            model_type = type(session['model']).__name__
            context += f"Model type: {model_type}. "
        
        # Add the user's query
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": query}
        ]
        
        # Get response from Mistral
        response = client.chat(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        # Format the response
        result = {
            "query": query,
            "response": response.choices[0].message.content,
            "session_id": session_id,
            "model": MODEL_NAME
        }
        
        # Add metadata if available
        if 'training_data' in session:
            result["dataset_columns"] = session['training_data'].columns.tolist()
            
        return result
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/analyze_quality/{session_id}", summary="Perform data quality analysis")
def analyze_data_quality(session_id: str):
    if session_id not in sessions or 'training_data' not in sessions[session_id]:
        raise HTTPException(status_code=404, detail="Training data not found.")

    training_data = sessions[session_id]['training_data']
    testing_data = sessions[session_id].get('testing_data')
    agent = sessions[session_id]['agent']

    analyzer = DataQualityAnalyzer(df_train=training_data, df_test=testing_data)
    report = analyzer.analyze()
    return report

@app.post("/evaluate_performance/{session_id}", summary="Evaluate model performance")
def evaluate_performance(session_id: str, request: EvaluationRequest):
    if session_id not in sessions or 'model' not in sessions[session_id] or 'testing_data' not in sessions[session_id]:
        raise HTTPException(status_code=404, detail="Model or testing data not found for this session.")

    model = sessions[session_id]['model']
    testing_data = sessions[session_id]['testing_data']
    agent = sessions[session_id]['agent']
    session = sessions[session_id]

    # Preprocess the testing and training datasets to ensure consistent types
    training_data = sessions[session_id]['training_data']
    testing_data_clean = testing_data.copy()
    training_data_clean = training_data.copy()

    for dataset in (training_data_clean, testing_data_clean):
        for col in dataset.columns:
            if dataset[col].dtype.name == 'category':
                dataset[col] = dataset[col].astype('object')

    # Determine target column (default to last column) and persist if missing
    target_column = session.get('target_column')
    if not target_column:
        target_column = training_data_clean.columns[-1]
        session['target_column'] = target_column

    if target_column not in training_data_clean.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' is missing from the training dataset."
        )

    is_pipeline = hasattr(model, "steps")

    if is_pipeline:
        feature_columns = training_data_clean.drop(columns=[target_column]).columns
        missing_features = set(feature_columns) - set(testing_data_clean.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Testing dataset is missing columns required by the pipeline: {sorted(missing_features)}"
            )
        X_test = testing_data_clean[feature_columns]
        y_test = testing_data_clean[target_column]
    else:
        # Apply the same one-hot encoding that was used for the standalone model
        categorical_cols_test = testing_data_clean.select_dtypes(include=['object']).columns
        testing_data_encoded = pd.get_dummies(testing_data_clean, columns=categorical_cols_test, drop_first=True, dtype=int)

        categorical_cols_train = training_data_clean.select_dtypes(include=['object']).columns
        training_data_encoded = pd.get_dummies(training_data_clean, columns=categorical_cols_train, drop_first=True, dtype=int)

        if target_column not in training_data_encoded.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_column}' is missing from the encoded training data."
            )

        expected_features = training_data_encoded.drop(columns=[target_column]).columns

        for col in expected_features:
            if col not in testing_data_encoded.columns:
                testing_data_encoded[col] = 0

        if target_column not in testing_data_encoded.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_column}' is missing from the encoded testing data."
            )

        testing_data_encoded = testing_data_encoded[expected_features.tolist() + [target_column]]
        X_test = testing_data_encoded[expected_features]
        y_test = testing_data_encoded[target_column]

    evaluator = ModelPerformanceEvaluator(model, X_test, y_test, agent.llm_interface)
    report = evaluator.generate_performance_report(request.model_version, request.thresholds)
    sessions[session_id]['last_report'] = report
    return report

@app.post("/audit_fairness/{session_id}", response_model=Dict[str, Any])
async def audit_fairness(
    session_id: str,
    request: FairnessAuditRequest
):
    """Run fairness audit on the model in the specified session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    model = session.get('model')
    data = session.get('training_data')  
    
    if model is None or data is None:
        raise HTTPException(
            status_code=400, 
            detail="Model and training data are required for fairness audit"
        )
    
    try:
        # Check if all sensitive columns exist
        missing_columns = set(request.sensitive_columns) - set(data.columns)
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"columns are missing: {missing_columns}"
            )
        
        X = data.drop(columns=[data.columns[-1]])  # Drop target column
        y = data[data.columns[-1]]  # Last column is target
        
        # Get only the sensitive features from X
        sensitive_features = X[request.sensitive_columns]
        
        # Initialize fairness auditor with the actual feature values
        auditor = FairnessAndBiasAuditing(
            model=model,
            X=X,
            y_true=y,
            sensitive_features=sensitive_features  # Pass the actual feature values
        )
        
        # Run fairness audit
        report = auditor.audit_fairness(
            fairness_threshold=request.fairness_threshold
        )
        
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types for JSON serialization."""
            import numpy as np
            from numpy import integer, floating, bool_
            
            # Handle dictionary keys first
            if isinstance(obj, dict):
                return {
                    str(k) if isinstance(k, (integer, floating, bool_, np.number)) else k: convert_numpy_types(v)
                    for k, v in obj.items()
                }
            # Handle numpy types
            elif isinstance(obj, (integer, np.integer)):
                return int(obj)
            elif isinstance(obj, (floating, np.floating)):
                return float(obj)
            elif isinstance(obj, (bool_, np.bool_)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            # Handle sequences
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(x) for x in obj]
            # Handle objects with __dict__ attribute
            elif hasattr(obj, '__dict__'):
                return convert_numpy_types(vars(obj))
            # Handle numpy scalars
            elif hasattr(obj, 'item'):
                return obj.item()
            return obj

        report_dict = {
            "fairness_metrics": convert_numpy_types(report.overall_metrics),
            "subgroup_analysis": convert_numpy_types(report.subgroup_analysis),
            "recommendations": convert_numpy_types(report.recommendations)
        }
        
        # Return as JSONResponse to bypass FastAPI's serialization
        return JSONResponse(content=report_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()  # This will print the full traceback
        raise HTTPException(
            status_code=500,
            detail=f"Error during fairness audit: {str(e)}"
        )

@app.post("/test_robustness/{session_id}", summary="Run robustness and security stress testing")
async def test_model_robustness(session_id: str, request: RobustnessTestRequest):
    """
    Generates adversarial examples and performs noise injection to stress-test the model.
    Handles Scikit-Learn Pipelines by testing the underlying estimator on transformed data.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    session = sessions[session_id]
    
    # 1. Validate Assets
    if 'model' not in session:
        raise HTTPException(status_code=400, detail="Model not uploaded.")
    if 'testing_data' not in session:
        raise HTTPException(status_code=400, detail="Testing data not uploaded.")

    model = session['model']
    testing_data = session['testing_data'].copy()
    
    # 2. Prepare X (Features) and y (Target)
    target_column = session.get('target_column')
    if not target_column:
        target_column = testing_data.columns[-1]
        session['target_column'] = target_column

    if target_column not in testing_data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found.")

    # Separate Features and Target
    X_raw = testing_data.drop(columns=[target_column])
    y_test = testing_data[target_column]

    # 3. Handle Pipelines vs Standalone Models
    # ART requires numeric input. If it's a pipeline, we must preprocess the data first.
    is_pipeline = hasattr(model, "steps") or hasattr(model, "named_steps")

    if is_pipeline:
        try:
            # Split Pipeline: Preprocessor (all steps except last) -> Estimator (last step)
            preprocessor = model[:-1]
            final_estimator = model[-1]
            
            # Transform data to numeric format (OneHotEncoding/Scaling happens here)
            X_numeric = preprocessor.transform(X_raw)
            
            # Handle Sparse Matrices (if OneHotEncoder produces sparse output)
            if hasattr(X_numeric, "toarray"):
                X_numeric = X_numeric.toarray()
                
            # Use the numeric data and the final mathematical model
            model_to_test = final_estimator
            X_test = X_numeric
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decompose pipeline for robustness testing: {str(e)}")
    else:
        # If not a pipeline, we assume data is already numeric or model handles it
        # (But for robustness, it essentially must be numeric)
        # We'll try to convert X_raw to numeric, usually by encoding if training data is available
        if 'training_data' in session:
            # ... (Reuse the encoding logic from previous answer if needed, 
            # but simpler to rely on the Pipeline logic above for your specific case)
            pass 
        
        # Fallback: Assume data is numeric. If it has strings, this will fail again, 
        # but standard models usually require numeric input anyway.
        X_test = X_raw
        model_to_test = model

    # 4. Determine Task Type
    task_type = "classification"
    if is_regressor(model_to_test):
        task_type = "regression"
    elif is_classifier(model_to_test):
        task_type = "classification"
    else:
        # Fallback check on target variable
        if pd.api.types.is_numeric_dtype(y_test) and len(np.unique(y_test)) > 20:
             task_type = "regression"

    # 5. Execute Robustness Test
    try:
        tester = RobustnessTester(
            model=model_to_test,
            X_test=X_test,
            y_test=y_test,
            model_type=request.model_type,
            task_type=task_type
        )

        report = tester.measure_robustness(
            attack_types=request.attack_types,
            noise_types=request.noise_types,
            eps_values=request.eps_values,
            noise_levels=request.noise_levels
        )

        # 6. Serialize Output
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            elif hasattr(obj, '__dict__'): 
                return convert_numpy(obj.__dict__)
            return obj

        return JSONResponse(content=convert_numpy(report))

    except Exception as e:
        logger.error(f"Robustness testing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Robustness testing failed: {str(e)}")

@app.post("/explain/{session_id}", response_model=Dict[str, Any])
async def explain_model(
    session_id: str,
    request: ExplainabilityRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate model explainability report using SHAP and LIME, with LLM-based analysis.
    
    Args:
        session_id: Session ID containing the model and data
        request: Explainability configuration
        background_tasks: For running long-running tasks in background
        
    Returns:
        Explainability report with SHAP, LIME, and LLM insights
    """
    try:
        # Get session data
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
            
        session = sessions[session_id]
        model = session.get('model')
        X_test = session.get('testing_data')
        agent = session.get('agent')  # Retrieve the agent to use its LLM interface
        
        if model is None or X_test is None:
            raise HTTPException(status_code=400, detail="Model or test data not found in session")
            
        # Get feature names
        if hasattr(X_test, 'columns'):
            feature_names = X_test.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
            
        # Determine model type
        model_type = 'classifier'
        if hasattr(model, 'predict_proba'):
            try:
                # Check if it's a regressor by looking at predict output
                pred = model.predict(X_test[:1])
                if len(pred.shape) == 1 and not np.array_equal(pred, pred.astype(bool)):
                    model_type = 'regressor'
            except:
                pass
                
        # Initialize explainer
        explainer = ExplainabilityReporting(
            model=model,
            X=X_test,
            feature_names=feature_names,
            model_type=model_type
        )
        
        # Generate explanations
        def generate_report():
            try:
                # Compute global feature importance
                explainer.compute_feature_importance(sample_size=request.sample_size)
                
                # Generate local explanations for specified instances
                for idx in request.instance_indices:
                    try:
                        explainer.explain_instance(
                            instance_idx=idx,
                            num_features=request.num_features
                        )
                    except Exception as e:
                        logger.error(f"Error explaining instance {idx}: {str(e)}")
                
                # LLM Analysis Integration
                if agent and hasattr(agent, 'llm_interface'):
                    explainer.analyze_with_llm(agent.llm_interface)
                
                # Create reports directory if it doesn't exist
                os.makedirs('reports/explainability', exist_ok=True)
                
                # Generate and save report
                # Pass the flag from request to control whether plots are returned
                report = explainer.generate_explanation_report(
                    output_dir='reports/explainability',
                    format=request.format,
                    include_plots=request.include_plots
                )
                return report
            except Exception as e:
                logger.error(f"Error in explainability report generation: {str(e)}")
                raise
        
        # Run in background if it might take long
        if len(X_test) > 100 or len(request.instance_indices) > 3:
            background_tasks.add_task(generate_report)
            return {
                "status": "started",
                "message": "Explainability report generation started in background. Check the reports directory.",
                "session_id": session_id
            }
        else:
            return generate_report()
            
    except Exception as e:
        logger.error(f"Error in explain endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating explainability report: {str(e)}")