import os
import uuid
import pickle
from fastapi import FastAPI, File, UploadFile, HTTPException,BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict,List, Optional,Any
from dotenv import load_dotenv
from agent_core import IntelligentAgent
from file_processor import FileProcessor
from data_preprocessor import DataPreprocessor
from data_quality_analyzer import DataQualityAnalyzer
from model_performance_evaluator import ModelPerformanceEvaluator
import pandas as pd
from pydantic import BaseModel
from robustness_tester import RobustnessTester
from fairness_bias_auditor import FairnessAndBiasAuditing
import logging
import numpy as np

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

@app.get("/analyze/quality/{session_id}", summary="Perform data quality analysis")
def analyze_data_quality(session_id: str):
    if session_id not in sessions or 'training_data' not in sessions[session_id]:
        raise HTTPException(status_code=404, detail="Training data not found.")

    training_data = sessions[session_id]['training_data']
    testing_data = sessions[session_id].get('testing_data')
    agent = sessions[session_id]['agent']

    analyzer = DataQualityAnalyzer(df_train=training_data, df_test=testing_data, llm_interface=agent.llm_interface)
    report = analyzer.analyze()
    return report

@app.post("/evaluate/performance/{session_id}", summary="Evaluate model performance")
def evaluate_performance(session_id: str, request: EvaluationRequest):
    if session_id not in sessions or 'model' not in sessions[session_id] or 'testing_data' not in sessions[session_id]:
        raise HTTPException(status_code=404, detail="Model or testing data not found for this session.")

    model = sessions[session_id]['model']
    testing_data = sessions[session_id]['testing_data']
    agent = sessions[session_id]['agent']

    # Preprocess the testing data to match training format exactly
    # Training script only applies one-hot encoding, no DataPreprocessor
    testing_data_clean = testing_data.copy()
    
    # Convert category dtypes back to object for get_dummies to work
    for col in testing_data_clean.columns:
        if testing_data_clean[col].dtype.name == 'category':
            testing_data_clean[col] = testing_data_clean[col].astype('object')
    
    # Debug: Print testing data columns before encoding
    print(f"Testing data columns before encoding: {testing_data_clean.columns.tolist()}")
    
    # Apply the same one-hot encoding as training script
    categorical_cols = testing_data_clean.select_dtypes(include=['object']).columns
    testing_data_encoded = pd.get_dummies(testing_data_clean, columns=categorical_cols, drop_first=True, dtype=int)
    
    # Debug: Print testing data columns after encoding
    print(f"Testing data columns after encoding: {testing_data_encoded.columns.tolist()}")
    
    # Load the training data to get the expected columns
    training_data = sessions[session_id]['training_data']
    
    # Convert category dtypes back to object for training data too
    training_data_clean = training_data.copy()
    for col in training_data_clean.columns:
        if training_data_clean[col].dtype.name == 'category':
            training_data_clean[col] = training_data_clean[col].astype('object')
    
    # Debug: Print training data columns before encoding
    print(f"Training data columns before encoding: {training_data_clean.columns.tolist()}")
    
    # Re-process training data with the same encoding to get expected columns
    categorical_cols_train = training_data_clean.select_dtypes(include=['object']).columns
    training_data_encoded = pd.get_dummies(training_data_clean, columns=categorical_cols_train, drop_first=True, dtype=int)
    
    # Debug: Print training data columns after encoding
    print(f"Training data columns after encoding: {training_data_encoded.columns.tolist()}")
    
    # Get expected feature columns (excluding target)
    expected_features = training_data_encoded.drop('price', axis=1).columns
    
    # Debug: Print expected features
    print(f"Expected features: {expected_features.tolist()}")
    
    # Add missing columns to testing data with 0 values
    for col in expected_features:
        if col not in testing_data_encoded.columns:
            testing_data_encoded[col] = 0
    
    # Ensure same column order plus target
    testing_data_encoded = testing_data_encoded[expected_features.tolist() + ['price']]
    
    # Debug: Print final testing data columns
    print(f"Final testing data columns: {testing_data_encoded.columns.tolist()}")
    
    # Assuming the last column is the target variable
    X_test = testing_data_encoded.iloc[:, :-1]
    y_test = testing_data_encoded.iloc[:, -1]

    evaluator = ModelPerformanceEvaluator(model, X_test, y_test, agent.llm_interface)
    report = evaluator.generate_performance_report(request.model_version, request.thresholds)
    sessions[session_id]['last_report'] = report
    return report

@app.post("/audit/fairness/{session_id}", response_model=Dict[str, Any])
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

@app.post("/test/robustness/{session_id}")
async def test_robustness(
    session_id: str,
    request: RobustnessTestRequest,
    background_tasks: BackgroundTasks
):
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        model = session.get('model')
        X_test = session.get('testing_data')
        preprocessor = session.get('preprocessor')
        
        if model is None or X_test is None:
            raise HTTPException(status_code=400, detail="Model or test data not found in session")
        
        try:
            # Get target column from session or use the last column as default
            target_column = session.get('target_column', X_test.columns[-1])
            y_test = X_test[target_column].values
            X_test = X_test.drop(columns=[target_column])
            
            logger.info(f"Raw test data columns: {X_test.columns.tolist()}")
            logger.info(f"Raw test data shape: {X_test.shape}")
            
            # Apply the same preprocessing used during training
            if preprocessor is not None:
                try:
                    # Get the expected feature names from the preprocessor
                    if hasattr(preprocessor, 'get_feature_names_out'):
                        expected_features = preprocessor.get_feature_names_out()
                        logger.info(f"Preprocessor expects features: {expected_features.tolist()}")
                    
                    # Get the actual features in the test data
                    actual_features = X_test.columns.tolist()
                    logger.info(f"Actual features: {actual_features}")
                    
                    # If the model is a pipeline, get the preprocessor from the pipeline
                    if hasattr(model, 'steps'):
                        for name, step in model.steps:
                            if hasattr(step, 'transform') and hasattr(step, 'fit_transform'):
                                preprocessor = step
                                logger.info(f"Found preprocessor in model pipeline: {name}")
                                break
                    
                    # Transform the test data
                    X_test_processed = preprocessor.transform(X_test)
                    logger.info(f"Preprocessed test data shape: {getattr(X_test_processed, 'shape', 'N/A')}")
                    
                except Exception as e:
                    logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
                    if hasattr(preprocessor, 'get_feature_names_out'):
                        logger.error(f"Expected features: {preprocessor.get_feature_names_out().tolist()}")
                    logger.error(f"Actual features: {X_test.columns.tolist()}")
                    if hasattr(preprocessor, 'transformers_'):
                        for name, transformer, columns in preprocessor.transformers_:
                            logger.error(f"Transformer '{name}' processes columns: {columns}")
                    raise ValueError(
                        f"Preprocessing failed: {str(e)}\n"
                        f"Expected {getattr(preprocessor, 'n_features_in_', 'unknown')} features, got {X_test.shape[1]}\n"
                        f"Input columns: {X_test.columns.tolist()}"
                    )
                
                # If we have a sparse matrix, convert to dense for ART
                if hasattr(X_test_processed, 'toarray'):
                    X_test_processed = X_test_processed.toarray()
            else:
                # Fallback: Use the model's predict method directly if no preprocessor is available
                X_test_processed = X_test.values
                logger.info(f"Using raw test data (no preprocessor): {X_test_processed.shape}")
            
            # Get feature ranges for scaling
            if hasattr(X_test_processed, 'shape'):
                logger.info(f"Final test data shape: {X_test_processed.shape}")
                if isinstance(X_test_processed, np.ndarray):
                    feature_ranges = (float(np.nanmin(X_test_processed)), 
                                    float(np.nanmax(X_test_processed)))
                elif hasattr(X_test_processed, 'min') and hasattr(X_test_processed, 'max'):
                    feature_ranges = (float(X_test_processed.min().min()), 
                                    float(X_test_processed.max().max()))
                else:
                    feature_ranges = (0.0, 1.0)
            else:
                feature_ranges = (0.0, 1.0)
                
            # Initialize robustness tester with task_type from request
            tester = RobustnessTester(
                model=model,
                X_test=X_test_processed,
                y_test=y_test,
                model_type='sklearn',
                task_type=request.model_type,
                feature_ranges=feature_ranges
            )
            
            # Run robustness tests
            report = tester.measure_robustness(
                attack_types=request.attack_types,
                noise_types=request.noise_types,
                noise_levels=request.noise_levels
            )
            
            # Convert NumPy types to native Python types for JSON serialization
            report_dict = convert_numpy_types(report.__dict__)
            return report_dict
            
        except Exception as e:
            logger.error(f"Error during robustness testing: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error during robustness testing: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in test_robustness: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")