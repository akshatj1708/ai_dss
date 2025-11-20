import os
import uuid
import pickle
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Dict
from dotenv import load_dotenv
from agent_core import IntelligentAgent
from file_processor import FileProcessor
from data_preprocessor import DataPreprocessor
from data_quality_analyzer import DataQualityAnalyzer
from model_performance_evaluator import ModelPerformanceEvaluator
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel

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

class FairnessAuditRequest(BaseModel):
    sensitive_columns: List[str]
    fairness_threshold: float = 0.8

class Query(BaseModel):
    session_id: str
    query: str

class EvaluationRequest(BaseModel):
    model_version: str
    thresholds: Dict[str, float]

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

@app.post("/audit/fairness/{session_id}", summary="Perform fairness audit on model predictions")
async def audit_fairness(
    session_id: str,
    request: FairnessAuditRequest
):
    """Run fairness audit on the model and data in the session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    model = session.get('model')
    testing_data = session.get('testing_data')
    
    if model is None or testing_data is None:
        raise HTTPException(
            status_code=400, 
            detail="Model and test data are required for fairness audit"
        )
    
    # Extract sensitive features
    sensitive_cols = [col for col in request.sensitive_columns if col in testing_data.columns]
    if not sensitive_cols:
        raise HTTPException(
            status_code=400,
            detail="No valid sensitive columns found in test data"
        )
    
    # Get target column (assuming it's the last column, adjust if needed)
    X_test = testing_data.iloc[:, :-1]
    y_test = testing_data.iloc[:, -1]
    
    # Initialize auditor
    try:
        from fairness_bias_auditor import FairnessAndBiasAuditing
        
        auditor = FairnessAndBiasAuditing(
            model=model,
            X=X_test,
            y_true=y_test,
            sensitive_features=X_test[sensitive_cols].values,
            sensitive_feature_names=sensitive_cols
        )
        
        # Run audit
        report = auditor.audit_fairness(fairness_threshold=request.fairness_threshold)
        
        # Convert report to dict for JSON serialization
        return {
            "fairness_metrics": report.overall_metrics,
            "subgroup_analysis": report.subgroup_analysis,
            "recommendations": report.recommendations
        }
        
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail="Fairness auditing requires fairlearn. Install with: pip install fairlearn"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
@app.post("/query/", summary="Process a user query on the training dataset")
def process_query(query: Query):
    if query.session_id not in sessions or 'agent' not in sessions[query.session_id]:
        raise HTTPException(status_code=404, detail="Session not found")

    agent = sessions[query.session_id]['agent']
    agent.context_manager.current_data = sessions[query.session_id]['training_data']
    try:
        response = agent.process_query(query.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
 