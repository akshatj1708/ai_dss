import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from agent_core import IntelligentAgent
from file_processor import FileProcessor
from data_preprocessor import DataPreprocessor
from data_quality_analyzer import DataQualityAnalyzer
import pandas as pd

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
    title="InsightX - AI Data Analysis Agent",
    description="An intelligent data analysis agent with data quality analysis features.",
    version="2.0.0"
)

# Session management
sessions = {}

class Query(BaseModel):
    session_id: str
    query: str

def _process_and_store_dataset(session_id: str, file: UploadFile, dataset_type: str):
    file_path = os.path.join("Datasets", file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    file_processor = FileProcessor()
    file_info = file_processor.process_file(file_path)

    if file_info['type'] != 'dataframe':
        raise HTTPException(status_code=400, detail="Only dataframe datasets are supported for quality analysis.")

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
        raise HTTPException(status_code=404, detail="Session not found. Please upload a training dataset first.")
    try:
        _process_and_store_dataset(session_id, file, 'testing_data')
        return {"session_id": session_id, "message": "Testing dataset uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/analyze/quality/{session_id}", summary="Perform data quality analysis")
def analyze_data_quality(session_id: str):
    if session_id not in sessions or 'training_data' not in sessions[session_id]:
        raise HTTPException(status_code=404, detail="Training data not found for this session.")

    training_data = sessions[session_id]['training_data']
    testing_data = sessions[session_id].get('testing_data')

    analyzer = DataQualityAnalyzer(df_train=training_data, df_test=testing_data)
    report = analyzer.analyze()

    return report

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
 