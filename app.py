import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from agent_core import IntelligentAgent
from file_processor import FileProcessor
from data_preprocessor import DataPreprocessor

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
    description="An intelligent data analysis agent powered by Mistral AI.",
    version="1.0.0"
)

# Session management
sessions = {}

class Query(BaseModel):
    session_id: str
    query: str

@app.post("/upload/", summary="Upload and process a dataset")
def upload_dataset(file: UploadFile = File(...)):
    """
    Uploads a dataset, processes it, and initializes a new analysis session.
    """
    session_id = str(uuid.uuid4())
    file_path = os.path.join("Datasets", file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    try:
        agent = IntelligentAgent(MISTRAL_API_KEY)
        file_processor = FileProcessor()
        file_info = file_processor.process_file(file_path)
        agent.register_dataset(file_path, file_info['type'])

        if file_info['type'] == 'dataframe':
            df = file_info['data']
            data_preprocessor = DataPreprocessor(agent=agent)
            processed_df, preproc_message = data_preprocessor.preprocess_dataframe(df)
            agent.context_manager.current_data = processed_df
            message = f"Dataset processed successfully. Shape: {processed_df.shape}. {preproc_message}"
        else:
            message = f"File processed successfully. Type: {file_info['type']}"

        sessions[session_id] = agent
        return {"session_id": session_id, "message": message}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/query/", summary="Process a user query")
def process_query(query: Query):
    """
    Processes a user query for a given session.
    """
    if query.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    agent = sessions[query.session_id]
    try:
        response = agent.process_query(query.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
 