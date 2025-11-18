# config.py
import os
from dotenv import load_dotenv
from mistralai.client import MistralClient

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = "open-mixtral-8x7b"

def get_mistral_client():
    if not MISTRAL_API_KEY:
        raise ValueError("API key not found")
    return MistralClient(api_key=MISTRAL_API_KEY)

def test_connection():
    try:
        client = get_mistral_client()
        response = client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say 'Hello'"}]
        )
        return "Connected"
    except Exception as e:
        return f" Failed: {e}"

print(test_connection())