from pydantic import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Auto-Prescription RAG API"
    
    # AWS Settings
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "")
    
    # NVIDIA API Settings
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
    
    # ICD API Settings
    ICD_API_KEY: str = os.getenv("ICD_API_KEY", "")
    ICD_API_URL: str = "https://icd.who.int/icdapi"
    
    # Model Settings
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    FAISS_INDEX_PATH: str = "data/faiss_index.index"
    GUIDELINE_SECTIONS_PATH: str = "data/sections.txt"
    
    class Config:
        case_sensitive = True

settings = Settings() 