from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.models.schema import HealthRecordCreate, HealthRecord
from app.services.s3_service import s3_service
from app.services.icd_service import icd_service
import uuid
from datetime import datetime
import json
from typing import Optional
import logging

# Import the RAG pipeline components
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG service
rag_service = RAGService()

@app.post("/api/v1/analyze-prescription", response_model=HealthRecord)
async def analyze_prescription(
    file: UploadFile = File(...),
    user_id: str = None,
    patient_name: str = None,
    patient_age: int = None,
    patient_gender: str = None
):
    """
    Analyze a prescription file and generate recommendations
    """
    try:
        # Upload file to S3
        file_url = await s3_service.upload_file(
            file.file,
            file.filename,
            file.content_type
        )
        
        if not file_url:
            raise HTTPException(status_code=500, detail="Failed to upload file to S3")

        # Process the prescription using RAG pipeline
        prescription_data = await rag_service.process_prescription(file_url)
        
        if not prescription_data:
            raise HTTPException(status_code=500, detail="Failed to process prescription")

        # Get ICD code for diagnosis
        icd_code = await icd_service.get_icd_code(prescription_data["diagnosis"])
        
        # Create health record
        health_record = HealthRecordCreate(
            id=uuid.uuid4(),
            user_id=uuid.UUID(user_id),
            date=datetime.now(),
            doctor_name="AI Doctor",  # This will be replaced with actual doctor info
            doctor_registration_no="AI-DOC-001",
            doctor_specialization="General Medicine",
            hospital_clinic_name="AI Medical Center",
            hospital_clinic_address="Virtual Location",
            status="UHP",
            record_type="PRESCRIPTION",
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            original_file_url=file_url,
            original_file_type=file.content_type,
            original_file_name=file.filename,
            file_size=0,  # This should be calculated
            prescription=prescription_data
        )

        # Store the health record in your database
        # This part will depend on your database setup
        # For now, we'll just return the created record
        
        return health_record

    except Exception as e:
        logger.error(f"Error processing prescription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"} 