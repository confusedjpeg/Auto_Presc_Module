# Auto-Prescription RAG API

This is a FastAPI-based service that uses RAG (Retrieval-Augmented Generation) to analyze prescriptions and generate recommendations based on clinical guidelines.

## Features

- Prescription analysis using RAG
- Integration with AWS S3 for file storage
- ICD code mapping for diagnoses
- Knowledge graph-based historical analysis
- FastAPI endpoints for easy integration

## Prerequisites

- Python 3.8+
- AWS account with S3 access
- NVIDIA API key
- ICD API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd auto-prescription-rag
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with the following variables:
```env
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
S3_BUCKET_NAME=your_bucket_name
NVIDIA_API_KEY=your_nvidia_api_key
ICD_API_KEY=your_icd_api_key
```

## Project Structure

```
app/
├── api/            # API endpoints
├── core/           # Core configuration
├── models/         # Pydantic models
├── services/       # Business logic
└── utils/          # Utility functions
```

## Running the Application

1. Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

2. The API will be available at `http://localhost:8000`

## API Endpoints

### POST /api/v1/analyze-prescription
Analyzes a prescription and generates recommendations.

Request:
- `file`: Prescription file (PDF, PNG, JPG)
- `user_id`: UUID of the user
- `patient_name`: Name of the patient
- `patient_age`: Age of the patient
- `patient_gender`: Gender of the patient

Response:
- Health record with prescription recommendations

### GET /api/v1/health
Health check endpoint.

## Development

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
pytest
```

## License

[Your License] 