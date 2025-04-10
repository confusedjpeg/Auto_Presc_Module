from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID

class PrescribedMedicineBase(BaseModel):
    medicine_name: str
    dosage: str
    frequency: str
    instructions: str
    duration: int
    chemical_composition: str

class LabTestBase(BaseModel):
    test_name: str
    test_description: str
    book_test_url: Optional[str] = None

class PrescriptionBase(BaseModel):
    prescription_type: str
    diagnosis: str
    symptoms: List[str]
    doctor_advice: Optional[str] = None
    follow_up_date: Optional[datetime] = None
    medicines: List[PrescribedMedicineBase]
    lab_tests: List[LabTestBase]

class LabReportBase(BaseModel):
    pathology_name: str
    test_name: str

class HealthRecordBase(BaseModel):
    user_id: UUID
    date: datetime
    doctor_name: str
    doctor_registration_no: str
    doctor_specialization: str
    doctor_profile_picture_url: Optional[str] = None
    hospital_clinic_name: str
    hospital_clinic_address: str
    hospital_clinic_logo_url: Optional[str] = None
    status: str
    notes: Optional[str] = None
    shared: bool = False
    record_type: str
    patient_name: str
    patient_age: int
    patient_gender: str
    original_file_url: Optional[str] = None
    original_file_type: str
    original_file_name: str
    pdf_url: Optional[str] = None
    file_size: int
    prescription: Optional[PrescriptionBase] = None
    lab_report: Optional[LabReportBase] = None

class HealthRecordCreate(HealthRecordBase):
    pass

class HealthRecord(HealthRecordBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True 