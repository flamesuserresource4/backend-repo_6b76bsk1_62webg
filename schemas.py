"""
Database Schemas for Deepneumoscan

Each Pydantic model maps to a MongoDB collection (lowercased class name).
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime


class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: EmailStr = Field(..., description="Email address")
    password_hash: str = Field(..., description="SHA256 hash of password")
    language: str = Field("en", description="Preferred language code: en|kn")
    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[str] = Field(None)


class SelfAssessment(BaseModel):
    user_id: str
    answers: Dict[str, Any]
    result_label: str
    result_confidence: float
    language: str = Field("en")


class XrayScan(BaseModel):
    user_id: str
    name: str
    age: int
    gender: str
    medical_condition: Optional[str] = None
    file_path: str
    annotated_path: Optional[str] = None
    model_used: str
    predicted_label: str
    confidence: float
    language: str = Field("en")


class CuringAssessment(BaseModel):
    user_id: str
    inputs: Dict[str, Any]
    status: str  # better | worse | stable
    score_delta: float
    language: str = Field("en")


class HistoryItem(BaseModel):
    user_id: str
    item_type: str  # self_assessment | xray | curing
    ref_id: str
    summary: str
    language: str = Field("en")


class DeleteRequest(BaseModel):
    id: str
