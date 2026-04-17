# models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union # Removed Tuple as it wasn't used
from datetime import datetime
import uuid

# --- Phase 1 Models ---
class ImageReport(BaseModel):
    """Represents the structured report from LLM-based image analysis."""
    image_ref: str # The original reference (e.g., URL or path used in prompt)

    # --- Analysis Results (Expected from LLM/Agent) ---
    # Keep findings simple for now, store full analysis text
    analysis_text: Optional[str] = Field(None, description="Full analysis text (e.g., markdown) from the LLM.")
    # You can parse analysis_text later if needed, or add more structured fields
    # findings: List[str] = Field(default_factory=list, description="Parsed findings.")
    # key_regions: Optional[Dict[str, Any]] = Field(None, ...)
    # confidence_scores: Optional[Dict[str, float]] = Field(None, ...)

    error: Optional[str] = None # To report errors during processing or LLM call

class TextFacts(BaseModel):
    # (Keep existing TextFacts structure)
    text_ref: str
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    relationships: Optional[List[Dict[str, Any]]] = None
    summary: Optional[str] = None
    error: Optional[str] = None

class PreDiagnosisInfo(BaseModel):
    # (Keep existing PreDiagnosisInfo structure)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: Optional[str] = None
    processed_text_facts: Optional[TextFacts] = None
    processed_image_reports: List[ImageReport] = Field(default_factory=list)
    raw_interactive_info: Optional[Dict[str, Any]] = None
    processing_timestamp: datetime = Field(default_factory=datetime.now)
    errors: List[str] = Field(default_factory=list)


# --- Phase 2 Models ---
class DiagnosisItem(BaseModel):
    # (Keep existing DiagnosisItem structure)
    disease_name: str
    icd_code: Optional[str] = None
    probability: Optional[float] = None
    reasoning: Optional[str] = None

class DiagnosisResult(BaseModel):
    # (Keep existing DiagnosisResult structure)
    request_id: str
    patient_id: Optional[str] = None
    diagnosis_list: List[DiagnosisItem] = Field(default_factory=list)
    primary_diagnosis: Optional[DiagnosisItem] = None
    kb_evidence: Optional[List[str]] = None
    diagnosis_timestamp: datetime = Field(default_factory=datetime.now)

class InteractionNeeded(BaseModel):
    # (Keep existing InteractionNeeded structure)
    request_id: str
    patient_id: Optional[str] = None
    questions_to_user: List[str]
    required_info: Optional[List[str]] = None
    suggested_options: Optional[Dict[str, List[str]]] = None
    feedback_context: Optional[str] = None