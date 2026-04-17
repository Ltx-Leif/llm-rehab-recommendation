# api/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime # <--- 添加这行导入

# Import internal models to potentially reuse or map to
# Ensure models are imported correctly based on your project structure
# If models.py is at the root:
# from models import PreDiagnosisInfo, DiagnosisResult as InternalDiagnosisResult, InteractionNeeded as InternalInteractionNeeded, DiagnosisItem
# If models.py is elsewhere, adjust the import path accordingly.
# Assuming models.py is at the root for now:
try:
    from models import PreDiagnosisInfo, DiagnosisResult as InternalDiagnosisResult, InteractionNeeded as InternalInteractionNeeded, DiagnosisItem
except ImportError:
    # Fallback if running schemas.py directly or path issues persist
    # This indicates a potential project structure or path problem if it fails often
    print("Warning: Could not import from root 'models'. Ensure PYTHONPATH is correct or adjust import.")
    # Define minimal versions here if needed for standalone schema validation (not recommended for full app)
    class PreDiagnosisInfo(BaseModel): pass
    class InternalDiagnosisResult(BaseModel): pass
    class InternalInteractionNeeded(BaseModel): pass
    class DiagnosisItem(BaseModel): pass


# --- Phase 1 Schemas (Keep Existing) ---
class PatientInputData(BaseModel):
    patient_id: Optional[str] = None
    text_data: List[str] = Field(default_factory=list, description="List of text inputs like history, complaints.")
    image_references: List[str] = Field(default_factory=list, description="List of paths or IDs to medical images.")
    # Allow receiving updated interactive info for re-diagnosis attempts
    interactive_info: Optional[Dict[str, Any]] = Field(None, description="Key-value pairs from user interaction.")

class PreprocessingResponse(BaseModel):
    request_id: str
    status: str
    message: str
    pre_diagnosis_info: Optional[PreDiagnosisInfo] = None


# --- Phase 2 Schemas (NEW) ---

class DiagnosisInput(BaseModel):
    """Schema for initiating a diagnosis request."""
    # Option 1: Pass the full pre-diagnosis info (Current approach)
    pre_diagnosis_info: PreDiagnosisInfo
    # Option 2: Pass only the request ID (requires state management)
    # request_id: str
    # Option 3: Pass original data + interaction response
    # patient_data: PatientInputData # Original data
    # interaction_response: Optional[Dict[str, Any]] = None # User's answers
    # request_id: Optional[str] = None # To link back

class DiagnosisResultSchema(BaseModel):
    """API Schema for returning a diagnosis result."""
    request_id: str
    patient_id: Optional[str] = None
    diagnosis_list: List[DiagnosisItem] = Field(default_factory=list)
    primary_diagnosis: Optional[DiagnosisItem] = None
    kb_evidence_count: Optional[int] = Field(None, description="Number of KB snippets used for diagnosis (if tracked)")
    diagnosis_timestamp: datetime # <--- This line now works because datetime is imported

class InteractionNeededSchema(BaseModel):
    """API Schema for requesting user interaction."""
    request_id: str
    patient_id: Optional[str] = None
    questions_to_user: List[str]
    required_info: Optional[List[str]] = None
    suggested_options: Optional[Dict[str, List[str]]] = None
    feedback_context: Optional[str] = None

class DiagnosisResponse(BaseModel):
    """
    Unified API response for the diagnosis endpoint.
    Indicates whether diagnosis is complete or interaction is needed.
    """
    status: str = Field(..., description="Status of the diagnosis ('Completed', 'Needs Interaction', 'Error')")
    message: str = Field(..., description="A summary message for the response.")
    # Use Optional fields for the two possible outcomes
    diagnosis_result: Optional[DiagnosisResultSchema] = None
    interaction_needed: Optional[InteractionNeededSchema] = None
    # Include request_id here too for easier tracking on the client-side?
    request_id: Optional[str] = None

    # Example usage comments (can be kept or removed)
    # if isinstance(service_output, InternalDiagnosisResult):
    #     return DiagnosisResponse(status="Completed", message="Diagnosis complete.",
    #                              diagnosis_result=DiagnosisResultSchema(**service_output.dict()),
    #                              request_id=service_output.request_id)
    # elif isinstance(service_output, InternalInteractionNeeded):
    #     return DiagnosisResponse(status="Needs Interaction", message="Further information required.",
    #                              interaction_needed=InteractionNeededSchema(**service_output.dict()),
    #                              request_id=service_output.request_id)