# api/routes.py

from fastapi import APIRouter, HTTPException, status, Body, Depends
from typing import Union, Optional
from uuid import uuid4
from datetime import datetime

from api.schemas import (
    PatientInputData,
    PreprocessingResponse,
    DiagnosisInput,
    DiagnosisResponse,
    DiagnosisResultSchema,
    InteractionNeededSchema,
)
from services.preprocessing_service import PreprocessingService, get_preprocessing_service
from services.diagnosis_service import DiagnosisService, get_diagnosis_service
from models import (
    PreDiagnosisInfo,
    DiagnosisResult as InternalDiagnosisResult,
    InteractionNeeded as InternalInteractionNeeded,
)
from utils import get_logger, ExternalToolError

logger = get_logger(__name__)

preprocess_router = APIRouter(prefix="/api/v1", tags=["Phase 1: Preprocessing"])
diagnosis_router = APIRouter(prefix="/api/v1", tags=["Phase 2: Diagnosis"])


@preprocess_router.post(
    "/preprocess",
    response_model=PreprocessingResponse,
    summary="Preprocess Multimodal Patient Data",
    description="Receives patient text, image references, and interactive info, processes them, and returns structured pre-diagnosis information.",
    status_code=status.HTTP_200_OK,
)
async def run_preprocessing(
    patient_data: PatientInputData = Body(...),
    preprocessing_svc: PreprocessingService = Depends(get_preprocessing_service),
):
    patient_id_log = patient_data.patient_id or "UNKNOWN_PATIENT_INPUT"
    generated_request_id = str(uuid4())

    logger.info(
        f"Received preprocessing request for patient_id: {patient_id_log} (assigned internal request_id: {generated_request_id})"
    )

    try:
        pre_diagnosis_info_obj = await preprocessing_svc.preprocess_data(patient_data)
        # Use the service-generated request_id if available
        generated_request_id = pre_diagnosis_info_obj.request_id
        if pre_diagnosis_info_obj.errors:
            response_message = f"Preprocessing completed with {len(pre_diagnosis_info_obj.errors)} error(s)."
            logger.warning(
                f"Request {generated_request_id} (patient: {patient_id_log}) processed with errors: {pre_diagnosis_info_obj.errors}"
            )
        else:
            response_message = "Preprocessing completed successfully."
    except Exception as e:
        logger.error(
            f"Unhandled exception during preprocessing for patient {patient_id_log} (request_id: {generated_request_id}): {e}",
            exc_info=True,
        )
        response_message = "Preprocessing failed due to an internal server error."
        pre_diagnosis_info_obj = PreDiagnosisInfo(
            request_id=generated_request_id,
            patient_id=patient_id_log,
            errors=[f"{type(e).__name__}: {str(e)}"],
            processing_timestamp=datetime.now(),
        )

    # 构造并返回符合 schema 的响应
    return PreprocessingResponse(
        request_id=pre_diagnosis_info_obj.request_id,
        status="Completed" if not pre_diagnosis_info_obj.errors else "CompletedWithErrors",
        message=response_message,
        pre_diagnosis_info=pre_diagnosis_info_obj,
    )


@diagnosis_router.post(
    "/diagnose",
    response_model=DiagnosisResponse,
    summary="Generate Disease Diagnosis",
    description="Takes pre-processed patient information and returns a diagnosis or requests further interaction.",
    status_code=status.HTTP_200_OK,
)
async def run_diagnosis(
    diagnosis_input: DiagnosisInput = Body(...),
    diagnosis_svc: DiagnosisService = Depends(get_diagnosis_service),
):
    # 从输入获取 request_id 和 patient_id
    req_id = diagnosis_input.pre_diagnosis_info.request_id
    patient_id_log = diagnosis_input.pre_diagnosis_info.patient_id or "UNKNOWN_PATIENT"

    logger.info(f"Received diagnosis request for request_id: {req_id}, patient_id: {patient_id_log}")

    try:
        # —— 关键修改：调用正确的 get_diagnosis 方法 ——  
        service_output: Union[InternalDiagnosisResult, InternalInteractionNeeded] = await diagnosis_svc.get_diagnosis(
            pre_diagnosis_info=diagnosis_input.pre_diagnosis_info
        )

        # 如果返回的是 DiagnosisResult
        if isinstance(service_output, InternalDiagnosisResult):
            api_result = DiagnosisResultSchema(
                request_id=service_output.request_id,
                patient_id=service_output.patient_id,
                diagnosis_list=service_output.diagnosis_list,
                primary_diagnosis=service_output.primary_diagnosis,
                kb_evidence_count=len(service_output.kb_evidence) if service_output.kb_evidence else 0,
                diagnosis_timestamp=service_output.diagnosis_timestamp,
            )
            return DiagnosisResponse(
                status="Completed",
                message="Diagnosis process completed.",
                diagnosis_result=api_result,
                request_id=service_output.request_id,
            )

        # 如果返回的是 InteractionNeeded
        if isinstance(service_output, InternalInteractionNeeded):
            api_interaction = InteractionNeededSchema(
                request_id=service_output.request_id,
                patient_id=service_output.patient_id,
                questions_to_user=service_output.questions_to_user,
                required_info=service_output.required_info,
                suggested_options=service_output.suggested_options,
                feedback_context=service_output.feedback_context,
            )
            return DiagnosisResponse(
                status="Needs Interaction",
                message="Diagnosis requires further clarification or information.",
                interaction_needed=api_interaction,
                request_id=service_output.request_id,
            )

        # 其余情况视为错误
        logger.error(f"Diagnosis service returned unexpected type for request {req_id}: {type(service_output)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Diagnosis service returned an unexpected result type.",
        )

    except ExternalToolError as ete:
        logger.error(f"External tool error during diagnosis for request {req_id}: {ete}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Diagnosis failed due to an external service error ({ete.tool_name}): {ete.message}",
        )
    except Exception as e:
        logger.exception(f"Unhandled exception during diagnosis request {req_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal server error occurred during diagnosis.",
        )
