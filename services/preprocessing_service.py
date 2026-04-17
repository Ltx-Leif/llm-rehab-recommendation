# services/preprocessing_service.py
from __future__ import annotations # MUST be the first line
import logging 
import asyncio
import json 
import uuid
import os
from datetime import datetime
from typing import List, Optional 

from api.schemas import PatientInputData 
from external import image_analyzer, text_nlp 
from models import PreDiagnosisInfo, ImageReport, TextFacts
from utils import get_logger, ExternalToolError

logger = get_logger(__name__)

class PreprocessingService:
    def __init__(self):
        # This default_image_prompt will now be passed to the image analyzer.
        self.default_image_prompt = (
            "请详细描述这张医学影像，重点分析图像中可能与脑卒中（中风）或肿瘤相关的任何发现。"
            "请指出主要的解剖结构，并详细说明任何可见的异常、病变或引人注意的区域，"
            "包括其位置、大小、形状、密度/信号特征以及可能的性质。"
            "如果未发现明确的脑卒中或肿瘤指征，请也明确说明。"
            "如果图像非医学影像或不清晰，请指出。"
        )
        logger.info(f"PreprocessingService initialized with a default image prompt strategy.")


    async def _analyze_single_image(self, image_path: str, request_id: str) -> ImageReport:
        """
        Helper to analyze one image using external.image_analyzer.
        Returns an ImageReport object; its 'error' field might be populated if analysis failed.
        """
        log_prefix = f"Request {request_id}, Image {os.path.basename(image_path)}: "
        try:
            logger.debug(f"{log_prefix}Calling analyze_image_with_huatuo...")
            # Pass the 'prompt' argument as it's now required by analyze_image_with_huatuo.
            # Assuming the parameter name in image_analyzer.py is 'prompt'.
            report = await image_analyzer.analyze_image_with_huatuo(
                image_path=image_path,
                prompt=self.default_image_prompt, # Pass the prompt
                request_id=request_id 
            )
            
            if report.error:
                 logger.warning(f"{log_prefix}Analysis by image_analyzer completed with an error: {report.error}")
            else:
                 logger.info(f"{log_prefix}Analysis by image_analyzer completed successfully.")
            return report
        except Exception as e:
            # This catches errors if the call to analyze_image_with_huatuo itself fails unexpectedly
            # This is where the current TypeError is caught.
            logger.error(
                f"{log_prefix}Unhandled exception directly calling analyze_image_with_huatuo.",
                exc_info=True 
            )
            return ImageReport( 
                image_ref=image_path, 
                analysis_text=None, 
                error=f"Service layer error calling image analyzer: {type(e).__name__} - {str(e)}"
            )

    async def preprocess_data(self, patient_data: PatientInputData) -> PreDiagnosisInfo:
        request_id = str(uuid.uuid4())
        patient_id_val = patient_data.patient_id or f"unknown_patient_{request_id}"
        logger.info(f"Starting preprocessing for request_id: {request_id}, patient_id: {patient_id_val}")

        processing_errors: List[str] = []
        processed_image_reports: List[ImageReport] = []
        processed_text_facts: Optional[TextFacts] = None

        # --- 1. Image Data Processing ---
        if patient_data.image_references:
            logger.info(f"Processing {len(patient_data.image_references)} image reference(s) for request {request_id}...")
            image_analysis_tasks = []
            for image_ref in patient_data.image_references:
                actual_image_path = image_ref 
                
                if not os.path.isabs(actual_image_path):
                    pass 

                if not os.path.exists(actual_image_path):
                    logger.warning(f"Request {request_id}: Image file reference '{actual_image_path}' does not exist. Skipping.")
                    report = ImageReport(image_ref=image_ref, error=f"File not found at path: {actual_image_path}")
                    processed_image_reports.append(report)
                    processing_errors.append(f"Image PreCheck ({os.path.basename(image_ref)}): File not found at {actual_image_path}")
                    continue

                image_analysis_tasks.append(
                    self._analyze_single_image(image_path=actual_image_path, request_id=request_id)
                )
            
            if image_analysis_tasks:
                results = await asyncio.gather(*image_analysis_tasks, return_exceptions=True)
                for res in results:
                    if isinstance(res, ImageReport):
                        processed_image_reports.append(res)
                        if res.error:
                            processing_errors.append(f"Image Analysis ({os.path.basename(res.image_ref)}): {res.error}")
                    elif isinstance(res, Exception): 
                        logger.error(f"Request {request_id}: asyncio.gather caught an unexpected exception from _analyze_single_image: {res}", exc_info=res)
                        processing_errors.append(f"Image Analysis Task: Unexpected error - {type(res).__name__}")
            logger.info(f"Finished processing image data for request {request_id}.")
        else:
            logger.info(f"No image references provided for request {request_id}.")


        # --- 2. Text Data Processing (Including Interactive Info) ---
        full_text_content = ""
        if patient_data.text_data:
            full_text_content += "\n".join(patient_data.text_data)
        
        interactive_info = patient_data.interactive_info 
        if interactive_info: 
            full_text_content += f"\n\n--- Interactive Supplementary Information ---\n{interactive_info}"
        
        if full_text_content.strip():
            text_ref_id = f"{request_id}_text"
            logger.info(f"Processing text data for request {request_id} (ref: {text_ref_id})...")
            try:
                processed_text_facts = await text_nlp.extract_text_features(
                    text_content=full_text_content.strip(),
                    text_ref=text_ref_id
                )
                if processed_text_facts and processed_text_facts.error:
                    logger.warning(f"Request {request_id}: Text Analysis ({text_ref_id}): LLM Error: {processed_text_facts.error}")
                    processing_errors.append(f"Text Analysis ({text_ref_id}): {processed_text_facts.error}")
            except ExternalToolError as ete: 
                error_msg = f"Text Analysis LLM Error ({text_ref_id}): {ete.message}"
                logger.error(f"Request {request_id}: {error_msg}")
                processing_errors.append(error_msg)
                processed_text_facts = TextFacts(text_ref=text_ref_id, error=ete.message) 
            except Exception as e: 
                error_msg = f"Text Analysis ({text_ref_id}): Unexpected error - {type(e).__name__}"
                logger.exception(f"Request {request_id}: Unhandled exception processing text")
                processing_errors.append(error_msg)
                processed_text_facts = TextFacts(text_ref=text_ref_id, error=f"Processing failed: {type(e).__name__}")

            logger.info(f"Finished processing text data for request {request_id}.")
        else:
            logger.info(f"No valid text data or interactive info provided for request {request_id}.")


        # --- 3. Create Final PreDiagnosisInfo ---
        processing_timestamp = datetime.now()
        pre_diagnosis_info = PreDiagnosisInfo(
            request_id=request_id,
            patient_id=patient_id_val,
            processed_text_facts=processed_text_facts,
            processed_image_reports=processed_image_reports,
            processing_timestamp=processing_timestamp,
            raw_interactive_info=patient_data.interactive_info, 
            errors=processing_errors 
        )

        log_status = "completed with errors" if processing_errors else "completed successfully"
        logger.info(f"Preprocessing request {request_id} {log_status}. Errors logged: {len(processing_errors)}")

        return pre_diagnosis_info

# Dependency provider function
def get_preprocessing_service() -> PreprocessingService:
    return PreprocessingService()