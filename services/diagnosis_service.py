# services/diagnosis_service.py
import json
import re
from typing import Union, List, Dict, Any, Optional
from models import PreDiagnosisInfo, DiagnosisResult, InteractionNeeded, DiagnosisItem, TextFacts, ImageReport
from external import llm_client, kb_client
from utils import get_logger, ExternalToolError
from datetime import datetime
import os

logger = get_logger(__name__)

# --- Helper function _format_pre_diagnosis_info_for_prompt (No changes from last version) ---
def _format_pre_diagnosis_info_for_prompt(info: PreDiagnosisInfo) -> str:
    """Formats pre-diagnosis info into a string suitable for LLM prompts."""
    prompt_parts = []
    prompt_parts.append(f"Patient ID: {info.patient_id or 'N/A'}")

    if info.processed_text_facts:
        prompt_parts.append("\n--- Extracted Text Facts ---")
        if info.processed_text_facts.summary:
            prompt_parts.append(f"Summary: {info.processed_text_facts.summary}")
        if info.processed_text_facts.entities:
            prompt_parts.append("Entities:")
            if isinstance(info.processed_text_facts.entities, dict):
                for category, items in info.processed_text_facts.entities.items():
                    if items and isinstance(items, list):
                        prompt_parts.append(f"  - {category.capitalize()}: {'; '.join(map(str, items))}")
            else:
                 logger.warning(f"Text facts entities for {info.request_id} is not a dictionary.")
        if info.processed_text_facts.error:
             prompt_parts.append(f"[Text Processing Error: {info.processed_text_facts.error}]")

    if info.processed_image_reports:
        prompt_parts.append("\n--- Image Analysis Reports ---")
        for report in info.processed_image_reports:
            prompt_parts.append(f"Image Ref: {os.path.basename(report.image_ref)}")
            if report.analysis_text:
                analysis_summary = report.analysis_text.split('\n')[0]
                prompt_parts.append(f"  Analysis Summary: {analysis_summary[:200]}...")
            if report.error:
                prompt_parts.append(f"  [Image Processing Error: {report.error}]")

    if info.raw_interactive_info:
        try:
            interactive_str = json.dumps(info.raw_interactive_info, indent=2, ensure_ascii=False)
            prompt_parts.append("\n--- Additional Interactive Information Provided ---")
            prompt_parts.append(interactive_str)
        except Exception:
             logger.warning(f"Could not JSON format interactive info for {info.request_id}, using raw string.")
             prompt_parts.append("\n--- Additional Interactive Information Provided ---")
             prompt_parts.append(str(info.raw_interactive_info))

    if info.errors:
        prompt_parts.append("\n--- Preprocessing Issues Encountered (Summary) ---")
        error_summary = "; ".join([err[:100] + '...' if len(err) > 100 else err for err in info.errors])
        prompt_parts.append(f"- {error_summary}")

    return "\n".join(prompt_parts)


# --- Helper function _parse_llm_diagnosis_response (Corrected Logging) ---
def _parse_llm_diagnosis_response(llm_output: str, request_id: str, patient_id: Optional[str]) -> Union[DiagnosisResult, InteractionNeeded]:
    logger.debug(f"Parsing LLM diagnosis response for request {request_id}...")
    parsed_successfully_as_json = False # Flag to track if initial JSON parse worked
    try:
        # Attempt 1: Strict JSON parsing from the cleaned response
        try:
            if llm_output.strip().startswith("```json"):
                json_str = llm_output.strip()[7:-3].strip()
            elif llm_output.strip().startswith("```"):
                 json_str = llm_output.strip()[3:-3].strip()
            else:
                 json_start = llm_output.find('{')
                 json_end = llm_output.rfind('}')
                 if json_start != -1 and json_end != -1 and json_end >= json_start:
                      json_str = llm_output[json_start:json_end+1]
                 else:
                      json_str = llm_output

            data = json.loads(json_str)
            # *** FIX: Correct the log message location ***
            # logger.info(f"Successfully parsed JSON from LLM diagnosis response for {request_id}.")
            parsed_successfully_as_json = True # Set flag

            # Check for interaction needed structure
            if "needs_interaction" in data and data["needs_interaction"] is True:
                questions = data.get("clarification_questions", [])
                required = data.get("required_info", [])
                options = data.get("suggested_options")
                feedback = data.get("feedback_context", "LLM determined more information is needed.")
                if not isinstance(questions, list): questions = [str(questions)]
                if not isinstance(required, list) and required is not None: required = [str(required)]

                if questions or required:
                    logger.info(f"LLM indicated interaction needed via JSON structure for request {request_id}.") # Corrected Log
                    return InteractionNeeded(
                        request_id=request_id, patient_id=patient_id,
                        questions_to_user=questions,
                        required_info=required,
                        suggested_options=options if isinstance(options, dict) else None,
                        feedback_context=str(feedback)
                    )
                else:
                     logger.warning(f"LLM indicated 'needs_interaction: true' in JSON but provided no questions/required_info for {request_id}. Treating as diagnosis attempt.")

            # Check for diagnosis structure
            if "diagnoses" in data and isinstance(data["diagnoses"], list):
                diagnosis_list = []
                primary_diag = None
                for item in data["diagnoses"]:
                    if isinstance(item, dict) and "disease_name" in item:
                         prob = item.get("probability")
                         if prob is not None:
                              try:
                                   prob_float = float(prob)
                                   if not (0.0 <= prob_float <= 1.0):
                                        logger.warning(f"Invalid probability value {prob} for {item.get('disease_name')} in request {request_id}. Setting to None.")
                                        prob = None
                                   else:
                                        prob = prob_float
                              except (ValueError, TypeError):
                                   logger.warning(f"Non-numeric probability value {prob} for {item.get('disease_name')} in request {request_id}. Setting to None.")
                                   prob = None

                         diag_item = DiagnosisItem(
                            disease_name=str(item.get("disease_name")),
                            icd_code=item.get("icd_code"),
                            probability=prob,
                            reasoning=item.get("reasoning")
                         )
                         diagnosis_list.append(diag_item)
                         if primary_diag is None:
                              primary_diag = diag_item

                if diagnosis_list:
                     logger.info(f"Extracted {len(diagnosis_list)} diagnoses via JSON for request {request_id}.") # Corrected Log
                     return DiagnosisResult(
                        request_id=request_id, patient_id=patient_id,
                        diagnosis_list=diagnosis_list,
                        primary_diagnosis=primary_diag,
                        diagnosis_timestamp=datetime.now()
                    )
                else:
                     logger.warning(f"LLM response for {request_id} was valid JSON but 'diagnoses' list was empty or invalid.")
                     # Fall through to text parsing

        except json.JSONDecodeError as e:
            logger.warning(f"LLM diagnosis response for {request_id} was not valid JSON: {e}. Falling back to text parsing.")
        except Exception as e:
             logger.warning(f"Error processing potential JSON response for {request_id}: {e}. Falling back to text parsing.")

        # Attempt 2: Text-based heuristics only if JSON parsing failed or yielded nothing useful
        if not parsed_successfully_as_json:
             logger.info(f"Attempting text-based parsing for LLM response for {request_id}.")
             interaction_keywords = ["need more information", "clarify", "insufficient data", "还需要", "请提供", "不明确", "缺乏信息", "进一步信息"]
             if any(keyword in llm_output.lower() for keyword in interaction_keywords):
                  questions = [line.strip(" -*?") for line in llm_output.split('\n') if line.strip().startswith('?') or any(q_word in line for q_word in ['什么', '是否', '明确', '哪', 'how', 'what', 'when', 'why', 'which', 'who'])]
                  if not questions: questions = ["AI请求进一步澄清信息，但无法提取具体问题。请人工审核。"]
                  logger.info(f"Detected interaction needed via keywords for request {request_id}.")
                  return InteractionNeeded(
                      request_id=request_id, patient_id=patient_id, questions_to_user=questions,
                      feedback_context="Interaction needed based on LLM response analysis (keywords detected)."
                  )

             potential_diagnoses = []
             lines = llm_output.split('\n')
             relevant_lines = [
                 line.strip(" -*:") for line in lines
                 if line.strip() and not line.lower().startswith((
                     "based on", "possible", "consider", "according to", "diagnosis:", "diagnoses:", "summary:", "assessment:",
                     "鉴于", "可能", "考虑", "分析", "总结", "诊断:", "初步诊断:", "鉴别诊断:", "印象:"
                 ))
             ]
             for line in relevant_lines:
                  if not line: continue
                  prob = None
                  match = re.search(r'[({\[]?\s*(?:probability|prob|置信度|可能性)\s*[:：=\s]*([\d.]+)\s*[%)}\]]?|[({\[]?\s*([\d.]+)\s*%\s*[)}\]]?', line, re.IGNORECASE)
                  prob_val_str = None
                  if match:
                      prob_val_str = match.group(1) or match.group(2)
                  if prob_val_str:
                      try:
                          prob_val = float(prob_val_str)
                          if '%' in match.group(0) or prob_val > 1.0:
                               prob = prob_val / 100.0
                          else:
                               prob = prob_val
                          line = re.sub(r'\s*[({\[]?\s*(?:probability|prob|置信度|可能性)\s*[:：=\s]*[\d.]+\s*[%)}\]]?|\s*[({\[]?\s*[\d.]+\s*%\s*[)}\]]?', '', line, flags=re.IGNORECASE).strip(" :-")
                      except ValueError:
                          logger.warning(f"Found potential probability but failed to parse: {prob_val_str} in line: {line}")
                          prob = None

                  if len(line.split()) < 15 and len(line) > 3:
                      potential_diagnoses.append(DiagnosisItem(disease_name=line.strip(), probability=prob))

             if potential_diagnoses:
                 logger.info(f"Extracted {len(potential_diagnoses)} potential diagnoses via text heuristics for {request_id}.")
                 try:
                      valid_prob_items = [d for d in potential_diagnoses if d.probability is not None]
                      none_prob_items = [d for d in potential_diagnoses if d.probability is None]
                      sorted_diagnoses = sorted(valid_prob_items, key=lambda x: x.probability, reverse=True) + none_prob_items
                 except TypeError:
                      sorted_diagnoses = potential_diagnoses
                 return DiagnosisResult(
                     request_id=request_id, patient_id=patient_id,
                     diagnosis_list=sorted_diagnoses,
                     primary_diagnosis=sorted_diagnoses[0],
                     diagnosis_timestamp=datetime.now()
                 )

        # Final fallback if nothing worked (even after text parsing attempt)
        logger.warning(f"Could not parse specific diagnoses or interaction request for {request_id} from raw output. Returning as 'Undetermined'. Raw output: {llm_output[:200]}...")
        fallback_diagnosis = DiagnosisItem(disease_name="待定诊断 (Undetermined)", reasoning=f"Raw LLM Output: {llm_output[:500]}...")
        return DiagnosisResult(
            request_id=request_id,
            patient_id=patient_id,
            diagnosis_list=[fallback_diagnosis],
            primary_diagnosis=fallback_diagnosis,
            diagnosis_timestamp=datetime.now()
            )

    except Exception as e:
        logger.exception(f"Unexpected error parsing LLM diagnosis response for {request_id}: {e}")
        return InteractionNeeded(
            request_id=request_id, patient_id=patient_id,
            questions_to_user=["处理诊断模型响应时发生系统错误。请人工审核。"],
            feedback_context=f"Parsing Error: {str(e)}"
        )

# --- DiagnosisService Class (No changes from last version) ---
class DiagnosisService:
    async def get_diagnosis(self, pre_diagnosis_info: PreDiagnosisInfo, top_k_kb: int = 3) -> Union[DiagnosisResult, InteractionNeeded]:
        """
        Performs disease diagnosis using preprocessed data, KB retrieval, and LLM inference.
        """
        request_id = pre_diagnosis_info.request_id
        patient_id = pre_diagnosis_info.patient_id
        logger.info(f"Starting diagnosis process for request_id: {request_id}, patient_id: {patient_id}")

        # 1. Format patient info
        patient_summary = _format_pre_diagnosis_info_for_prompt(pre_diagnosis_info)
        if not patient_summary.strip():
             logger.warning(f"No significant patient information found in PreDiagnosisInfo for request {request_id}.")
             return InteractionNeeded(
                 request_id=request_id,
                 patient_id=patient_id,
                 questions_to_user=["缺少可供分析的患者信息。请提供详细的病历文本或影像参考。"],
                 feedback_context="Initial input data was empty or insufficient."
             )

        # 2. Retrieve relevant knowledge from KB
        kb_context = ""
        kb_evidence_chunks = []
        try:
            query_for_kb_base = ""
            if pre_diagnosis_info.processed_text_facts:
                 if pre_diagnosis_info.processed_text_facts.summary:
                      query_for_kb_base = pre_diagnosis_info.processed_text_facts.summary
                 elif pre_diagnosis_info.processed_text_facts.entities:
                      entities_str = "; ".join(
                          f"{cat}: {', '.join(items)}" for cat, items in pre_diagnosis_info.processed_text_facts.entities.items() if items
                      )
                      query_for_kb_base = entities_str[:500]

            if not query_for_kb_base:
                 query_for_kb_base = patient_summary

            focus_keywords = "肿瘤 中风 康复 诊断 "
            query_for_kb_focused = focus_keywords + query_for_kb_base
            logger.debug(f"KB Query for {request_id}: {query_for_kb_focused[:200]}...")

            kb_evidence_chunks = kb_client.search_kb(query_for_kb_focused, top_k=top_k_kb)
            if kb_evidence_chunks:
                kb_context = "\n\n--- 相关知识库摘要 ---\n"
                for i, chunk in enumerate(kb_evidence_chunks):
                    chunk_str = str(chunk) if chunk else ""
                    kb_context += f"\n[参考 {i+1}]\n{chunk_str[:200]}...\n"
                logger.info(f"Retrieved {len(kb_evidence_chunks)} chunks from KB for request {request_id}.")
            else:
                 logger.info(f"No relevant documents found in KB for request {request_id} using query: {query_for_kb_focused[:100]}...")
        except ExternalToolError as e:
            logger.error(f"Failed to retrieve from KB for request {request_id}: {e}. Proceeding without KB context.")
            kb_context = "\n\n[知识库检索失败]"
        except Exception as e:
             logger.exception(f"Unexpected error during KB retrieval for request {request_id}: {e}. Proceeding without KB context.")
             kb_context = "\n\n[知识库检索发生意外错误]"


        # 3. Construct the LLM Prompt
        prompt = f"""你是一名专注于“肿瘤”和“中风”康复领域的AI医疗助手。你的任务是分析所提供的患者信息和相关的知识库摘要，以生成初步诊断或确定是否需要进一步澄清。

患者信息:
---
{patient_summary}
---
{kb_context}
---

指令:
1.  **分析:** 仅根据上面提供的“患者信息”和“相关知识库摘要”，识别与患者状况相关的潜在疾病或病症，**优先考虑与肿瘤或中风康复相关的诊断**。
2.  **诊断:** 列出最可能的诊断。对于每个诊断，请选择性地提供：
    * `"disease_name"`: (必需) 疾病名称。
    * `"icd_code"`: (可选) 相关的ICD代码字符串。
    * `"probability"`: (可选) 置信度（0.0到1.0之间的小数）。
    * `"reasoning"`: (可选) 基于所提供信息的简要理由说明，特别是与肿瘤/中风相关性。
3.  **评估充分性:** 严格评估当前信息是否足以做出可靠诊断。尤其注意区分是肿瘤/中风本身，还是其并发症/后遗症。
4.  **输出格式:** 严格要求只返回一个有效的 JSON 对象。不要包含 JSON 对象之前或之后的任何文本。
    * **如果信息充分:** JSON 应包含一个键 `"diagnoses"`，其值为一个对象列表，每个对象包含 `"disease_name"`, 可选的 `"icd_code"`, `"probability"`, `"reasoning"`。
        示例: `{{"diagnoses": [{{"disease_name": "脑卒中后遗症(吞咽障碍)", "icd_code": "I69.391", "probability": 0.85, "reasoning": "患者卒中病史与知识库中关于卒中后吞咽困难的描述相符。"}}, {{"disease_name": "肺部感染(吸入性)", "probability": 0.4, "reasoning": "吞咽障碍是吸入性肺炎的高风险因素。"}}]}}`
    * **如果信息不充分:** JSON 应包含:
        * `"needs_interaction"`: 设置为 `true`。
        * `"clarification_questions"`: (必需) 一个具体、清晰的问题列表，用于向用户（医生/患者）澄清模糊之处或收集关键缺失信息（例如，“患者肿瘤的具体分期是什么？”或“卒中发生的确切日期？”）。
        * `"required_info"`: (可选) 一个所需信息类型的列表（例如，“最新的影像学报告”，“详细的神经系统检查结果”）。
        * `"suggested_options"`: (可选) 一个字典，键是问题，值是可能的答案列表，以指导用户。
        * `"feedback_context"`: (可选) 解释为何需要更多信息的简要说明。
        示例: `{{"needs_interaction": true, "clarification_questions": ["患者中风是缺血性还是出血性？", "是否有进行过相关的肿瘤标志物检查？"], "required_info": ["卒中类型", "肿瘤标志物结果"]}}`

请务必只输出有效的 JSON 对象:
"""

        # 4. Call LLM
        try:
            llm_response_raw = await llm_client.call_deepseek_llm(prompt, json_mode=True)
            if not llm_response_raw:
                 logger.error(f"LLM returned an empty response for request {request_id}.")
                 return InteractionNeeded(request_id=request_id, patient_id=patient_id, questions_to_user=["诊断模型未返回响应。请人工审核。"])

            # 5. Parse LLM Response
            result = _parse_llm_diagnosis_response(llm_response_raw, request_id, patient_id)

            if isinstance(result, DiagnosisResult):
                 result.kb_evidence = kb_evidence_chunks
                 logger.info(f"Diagnosis process finished for request {request_id}. Included {len(kb_evidence_chunks)} KB snippets.")
            elif isinstance(result, InteractionNeeded):
                 logger.info(f"Diagnosis process determined interaction needed for request {request_id}.")

            return result

        except ExternalToolError as e:
            logger.error(f"LLM call failed during diagnosis for request {request_id}: {e}")
            return InteractionNeeded(
                request_id=request_id, patient_id=patient_id,
                questions_to_user=["由于AI模型出现问题，无法完成诊断。请稍后重试或人工审核。"],
                feedback_context=f"LLM Error: {e.message}"
            )
        except Exception as e:
            logger.exception(f"Unexpected error during diagnosis service execution for {request_id}: {e}")
            return InteractionNeeded(
                request_id=request_id, patient_id=patient_id,
                questions_to_user=["诊断过程中发生意外错误。请人工审核。"],
                feedback_context=f"System Error: {str(e)}"
            )

# Dependency provider function
def get_diagnosis_service() -> DiagnosisService:
     return DiagnosisService()