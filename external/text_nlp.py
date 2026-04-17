# external/text_nlp.py
# Use absolute imports assuming project structure allows it
from models import TextFacts
from external.llm_client import call_deepseek_llm
from utils import get_logger, ExternalToolError
import json
from typing import Dict, List, Any, Optional
import asyncio # Keep for main block test

logger = get_logger(__name__)

# Placeholder for future medical term standardization (e.g., using UMLS)
def standardize_term(term: str, category: str) -> str:
    """Placeholder function for medical term standardization."""
    # In a real implementation, query UMLS, SNOMED CT API, or local dictionary
    # logger.debug(f"Standardizing '{term}' (category: {category}) -> (Not implemented yet)")
    return term.lower().strip() # Simple lowercase/strip for now

async def extract_text_features(text_content: str, text_ref: str = "source_text") -> TextFacts:
    """
    Extracts structured features (entities, relationships, summary) from clinical text using LLM.
    Relies on the LLM's ability to follow complex instructions and consistently return valid JSON.

    Args:
        text_content: The combined clinical text to analyze.
        text_ref: An identifier for the text source.

    Returns:
        A TextFacts object containing the extracted information or errors.
    """
    logger.info(f"Extracting features from text_ref: {text_ref} (length: {len(text_content)})")

    # --- Prompt Engineering is CRUCIAL here ---
    # TODO: Iteratively refine this prompt. Add examples within the prompt if needed.
    # Explicitly stating "ONLY JSON", "valid JSON", "strict format" helps.
    prompt = f"""
    Analyze the following clinical text. Extract key medical entities, categorize them,
    and optionally identify simple relationships and provide a brief summary.

    Categories to extract (use these exact keys in the JSON):
    - symptoms: Patient-reported symptoms.
    - signs: Clinician-observed signs.
    - diagnoses: Diagnosed conditions or differential diagnoses mentioned.
    - medications: Medications mentioned (try to include dosage/frequency if available as part of the string).
    - procedures: Procedures or tests mentioned.
    - patient_info: Key patient demographic or attribute data found (e.g., age, gender - use with caution for privacy).

    Relationships should be simple subject-relation-object triples (e.g., medication treats symptom).

    Format the output ONLY as a single, valid JSON object. Do not include any text before or after the JSON object.
    The JSON object must have the following top-level keys:
    - "entities": (Required) A dictionary where keys are the categories above (e.g., "symptoms")
                   and values are lists of extracted strings for that category. If a category has no findings, use an empty list [].
    - "relationships": (Optional) A list of dictionaries, each with "subject", "relation", "object" string keys, or null if none are found/extracted.
    - "summary": (Optional) A brief clinical summary of the text, or null.

    Clinical Text:
    ---
    {text_content}
    ---

    Valid JSON Output Only:
    """

    facts = TextFacts(text_ref=text_ref)
    try:
        llm_response_str = await call_deepseek_llm(prompt)

        # --- Robust JSON Parsing ---
        # Attempts to extract a JSON object even if the LLM added surrounding text.
        parsed_data: Optional[Dict[str, Any]] = None
        json_parsing_error: Optional[str] = None
        try:
            json_start = llm_response_str.find('{')
            json_end = llm_response_str.rfind('}')
            if json_start != -1 and json_end != -1 and json_end >= json_start:
                json_str = llm_response_str[json_start:json_end+1]
                parsed_data = json.loads(json_str)
                logger.info(f"Successfully parsed JSON from LLM response for {text_ref}.")
            else:
                # Check if the entire response is maybe a JSON list (less likely based on prompt but possible)
                 try:
                    parsed_data = json.loads(llm_response_str)
                    if not isinstance(parsed_data, dict): # We expect a dict based on prompt
                        logger.warning(f"LLM response parsed as JSON but was not a dictionary for {text_ref}.")
                        json_parsing_error = "LLM response parsed as JSON but was not the expected dictionary format."
                        parsed_data = None # Discard if not a dict
                 except json.JSONDecodeError:
                    logger.warning(f"Could not find JSON object markers or parse the full response as JSON for {text_ref}.")
                    json_parsing_error = "LLM response did not contain a recognizable JSON object or was malformed."


        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to decode JSON from LLM response for {text_ref}: {json_err}")
            logger.debug(f"LLM raw response for {text_ref} was:\n{llm_response_str}")
            json_parsing_error = f"LLM response was not valid JSON: {json_err}"

        # Record parsing error if one occurred
        if json_parsing_error:
             facts.error = json_parsing_error

        # --- Populate TextFacts from Parsed Data ---
        if parsed_data and isinstance(parsed_data, dict):
            # Extract and standardize entities
            raw_entities = parsed_data.get("entities")
            if isinstance(raw_entities, dict):
                processed_entities: Dict[str, List[str]] = {}
                for category, terms in raw_entities.items():
                    # Ensure terms is a list and filter out non-string/empty items robustness
                    if isinstance(terms, list):
                        processed_entities[category] = [
                            standardize_term(str(term), category)
                            for term in terms if term and isinstance(term, (str, int, float)) # Allow simple types, convert to str
                        ]
                    else:
                         logger.warning(f"Category '{category}' in parsed entities for {text_ref} was not a list.")
                facts.entities = processed_entities
            else:
                 logger.warning(f"Parsed JSON 'entities' field for {text_ref} is missing or not a dictionary.")
                 if facts.error is None: facts.error = "Parsed JSON 'entities' field had incorrect format or was missing."

            # Extract relationships (optional field)
            relationships = parsed_data.get("relationships")
            if relationships is not None: # Allow null or empty list
                if isinstance(relationships, list):
                    # Optional: Add validation for relationship structure here
                    facts.relationships = relationships
                else:
                    logger.warning(f"Parsed JSON 'relationships' field for {text_ref} was not a list or null.")
                    # Don't overwrite a previous error unless none existed
                    if facts.error is None: facts.error = "Parsed JSON 'relationships' field had incorrect format."

            # Extract summary (optional field)
            summary = parsed_data.get("summary")
            if summary is not None: # Allow null or empty string
                if isinstance(summary, str):
                    facts.summary = summary.strip()
                else:
                    logger.warning(f"Parsed JSON 'summary' field for {text_ref} was not a string or null.")
                    if facts.error is None: facts.error = "Parsed JSON 'summary' field had incorrect format."

            # If we successfully extracted entities, potentially override earlier minor parsing error
            if facts.entities and json_parsing_error:
                 logger.info(f"Overriding minor JSON parsing error for {text_ref} as entities were extracted.")
                 facts.error = None


        elif facts.error is None: # If parsing didn't yield a dict but no prior error recorded
             facts.error = "LLM response could not be parsed into the expected dictionary structure."

    except ExternalToolError as e:
        logger.error(f"LLM failed during text feature extraction for {text_ref}: {e}")
        # Overwrite previous errors if LLM call itself failed
        facts.error = f"LLM Error: {e.message}"
        # Clear partially extracted data if LLM call failed
        facts.entities = {}
        facts.relationships = None
        facts.summary = None
    except Exception as e:
        logger.error(f"Unexpected error during text feature extraction for {text_ref}: {e}", exc_info=True)
        facts.error = f"Unexpected processing error: {str(e)}"
        facts.entities = {}
        facts.relationships = None
        facts.summary = None

    return facts


# Example usage (optional, for testing)
async def main():
    test_text = """
    Patient: John Doe, 45yo Male.
    CC: Persistent cough and fever for 3 days.
    HPI: Reports fever up to 102F, non-productive cough, fatigue. Denies chest pain.
    PMH: Hypertension, managed with Lisinopril 10mg daily.
    Meds: Lisinopril 10mg qd. Occasional Ibuprofen for headache.
    Exam: Temp 101.8F, RR 20, O2 Sat 97% RA. Lungs clear.
    Assessment: Suspect viral URI vs early pneumonia. Plan: Chest X-ray, recommend supportive care. Prescribed Tamiflu 75mg BID for 5 days due to flu season.
    """
    facts = await extract_text_features(test_text, text_ref="test_case_1")
    print("\n--- Text Features Test Report ---")
    print(f"Text Ref: {facts.text_ref}")
    print(f"Entities: {json.dumps(facts.entities, indent=2)}")
    print(f"Relationships: {facts.relationships}")
    print(f"Summary: {facts.summary}")
    print(f"Error: {facts.error}")
    print("---------------------------------\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())