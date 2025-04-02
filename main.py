import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from openai import OpenAI
import json
from typing import Any, List
import re

# ----------------------------
# NVIDIA LLM Implementation using NVIDIA API
# ----------------------------
class NvidiaLLM(LLM):
    def __init__(self, api_key: str):
        super().__init__()
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "base_url", "https://integrate.api.nvidia.com/v1")
        object.__setattr__(self, "model", "meta/llama3-70b-instruct")
        object.__setattr__(self, "temperature", 0.6)
        object.__setattr__(self, "top_p", 0.95)
        object.__setattr__(self, "max_tokens", 4096)
        object.__setattr__(self, "client", OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers={"Content-Type": "application/json"}
        ))
    
    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise ValueError(f"API Error: {str(e)}")

    @property
    def _llm_type(self) -> str:
        return "nvidia"
# ----------------------------
# Helper Functions for Data Loading and Query Creation
# ----------------------------
def load_faiss_index(index_path: str) -> faiss.Index:
    return faiss.read_index(index_path)

def load_sections(sections_path: str) -> List[str]:
    with open(sections_path, 'r', encoding='utf-8') as f:
        sections = [sec.strip() for sec in f.read().split("\n\n") if sec.strip()]
    return sections

def formulate_query(age: int, gender: str, symptoms: List[str], chronic_conditions: List[str]) -> str:
    query = f"A {age}-year-old {gender} presents with {', '.join(symptoms)}."
    if chronic_conditions:
        query += f" The patient has a history of {', '.join(chronic_conditions)}."
    query += " What are the recommended prescription regimens for treating this condition considering drug interactions and clinical guidelines?"
    return query

# ----------------------------
# Main Processing Pipeline
# ----------------------------
if __name__ == "__main__":
    # Paths for FAISS index and guideline sections file
    index_path = 'data/faiss_index.index'
    sections_path = 'data/sections.txt'
    
    # Load FAISS index and guideline sections
    faiss_index = load_faiss_index(index_path)
    sections = load_sections(sections_path)
    print(f"Loaded FAISS index with {faiss_index.ntotal} vectors.")
    print(f"Loaded {len(sections)} guideline sections.")
    
    # Example patient details
    patient_age = 28
    patient_gender = "male"
    patient_symptoms = ["persistent cough", "fever", "mild chest pain"]
    patient_chronic = ["asthma"]
    
    # Formulate the patient query
    query = formulate_query(patient_age, patient_gender, patient_symptoms, patient_chronic)
    print("\nGenerated Query:")
    print(query)
    
    # Encode the query using SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    
    # Retrieve top 3 relevant guideline sections using FAISS
    top_k = 3
    distances, indices = faiss_index.search(np.array(query_embedding), top_k)
    retrieved_guidelines = [sections[i] for i in indices[0] if i < len(sections)]
    
    print("\nRetrieved Guidelines:")
    for idx, guideline in enumerate(retrieved_guidelines, start=1):
        print(f"\nGuideline {idx}:\n{guideline}")
    
    # ----------------------------
    # Define the prompt template for prescription recommendation
    # ----------------------------
    prompt_template = PromptTemplate(
        input_variables=["query", "guidelines"],
        template="""
Based on the following clinical guidelines and patient information, generate a set of prescription recommendations.

Patient Information: {query}

Clinical Guidelines Context:
{guidelines}

Provide your answer as a structured JSON object with the following keys:
- "recommended_medications": a list of objects, each containing "drug", "dosage", and "instructions".
- "rationale": an explanation for the recommendations.
"""
    )
    
    nvidia_llm = NvidiaLLM(api_key="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")
    
    # Create chain using the NVIDIA LLM and prompt template
    chain = LLMChain(llm=nvidia_llm, prompt=prompt_template)
    
    # Use the new invoke() method to generate the prescription recommendation
    response = chain.invoke({"query": query, "guidelines": "\n\n".join(retrieved_guidelines)})
    
    print("\nPrescription Recommendation:")
    print(response)
if isinstance(response, dict) and 'text' in response:
    text_response = response['text']
    # Use regex to extract the code block between triple backticks
    pattern = re.compile(r"```(.*?)```", re.DOTALL)
    match = pattern.search(text_response)
    if match:
        json_text = match.group(1).strip()
        try:
            parsed_data = json.loads(json_text)
            formatted_json = json.dumps(parsed_data, indent=2)
            print("\nParsed Prescription Recommendation:")
            print(formatted_json)
        except json.JSONDecodeError as e:
            print("Error parsing JSON:", e)
            print("\nRaw extracted JSON text:")
            print(json_text)
    else:
        print("No JSON code block found in the text.")
else:
    print("The response does not contain a 'text' key or is not a dictionary.")