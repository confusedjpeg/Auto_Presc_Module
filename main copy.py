import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pydantic import Field, root_validator, validator
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from openai import OpenAI,APIError 
from typing import Any, List, Dict, Optional, Mapping
# import boto3 # Not used in the current logic, commented out
import json
from typing import Any, List, Dict, Optional
import re
import os
import networkx as nx
from node2vec import Node2Vec
import torch
import torch.nn as nn
import sys
# import torch.optim as optim # Optimizers not needed for inference-only MLP

# ----------------------------
# Environment Variable Loading
# ----------------------------
def load_environment_variables(env_path: Optional[str] = None):
    """Loads environment variables from .env file."""
    if env_path:
        load_dotenv(dotenv_path=env_path)
        print(f"Loaded environment variables from: {env_path}")
    else:
        load_dotenv() # Looks for .env in the current directory or parent directories
        print("Loaded environment variables from default .env location.")
    
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if not nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY not found in environment variables. Please ensure it is set in your .env file or system environment.")
    return nvidia_api_key

# ----------------------------
# NVIDIA LLM Implementation
# ----------------------------

class NvidiaLLM(LLM):
    """
    LangChain Custom LLM for NVIDIA API.
    Initializes client in __init__ after standard Pydantic/Langchain setup.
    """
    
    # --- Pydantic Fields for Configuration ---
    # Define configuration parameters expected by LangChain & Pydantic
    model: str = "meta/llama3-70b-instruct" # Provide defaults directly
    temperature: float = 0.5
    top_p: float = 0.9
    max_tokens: int = 2048
    base_url: str = "https://integrate.api.nvidia.com/v1"
    
    # API Key - Will be handled in __init__
    # Mark as Optional, it's not strictly required *by Pydantic* if handled later
    nvidia_api_key: Optional[str] = None 
    
    # Client - Also handled in __init__
    # Declare the type hint but initialize to None
    client: Optional[OpenAI] = None 

    # Use standard __init__ alongside Pydantic fields
    def __init__(self, **data: Any):
        """
        Initialize the LLM. Handles API key retrieval and client setup
        after Pydantic/Langchain base initialization.
        """
        print(f"DEBUG: Entering NvidiaLLM __init__. Initial data: { {k:v for k,v in data.items() if k != 'api_key' and k != 'nvidia_api_key'} }") # Avoid printing key
        
        # Let Pydantic/LangChain handle its own initialization first
        # It will populate fields like model, temperature, etc. from 'data' or defaults
        super().__init__(**data) 
        print(f"DEBUG: After super().__init__. self.model={self.model}, self.temperature={self.temperature}")

        # --- API Key Handling ---
        # Get key from input data (might be under 'api_key' or 'nvidia_api_key')
        # or from environment variable.
        _api_key_input = data.get('api_key', data.get('nvidia_api_key')) # Check common aliases
        _api_key_to_use = _api_key_input or os.getenv("NVIDIA_API_KEY")
        
        if not _api_key_to_use:
            print("ERROR: API key missing in __init__.", file=sys.stderr)
            raise ValueError("NVIDIA API key not found. Pass via 'api_key'/'nvidia_api_key' or set NVIDIA_API_KEY env var.")
        else:
             # Optionally store it on self if needed elsewhere, but maybe not necessary
             # self.nvidia_api_key = _api_key_to_use 
             print("DEBUG: API key resolved successfully.")

        # --- Client Initialization ---
        # Check if client was somehow passed in data (unlikely but possible)
        if self.client is None:
            print(f"DEBUG: Initializing OpenAI client in __init__ with base_url={self.base_url}")
            try:
                # Directly assign to self.client
                self.client = OpenAI(
                    base_url=self.base_url, # Use base_url set by Pydantic/super().__init__
                    api_key=_api_key_to_use,
                )
                print(f"DEBUG: Client initialized in __init__. Type: {type(self.client)}")
            except Exception as e:
                print(f"ERROR: Failed to initialize OpenAI client in __init__: {e}", file=sys.stderr)
                # Ensure client is None if init fails
                self.client = None
                raise ValueError(f"Failed to initialize OpenAI client: {e}") from e
        else:
             print("DEBUG: Client already exists (passed via data?), skipping init.")

        # --- Final Check ---
        if self.client is None:
             print("ERROR: Client is None at the end of __init__!", file=sys.stderr)
             raise RuntimeError("Client initialization failed within __init__.")


    # --- LangChain Required Methods ---
    # ... ( _llm_type, _identifying_params, _call remain the same as the previous attempt ) ...
    @property
    def _llm_type(self) -> str:
        return "nvidia_pydantic_init"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters for LangChain serialization/caching."""
        # Return the Pydantic fields defined above (excluding client and API key)
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url,
        }

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """Makes the API call to NVIDIA."""
        print(f"DEBUG: Entering _call. Client object: {self.client} (Type: {type(self.client)})")
        
        # Client *should* be initialized by __init__ if it completed without error
        if not self.client:
            print("ERROR: _call method found self.client is None. __init__ likely failed.", file=sys.stderr)
            raise ValueError("NVIDIA LLM client is not initialized.")
            
        messages = [{"role": "user", "content": prompt}]
        
        api_params = {
             "temperature": self.temperature,
             "top_p": self.top_p,
             "max_tokens": self.max_tokens,
             **kwargs # Runtime kwargs override class defaults
        }

        try:
            print(f"DEBUG: Calling chat.completions.create with model={self.model}, params={api_params}")
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                stop=stop if stop else None,
                **api_params
            )
            print("DEBUG: chat.completions.create call successful.")

            content = completion.choices[0].message.content if (
                completion.choices and completion.choices[0].message
            ) else None

            if content is None:
                 print("Warning: LLM response content is None.")
                 return ""
            return content

        except APIError as e:
            print(f"ERROR: OpenAI API error during _call: Status={e.status_code} Response={e.response}", file=sys.stderr)
            raise ValueError(f"NVIDIA API Error (APIError): {str(e)}") from e
        except Exception as e:
            print(f"ERROR: Generic error during _call: {type(e).__name__}: {e}", file=sys.stderr)
            raise ValueError(f"NVIDIA API Error (Generic): {str(e)}") from e
            
    # No Pydantic Config needed here as we rely on __init__
# ----------------------------
# Simple MLP Decoder (PyTorch)
# ----------------------------
class MLPDecoder(nn.Module):
    """Simple MLP for processing fused features."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2) # Added another layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim) # Adjusted input dim

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def run_mlp_decoder(fused_features: np.ndarray) -> str:
    """
    Runs inference using a pre-defined (or randomly initialized) MLP Decoder.
    Returns a qualitative assessment based on the output score.
    """
    if not isinstance(fused_features, np.ndarray):
        print("Warning: Invalid input type for MLP decoder. Expected numpy array.")
        return "MLP analysis skipped due to invalid input type."
        
    # Define expected dimensions based on SentenceTransformer + Node2Vec
    st_dim = 384 # all-MiniLM-L6-v2
    n2v_dim = 64
    expected_input_dim = st_dim + n2v_dim # 448

    if fused_features.shape != (expected_input_dim,):
        print(f"Warning: Fused features have unexpected shape {fused_features.shape}. Expected ({expected_input_dim},). MLP input might be incorrect.")
        # Decide how to handle this: return error, use zeros, or try reshaping (latter is risky)
        # For now, return an informative message
        return f"MLP analysis skipped due to feature shape mismatch (Expected {expected_input_dim}, Got {fused_features.shape[0]})."

    # Ensure input is float32 tensor
    x_tensor = torch.tensor(fused_features, dtype=torch.float32).unsqueeze(0) # Add batch dimension

    # Define model architecture (must match potential training)
    hidden_dim = 128
    output_dim = 1 # Assuming a single score output
    model = MLPDecoder(expected_input_dim, hidden_dim, output_dim)

    # Set model to evaluation mode (important for layers like dropout, batchnorm if used)
    model.eval()

    # Perform inference
    try:
        with torch.no_grad(): # Disable gradient calculations for inference
            output = model(x_tensor)
        
        score = output.item() # Get the scalar value from the output tensor

        # Interpret the score (threshold is arbitrary without training)
        # Let's make the interpretation slightly more nuanced
        if score > 0.5:
            return "Historical treatments appear potentially effective; similar regimens may be suitable."
        elif score < -0.5:
            return "Historical treatments appear potentially less effective; alternative approaches might be warranted."
        else:
            return "Historical treatment effectiveness is inconclusive based on current analysis; consider guidelines and current presentation strongly."
            
    except Exception as e:
        print(f"Error during MLP inference: {e}")
        return "MLP analysis encountered an error."

# ----------------------------
# Helper Functions
# ----------------------------
def load_faiss_index(index_path: str) -> Optional[faiss.Index]:
    """Loads a FAISS index from the specified path."""
    try:
        index = faiss.read_index(index_path)
        print(f"Successfully loaded FAISS index from {index_path} with {index.ntotal} vectors.")
        return index
    except Exception as e:
        print(f"Error loading FAISS index from {index_path}: {e}")
        # Depending on requirements, either raise the error or return None
        # raise e
        return None

def load_guideline_sections(sections_path: str) -> List[str]:
    """Loads guideline text sections from a file."""
    try:
        with open(sections_path, 'r', encoding='utf-8') as f:
            # Split by double newline, strip whitespace, filter empty sections
            sections = [sec.strip() for sec in f.read().split("\n\n") if sec.strip()]
        print(f"Loaded {len(sections)} guideline sections from {sections_path}.")
        return sections
    except FileNotFoundError:
        print(f"Error: Guideline sections file not found at {sections_path}.")
        return []
    except Exception as e:
        print(f"Error reading guideline sections from {sections_path}: {e}")
        return []

def load_json_file(file_path: str) -> Optional[Dict]:
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded JSON data from {file_path}.")
        return data
    except FileNotFoundError:
        print(f"Warning: JSON file not found at {file_path}.")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {file_path}. File might be empty or malformed.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return None

def extract_relevant_treatment_history(health_record: Optional[Dict], relevant_diagnoses: List[str]) -> List[str]:
    """
    Extracts treatment history (medication names and dosages) for specified diagnoses.
    Converts diagnoses to lowercase for comparison.
    """
    history = []
    if not health_record or 'prescriptions' not in health_record:
        return history

    relevant_diagnoses_lower = [diag.lower() for diag in relevant_diagnoses]
    
    prescriptions = health_record.get("prescriptions")
    if not isinstance(prescriptions, list):
        print("Warning: 'prescriptions' field in health record is not a list.")
        return history # Expecting a list of prescriptions

    for pres in prescriptions:
        if not isinstance(pres, dict):
            print("Warning: Encountered non-dictionary item in prescriptions list.")
            continue # Skip malformed prescription entries

        diagnosis = pres.get("diagnosis", "")
        if not isinstance(diagnosis, str): diagnosis = "" # Ensure diagnosis is a string
        
        # Check if this prescription's diagnosis is relevant
        if diagnosis.lower() in relevant_diagnoses_lower:
            medicines = pres.get("medicines", [])
            if not isinstance(medicines, list):
                 print(f"Warning: 'medicines' field is not a list in prescription ID {pres.get('prescription_id', 'N/A')}.")
                 continue # Skip if medicines is not a list

            for med in medicines:
                if not isinstance(med, dict):
                    print(f"Warning: Encountered non-dictionary item in medicines list for prescription ID {pres.get('prescription_id', 'N/A')}.")
                    continue # Skip malformed medicine entries
                    
                name = med.get("medicineName", "Unknown Medicine")
                dosage = med.get("dosage", "") # Dosage might be empty, that's okay
                
                # Ensure name and dosage are strings before formatting
                name_str = str(name)
                dosage_str = str(dosage)
                
                history_entry = f"{name_str} ({dosage_str})" if dosage_str else name_str
                history.append(history_entry.strip())
                
    return list(set(history)) # Return unique history entries

def extract_lab_reports(health_record: Optional[Dict]) -> List[str]:
    """Extracts formatted lab report results."""
    lab_reports = []
    if not health_record or 'labReport' not in health_record:
        return lab_reports

    lab_data = health_record.get("labReport")
    
    # Normalize to a list, handle None or non-list types gracefully
    reports = []
    if isinstance(lab_data, list):
        reports = lab_data
    elif isinstance(lab_data, dict):
        reports = [lab_data]
    else:
        print("Warning: 'labReport' field in health record is not a list or dictionary.")
        return lab_reports # Expecting list or dict

    for report in reports:
        if not isinstance(report, dict):
            print("Warning: Encountered non-dictionary item in lab reports.")
            continue # Skip malformed entries

        test_name = report.get("testName", "Unknown Test")
        pathology_name = report.get("pathologyName", "") # Optional
        result = report.get("testResult", "") # Result might be empty

        # Ensure components are strings
        test_name_str = str(test_name)
        pathology_name_str = str(pathology_name)
        result_str = str(result)

        # Format the string
        report_str = f"{test_name_str}"
        if pathology_name_str:
            report_str += f" ({pathology_name_str})"
        report_str += f": {result_str}"
        
        lab_reports.append(report_str.strip())
        
    return lab_reports

def formulate_query_for_llm(age: int, gender: str, symptoms: List[str], 
                        chronic_conditions: List[str], treatment_history: List[str], 
                        lab_reports: List[str], historical_analysis: str) -> str:
    """Formats the patient information and analysis into a comprehensive query for the LLM."""
    
    query_parts = []
    query_parts.append(f"Patient Profile: {age}-year-old {gender}.")
    
    if symptoms:
        query_parts.append(f"Presenting Symptoms: {', '.join(symptoms)}.")
    else:
        query_parts.append("Presenting Symptoms: None reported.")

    if chronic_conditions:
        query_parts.append(f"Chronic Conditions/History: {', '.join(chronic_conditions)}.")
    else:
        query_parts.append("Chronic Conditions/History: None reported.")

    if treatment_history:
        # Limit length if history is very long?
        query_parts.append(f"Relevant Past Treatments for Chronic Conditions: {'; '.join(treatment_history)}.")
    # else:
        # query_parts.append("Relevant Past Treatments: None extracted for chronic conditions.") # Maybe omit if empty

    if lab_reports:
        query_parts.append(f"Relevant Lab Findings: {'; '.join(lab_reports)}.")
    # else:
        # query_parts.append("Relevant Lab Findings: None extracted.") # Maybe omit if empty
        
    if historical_analysis:
         query_parts.append(f"Automated Historical Trend Analysis: {historical_analysis}")

    # The core request to the LLM
    # query_parts.append("\nRequest: Based *only* on the Clinical Guidelines Context provided previously and the Patient Information above, generate a prescription recommendation.")
    # query_parts.append("Consider potential drug interactions implicitly suggested by guidelines or standard practice, guideline adherence, previous treatment outcomes suggested by the history, and lab findings.")
    # query_parts.append("Provide your answer *strictly* as a single, valid JSON object enclosed in ```json ... ``` tags, following the specified output format.") # Instruction moved to prompt template

    return "\n".join(query_parts)

def build_knowledge_graph(health_record: Optional[Dict]) -> nx.Graph:
    """Builds a NetworkX graph from prescription data in the health record."""
    G = nx.Graph()
    if not health_record or 'prescriptions' not in health_record:
        print("Info: Cannot build knowledge graph, 'prescriptions' missing or health record is empty.")
        return G

    prescriptions = health_record.get("prescriptions")
    if not isinstance(prescriptions, list):
        print("Warning: 'prescriptions' field is not a list. Cannot build graph.")
        return G

    for pres in prescriptions:
        if not isinstance(pres, dict): continue # Skip malformed

        diagnosis = pres.get("diagnosis")
        symptoms_in_pres = pres.get("symptoms", [])
        medicines_in_pres = pres.get("medicines", [])

        # Ensure types are correct and clean data
        diag_node = str(diagnosis).lower().strip() if isinstance(diagnosis, str) and diagnosis.strip() else None
        
        symptom_nodes = [str(s).lower().strip() for s in symptoms_in_pres if isinstance(s, str) and s.strip()]
        
        med_nodes = []
        if isinstance(medicines_in_pres, list):
            for med in medicines_in_pres:
                 if isinstance(med, dict):
                     med_name = med.get("medicineName")
                     if isinstance(med_name, str) and med_name.strip():
                         med_nodes.append(med_name.lower().strip())

        # Add nodes to graph
        if diag_node:
            G.add_node(diag_node, type="diagnosis")
        
        for sym_node in symptom_nodes:
            G.add_node(sym_node, type="symptom")
            # Link symptom to diagnosis if present in this prescription
            if diag_node:
                G.add_edge(sym_node, diag_node)
                
        for med_node in med_nodes:
            G.add_node(med_node, type="medication")
            # Link medication to diagnosis if present in this prescription
            if diag_node:
                G.add_edge(diag_node, med_node)
            # Link medication to all symptoms present in the same prescription
            for sym_node in symptom_nodes:
                G.add_edge(sym_node, med_node)

    print(f"Built knowledge graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def get_node_embeddings(graph: nx.Graph, dimensions: int = 64, **kwargs) -> Dict[str, np.ndarray]:
    """Generates node embeddings using Node2Vec."""
    node_embeddings = {}
    
    if graph.number_of_nodes() == 0:
        print("Info: Knowledge graph is empty. Skipping Node2Vec.")
        return node_embeddings
        
    # Node2Vec requires nodes to be integers or strings convertible to integers.
    # We'll use string node names directly if possible, but handle potential issues.
    # Ensure all nodes are strings for consistency before passing to Node2Vec
    graph_str_nodes = nx.Graph()
    node_mapping = {} # Keep track if relabeling happens
    has_non_str_nodes = False
    for i, node in enumerate(graph.nodes()):
        if not isinstance(node, str): has_non_str_nodes = True
        str_node = str(node)
        graph_str_nodes.add_node(str_node, **graph.nodes[node]) # Copy attributes
        node_mapping[node] = str_node # Original -> String

    # Add edges using the string representations
    for u, v, data in graph.edges(data=True):
        u_str, v_str = str(u), str(v)
        if u_str in graph_str_nodes and v_str in graph_str_nodes:
             graph_str_nodes.add_edge(u_str, v_str, **data) # Copy edge attributes
        else:
             # This case should ideally not happen if nodes were added correctly
             print(f"Warning: Could not find string representation for edge nodes ({u}, {v}) during graph rebuild.")
             
    target_graph = graph_str_nodes
    print(f"Using graph with {target_graph.number_of_nodes()} string nodes for Node2Vec.")


    # Check connectivity and run on largest component if needed
    if not nx.is_connected(target_graph):
        print("Warning: Graph is not connected. Node2Vec might produce suboptimal embeddings.")
        # Find the largest connected component
        largest_cc_nodes = max(nx.connected_components(target_graph), key=len)
        subgraph = target_graph.subgraph(largest_cc_nodes).copy() # Create a copy to work with
        print(f"Running Node2Vec on the largest connected component ({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges).")
        if subgraph.number_of_nodes() < 2 or subgraph.number_of_edges() == 0:
             print("Warning: Largest connected component is too small or has no edges. Cannot run Node2Vec.")
             return node_embeddings # Return empty dict
        target_graph = subgraph # Use the subgraph for embedding generation

    elif target_graph.number_of_nodes() < 2 or target_graph.number_of_edges() == 0:
         print("Warning: Graph has too few nodes or no edges. Cannot run Node2Vec.")
         return node_embeddings

    try:
        # Initialize Node2Vec
        # Common parameters: walk_length, num_walks, p, q, workers
        node2vec = Node2Vec(
            target_graph, # Use the (potentially largest component) graph with string nodes
            dimensions=dimensions,
            walk_length=kwargs.get("walk_length", 30),
            num_walks=kwargs.get("num_walks", 100), # Reduced for faster execution
            p=kwargs.get("p", 1), # Return parameter
            q=kwargs.get("q", 1), # In-out parameter
            workers=kwargs.get("workers", 2), # Use fewer workers if CPU constrained
            quiet=True # Suppress verbose output
        )

        # Train the model
        # Common parameters: window, min_count, batch_words
        model = node2vec.fit(
            window=kwargs.get("window", 10),
            min_count=kwargs.get("min_count", 1),
            batch_words=kwargs.get("batch_words", 4)
        )

        # Extract embeddings into a dictionary {node_name: embedding_vector}
        # The model.wv contains the embeddings keyed by node names (which are strings)
        for node in target_graph.nodes():
            if node in model.wv:
                 node_embeddings[node] = model.wv[node]
            # else: # Should not happen if min_count=1, but good to be aware
            #     print(f"Warning: Node '{node}' from graph component not found in Word2Vec model vocabulary.")


        print(f"Generated Node2Vec embeddings for {len(node_embeddings)} nodes.")
        
    except Exception as e:
        print(f"Error during Node2Vec processing: {e}")
        # Optionally: log the traceback for detailed debugging
        # import traceback
        # traceback.print_exc()
        return {} # Return empty dict on error

    return node_embeddings

def get_aggregated_embedding(keys: List[str], embeddings: Dict[str, np.ndarray], dim: int) -> np.ndarray:
    """Aggregates embeddings for a list of keys (e.g., symptoms) using averaging."""
    vectors = []
    found_keys = []
    keys_lower = [str(key).lower().strip() for key in keys if isinstance(key, str) and key.strip()] # Clean keys

    for key in keys_lower:
        if key in embeddings:
            vectors.append(embeddings[key])
            found_keys.append(key)
    
    if vectors:
        aggregated_vector = np.mean(vectors, axis=0)
        print(f"Aggregated graph embedding from {len(found_keys)} relevant nodes: {', '.join(found_keys)}")
        return aggregated_vector
    else:
        print("Warning: No relevant nodes found in graph embeddings for aggregation keys.")
        return np.zeros(dim) # Return zero vector if no keys found

def retrieve_relevant_guidelines(query_embedding: np.ndarray, index: faiss.Index, sections: List[str], top_k: int) -> List[str]:
    """Retrieves top_k relevant guideline sections using FAISS."""
    retrieved_guidelines = []
    num_sections = len(sections)
    if index is None:
         print("Warning: FAISS index is not loaded. Cannot retrieve guidelines.")
         return retrieved_guidelines
    if index.ntotal == 0:
        print("Warning: FAISS index is empty. Cannot retrieve guidelines.")
        return retrieved_guidelines
    if num_sections == 0:
        print("Warning: No guideline sections loaded. Cannot retrieve guidelines.")
        return retrieved_guidelines
        
    try:
        # Ensure query embedding is float32 and has the correct shape (1, dim)
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        
        query_embedding_f32 = query_embedding.astype(np.float32)

        # Perform the search
        distances, indices = index.search(query_embedding_f32, k=min(top_k, index.ntotal)) # Search for k or max available

        # Process results
        valid_indices = [i for i in indices[0] if 0 <= i < num_sections] # Filter out invalid indices (e.g., -1) and out-of-bounds
        retrieved_guidelines = [sections[i] for i in valid_indices]
        
        print(f"Retrieved {len(retrieved_guidelines)} guidelines (Top {top_k} search).")
        
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        # Depending on the error, might want to raise it or just return empty list
        
    return retrieved_guidelines

def parse_llm_json_output(llm_raw_output: str) -> Optional[Dict]:
    """Attempts to parse a JSON object from the LLM's raw string output."""
    parsed_output = None
    if not llm_raw_output or not isinstance(llm_raw_output, str):
        print("Warning: LLM raw output is empty or not a string.")
        return None # Cannot parse

    # 1. Try finding JSON within ```json ... ``` blocks
    pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    match = pattern.search(llm_raw_output)
    
    json_string_to_parse = None
    if match:
        json_string_to_parse = match.group(1).strip()
        print("Info: Found JSON block within ``` markers.")
    else:
        # 2. If no block found, check if the entire string might be JSON
        trimmed_output = llm_raw_output.strip()
        # Basic check: starts with { ends with } (might need more robustness)
        if trimmed_output.startswith('{') and trimmed_output.endswith('}'):
             # Attempt to parse the whole string
             json_string_to_parse = trimmed_output
             print("Info: No ``` markers found, attempting to parse entire response as JSON.")
        else:
             print("Warning: Could not find JSON block within ``` markers, and the response doesn't appear to be a raw JSON object.")
             # Optional: Try a more lenient search for '{' ... '}' ? Might be risky.

    # 3. Parse the identified string
    if json_string_to_parse:
        try:
            parsed_output = json.loads(json_string_to_parse)
            
            # Basic structure validation (can be made more rigorous)
            if isinstance(parsed_output, dict) and \
               "recommended_medications" in parsed_output and \
               "rationale" in parsed_output and \
               isinstance(parsed_output.get("recommended_medications"), list):
                print("Info: Successfully parsed LLM output into expected JSON structure.")
            else:
                print("Warning: Parsed JSON does not fully match the expected structure (required keys: 'recommended_medications' (list), 'rationale' (str)). Storing parsed data as is.")
                # Keep the parsed output even if structure is slightly off

        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse extracted JSON string: {e}")
            print(f"--- Extracted String Start ---\n{json_string_to_parse[:500]}...\n--- Extracted String End ---") # Print part of the string for debugging
            # Fallback: Return structure indicating failure
            return { 
                "error": "Failed to parse LLM response as JSON.",
                "parsing_error_message": str(e),
                "raw_response_snippet": json_string_to_parse[:500] + "..." # Store snippet
            }
        except Exception as e:
             print(f"An unexpected error occurred during JSON parsing: {e}")
             return {
                 "error": "Unexpected error during JSON parsing.",
                 "error_message": str(e),
                 "raw_response_snippet": json_string_to_parse[:500] + "..."
             }
    else:
        # Fallback if no JSON string could be identified
         return {
             "error": "Could not identify a JSON block or object in the LLM response.",
             "raw_response": llm_raw_output # Store the full raw response in this case
         }
         
    return parsed_output

def store_input_output(input_data: dict, output_data: dict, output_path: str):
    """Stores the combined input and output data to a JSON file."""
    combined = {
        "input_data_used": input_data,
        "generated_output": output_data # Should contain the parsed recommendation or error info
    }
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Use ensure_ascii=False for broader character support (e.g., drug names)
            json.dump(combined, f, indent=2, ensure_ascii=False) 
        print(f"Stored combined input and output in: {output_path}")
    except Exception as e:
        print(f"Error storing data to {output_path}: {e}")

# ----------------------------
# Main Processing Pipeline
# ----------------------------
if __name__ == "__main__":
    print("--- Starting Auto-Prescription RAG Pipeline ---")

    # --- Configuration ---
    # Construct paths relative to the script location or use absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__)) # Directory of the script
    env_file_path = os.path.join(base_dir, '.env') # Look for .env next to the script
    # Alternatively, specify an absolute path:
    # env_file_path = 'd:/arogo/auto_presc_RAG/.env' 
    
    data_dir = os.path.join(base_dir, 'data')
    patient_data_dir = os.path.join(base_dir, 'patient_data')
    
    index_path = os.path.join(data_dir, 'faiss_index.index')
    sections_path = os.path.join(data_dir, 'sections.txt')
    patient_input_path = os.path.join(patient_data_dir,'patient_input.json')
    past_record_path = os.path.join(patient_data_dir,'past_prescriptions.json') # The file provided by user
    output_path = os.path.join(patient_data_dir,'prescription_output.json')

    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(patient_data_dir, exist_ok=True)

    # --- Load Environment Variables ---
    try:
        nvidia_api_key = load_environment_variables(env_file_path)
    except ValueError as e:
        print(f"Critical Error: {e}")
        exit(1) # Stop execution if API key is missing

    # --- Load Core Data ---
    faiss_index = load_faiss_index(index_path)
    guideline_sections = load_guideline_sections(sections_path)
    patient_input = load_json_file(patient_input_path)
    health_record = load_json_file(past_record_path) # Contains prescriptions and lab reports

    # --- Validate Essential Loaded Data ---
    if faiss_index is None or not guideline_sections:
         print("Warning: FAISS index or guideline sections missing. RAG retrieval will be limited.")
         # Decide if execution should stop or continue with limited context
         # exit(1) # Or just proceed
    if patient_input is None:
         print("Critical Error: Patient input file could not be loaded. Cannot proceed.")
         exit(1)
    # Health record might be optional or partially available
    if health_record is None:
         print("Warning: Past health record file could not be loaded. Proceeding without historical data.")
         health_record = {} # Use empty dict to avoid errors later

    # --- Extract Patient Information ---
    # Provide defaults if keys are missing in patient_input.json
    patient_age = patient_input.get("patient_age", 30)
    patient_gender = patient_input.get("patient_gender", "unknown")
    patient_symptoms = patient_input.get("patient_symptoms", [])
    patient_chronic = patient_input.get("patient_chronic", []) # This is crucial for history extraction

    # Validate extracted types
    if not isinstance(patient_age, int): patient_age = 30; print("Warning: patient_age invalid, using default.")
    if not isinstance(patient_gender, str): patient_gender = "unknown"; print("Warning: patient_gender invalid, using default.")
    if not isinstance(patient_symptoms, list): patient_symptoms = []; print("Warning: patient_symptoms invalid, using empty list.")
    if not isinstance(patient_chronic, list): patient_chronic = []; print("Warning: patient_chronic invalid, using empty list.")

    # --- Extract Data from Health Record ---
    # Extract history relevant to the patient's chronic conditions
    treatment_history = extract_relevant_treatment_history(health_record, patient_chronic)
    lab_reports = extract_lab_reports(health_record)
    
    print("\n--- Extracted Patient Data ---")
    print(f"Age: {patient_age}, Gender: {patient_gender}")
    print(f"Symptoms: {patient_symptoms}")
    print(f"Chronic Conditions: {patient_chronic}")
    print(f"Relevant Treatment History: {treatment_history if treatment_history else 'None found'}")
    print(f"Lab Reports: {lab_reports if lab_reports else 'None found'}")

    # --- Knowledge Graph Processing ---
    print("\n--- Knowledge Graph & Embeddings ---")
    knowledge_graph = build_knowledge_graph(health_record)
    
    # Define Node2Vec parameters (can be tuned)
    n2v_params = {
        "dimensions": 64,
        "walk_length": 30,
        "num_walks": 100, # Fewer walks for faster demo/debug, increase for production
        "workers": 4 
    }
    graph_embeddings = get_node_embeddings(knowledge_graph, **n2v_params)
    
    # Aggregate graph embeddings for current patient's context
    # Use symptoms + chronic conditions as keys for aggregation
    aggregation_keys = patient_symptoms + patient_chronic
    aggregated_graph_embedding = get_aggregated_embedding(
        aggregation_keys, 
        graph_embeddings, 
        n2v_params["dimensions"]
    )

    # --- Feature Engineering & MLP Analysis ---
    print("\n--- Feature Engineering & Historical Analysis ---")
    # Encode current patient features (textual description)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2") # 384 dimensions
    
    # Create a text representation of current state for sentence embedding
    current_features_text = f"Symptoms: {', '.join(patient_symptoms)}. Chronic Conditions: {', '.join(patient_chronic)}."
    current_query_embedding = embed_model.encode([current_features_text], convert_to_numpy=True)
    
    # Fuse embeddings: Concatenate Sentence Embedding + Graph Embedding
    # Ensure graph embedding has the correct shape (dim,)
    if aggregated_graph_embedding.ndim > 1:
        aggregated_graph_embedding = aggregated_graph_embedding.squeeze() # Remove extra dims if any
        
    # Check dimensions before concatenation
    expected_st_dim = 384
    expected_n2v_dim = n2v_params['dimensions']
    if current_query_embedding.shape[1] == expected_st_dim and aggregated_graph_embedding.shape == (expected_n2v_dim,):
         fused_features = np.concatenate([current_query_embedding[0], aggregated_graph_embedding])
         print(f"Fused feature vector created with shape: {fused_features.shape}")
         # Run MLP Decoder for historical analysis
         historical_analysis = run_mlp_decoder(fused_features)
    else:
         print("Warning: Dimension mismatch between sentence embedding and graph embedding. Skipping fusion and MLP analysis.")
         print(f"Sentence Embedding Shape: {current_query_embedding.shape}")
         print(f"Aggregated Graph Embedding Shape: {aggregated_graph_embedding.shape}")
         fused_features = None # Indicate fusion failed
         historical_analysis = "Analysis skipped due to embedding dimension mismatch."
         
    print(f"Historical Analysis from MLP Decoder: {historical_analysis}")

    # --- RAG - Retrieval Step ---
    print("\n--- RAG - Guideline Retrieval ---")
    # Formulate a query suitable for guideline retrieval (might differ from LLM query)
    # Let's use the textual features + chronic conditions for retrieval query
    retrieval_query_text = f"Patient with {', '.join(patient_chronic)} presenting with symptoms: {', '.join(patient_symptoms)}."
    retrieval_query_embedding = embed_model.encode([retrieval_query_text], convert_to_numpy=True)
    
    top_k_retrieval = 3
    retrieved_guidelines = retrieve_relevant_guidelines(retrieval_query_embedding, faiss_index, guideline_sections, top_k_retrieval)

    if retrieved_guidelines:
        print("Retrieved Guideline Snippets:")
        for idx, guideline in enumerate(retrieved_guidelines, start=1):
            print(f"\n--- Guideline {idx} ---\n{guideline[:300]}...") # Print snippet
    else:
        print("No relevant guidelines retrieved.")

    # --- LLM - Generation Step ---
    print("\n--- LLM - Prescription Generation ---")
    # Formulate the final, comprehensive query for the LLM
    llm_query = formulate_query_for_llm(
        patient_age, patient_gender, patient_symptoms, patient_chronic,
        treatment_history, lab_reports, historical_analysis
    )
    print("\nFormatted Patient Information for LLM:")
    print(llm_query)

    # Define the prompt template
    # Moved the detailed JSON instruction into the template for clarity
    prompt_template = PromptTemplate(
        input_variables=["patient_info", "guidelines"],
        template="""
        **Clinical Context:**
        {guidelines}

        **Patient Information:**
        {patient_info}

        **Task:**
        Based *strictly* on the provided **Clinical Context (Guidelines)** and **Patient Information**, generate a prescription recommendation.
        
        **Output Format Instructions:**
        Provide your answer *exclusively* as a single, valid JSON object enclosed in ```json ... ``` code blocks. 
        The JSON object *must* have exactly two top-level keys:
        1.  `recommended_medications`: A list of JSON objects. Each object in this list *must* contain the keys `drug` (string), `dosage` (string), and `instructions` (string).
        2.  `rationale`: A string explaining the reasoning for the recommendations, citing evidence *only* from the provided Clinical Context and Patient Information.

        **Example JSON Output Structure:**
        ```json
        {{
          "recommended_medications": [
            {{"drug": "GuidelineDrugA", "dosage": "X mg daily", "instructions": "Take with food as per guideline Z."}},
            {{"drug": "SymptomReliefDrugB", "dosage": "Y puffs PRN", "instructions": "Use for symptom Q based on patient presentation."}}
          ],
          "rationale": "Recommendation based on Guideline X for condition C observed in the patient. Lab result L supports this. Drug B addresses symptom Q..."
        }}
        ```

        **Generate the JSON object now.**
        """
    )

    # Initialize LLM and Chain
    try:
        nvidia_llm = NvidiaLLM(api_key=nvidia_api_key)
        chain = LLMChain(llm=nvidia_llm, prompt=prompt_template)

        # Prepare input for the chain
        chain_input = {
            "patient_info": llm_query,
            "guidelines": "\n\n".join(retrieved_guidelines) if retrieved_guidelines else "No specific guidelines were retrieved. Base recommendations on patient information and general medical knowledge inferred from your training data, adhering strictly to the requested JSON output format." # Provide fallback instruction
        }
        
        # Invoke the chain
        print("\n--- Calling LLM for Recommendation ---")
        llm_response = chain.invoke(chain_input)
        
        # The actual response string is usually in the 'text' key of the output dict
        llm_raw_output = llm_response.get('text', '') 
        
        print("\n--- Raw LLM Output Received ---")
        print(llm_raw_output) # Print the raw output for debugging

    except Exception as e:
        print(f"\nCritical Error during LLM chain execution: {e}")
        llm_raw_output = f"Error during LLM call: {e}" # Store error message
        # Decide if script should exit or store the error
        # exit(1)


    # --- Parse LLM Output ---
    print("\n--- Parsing LLM Response ---")
    parsed_llm_output = parse_llm_json_output(llm_raw_output)

    # Prepare final structured output (either parsed data or error info)
    if parsed_llm_output and "error" not in parsed_llm_output:
        final_output_data = {"prescription_recommendation": parsed_llm_output}
        print("\n--- Final Parsed Recommendation ---")
        print(json.dumps(final_output_data, indent=2, ensure_ascii=False))
    else:
        # Handle parsing errors or LLM errors
        final_output_data = {"prescription_recommendation_error": parsed_llm_output if parsed_llm_output else {"error": "LLM response was empty or generation failed.", "raw_response": llm_raw_output} }
        print("\n--- Error in Final Recommendation ---")
        print(json.dumps(final_output_data, indent=2, ensure_ascii=False))

# --- Store Results ---
    print("\n--- Storing Results ---")
    # Consolidate all the input information used for the final generation
    input_data_summary = {
        "patient_input_file": patient_input_path,
        "health_record_file": past_record_path,
        "patient_age": patient_age,
        "patient_gender": patient_gender,
        "patient_symptoms": patient_symptoms,
        "patient_chronic_conditions": patient_chronic,
        "extracted_treatment_history": treatment_history,
        "extracted_lab_reports": lab_reports,
        "graph_nodes": knowledge_graph.number_of_nodes(),
        "graph_edges": knowledge_graph.number_of_edges(),
        "node2vec_embeddings_generated": len(graph_embeddings),
        # --- FIX HERE ---
        "aggregated_graph_embedding_used": bool(aggregated_graph_embedding.any()), # Convert numpy.bool_ to bool
        # --- END FIX ---
        "mlp_historical_analysis": historical_analysis,
        "retrieved_guideline_count": len(retrieved_guidelines),
        "llm_query_details": llm_query, # The formatted info sent to LLM
        "retrieved_guideline_snippets": [g[:100]+"..." for g in retrieved_guidelines] 
    }
    
    store_input_output(input_data_summary, final_output_data, output_path)

    print("\n--- Pipeline Finished ---")