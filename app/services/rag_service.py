from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import networkx as nx
from node2vec import Node2Vec
from app.core.config import settings
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from app.services.nvidia_llm import NvidiaLLM

logger = logging.getLogger(__name__)

class MLPDecoder(nn.Module):
    """Simple MLP for processing fused features."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

class RAGService:
    def __init__(self):
        # Initialize models and components
        self.embed_model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL)
        self.faiss_index = self._load_faiss_index()
        self.guideline_sections = self._load_guideline_sections()
        self.llm = NvidiaLLM(api_key=settings.NVIDIA_API_KEY)
        self.mlp_decoder = MLPDecoder(448, 128, 1)  # 384 (ST) + 64 (Node2Vec)
        
    def _load_faiss_index(self) -> Optional[faiss.Index]:
        """Loads the FAISS index from disk."""
        try:
            return faiss.read_index(settings.FAISS_INDEX_PATH)
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            return None

    def _load_guideline_sections(self) -> List[str]:
        """Loads guideline sections from file."""
        try:
            with open(settings.GUIDELINE_SECTIONS_PATH, 'r', encoding='utf-8') as f:
                return [sec.strip() for sec in f.read().split("\n\n") if sec.strip()]
        except Exception as e:
            logger.error(f"Error loading guideline sections: {str(e)}")
            return []

    async def process_prescription(self, file_url: str) -> Optional[Dict]:
        """
        Process a prescription and generate recommendations
        """
        try:
            # Extract information from the prescription
            # This would involve OCR or similar processing
            # For now, we'll use dummy data
            prescription_info = {
                "diagnosis": "Hypertension",
                "symptoms": ["High blood pressure", "Headache"],
                "current_medications": ["Amlodipine 5mg"]
            }

            # Create knowledge graph
            G = self._build_knowledge_graph(prescription_info)
            
            # Get node embeddings
            node_embeddings = self._get_node_embeddings(G)
            
            # Get aggregated embedding
            agg_embedding = self._get_aggregated_embedding(
                prescription_info["symptoms"],
                node_embeddings,
                64
            )

            # Get sentence embedding
            text = f"Symptoms: {', '.join(prescription_info['symptoms'])}"
            sentence_embedding = self.embed_model.encode([text])[0]

            # Fuse embeddings
            fused_features = np.concatenate([sentence_embedding, agg_embedding])

            # Get historical analysis
            historical_analysis = self._run_mlp_decoder(fused_features)

            # Retrieve relevant guidelines
            guidelines = self._retrieve_guidelines(sentence_embedding)

            # Generate recommendations using LLM
            recommendations = await self._generate_recommendations(
                prescription_info,
                guidelines,
                historical_analysis
            )

            return recommendations

        except Exception as e:
            logger.error(f"Error processing prescription: {str(e)}")
            return None

    def _build_knowledge_graph(self, prescription_info: Dict) -> nx.Graph:
        """Builds a knowledge graph from prescription information."""
        G = nx.Graph()
        
        # Add nodes and edges based on prescription info
        diagnosis = prescription_info["diagnosis"].lower()
        G.add_node(diagnosis, type="diagnosis")
        
        for symptom in prescription_info["symptoms"]:
            symptom = symptom.lower()
            G.add_node(symptom, type="symptom")
            G.add_edge(symptom, diagnosis)
            
        for med in prescription_info.get("current_medications", []):
            med = med.lower()
            G.add_node(med, type="medication")
            G.add_edge(med, diagnosis)
            
        return G

    def _get_node_embeddings(self, graph: nx.Graph) -> Dict[str, np.ndarray]:
        """Generates node embeddings using Node2Vec."""
        if graph.number_of_nodes() == 0:
            return {}
            
        try:
            node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=100)
            model = node2vec.fit(window=10, min_count=1)
            
            return {node: model.wv[node] for node in graph.nodes()}
        except Exception as e:
            logger.error(f"Error generating node embeddings: {str(e)}")
            return {}

    def _get_aggregated_embedding(self, keys: List[str], embeddings: Dict[str, np.ndarray], dim: int) -> np.ndarray:
        """Aggregates embeddings for a list of keys."""
        vectors = []
        for key in keys:
            key = key.lower()
            if key in embeddings:
                vectors.append(embeddings[key])
                
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(dim)

    def _run_mlp_decoder(self, features: np.ndarray) -> str:
        """Runs the MLP decoder for historical analysis."""
        try:
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                output = self.mlp_decoder(x)
                score = output.item()
                
                if score > 0.5:
                    return "Historical treatments appear potentially effective"
                elif score < -0.5:
                    return "Historical treatments appear potentially less effective"
                else:
                    return "Historical treatment effectiveness is inconclusive"
        except Exception as e:
            logger.error(f"Error running MLP decoder: {str(e)}")
            return "Analysis skipped due to error"

    def _retrieve_guidelines(self, query_embedding: np.ndarray) -> List[str]:
        """Retrieves relevant guidelines using FAISS."""
        if self.faiss_index is None or not self.guideline_sections:
            return []
            
        try:
            distances, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype('float32'),
                k=3
            )
            
            return [self.guideline_sections[i] for i in indices[0] if i < len(self.guideline_sections)]
        except Exception as e:
            logger.error(f"Error retrieving guidelines: {str(e)}")
            return []

    async def _generate_recommendations(
        self,
        prescription_info: Dict,
        guidelines: List[str],
        historical_analysis: str
    ) -> Dict:
        """Generates recommendations using the LLM."""
        try:
            prompt_template = PromptTemplate(
                input_variables=["patient_info", "guidelines"],
                template="""
                **Clinical Context:**
                {guidelines}

                **Patient Information:**
                {patient_info}

                **Task:**
                Based on the provided Clinical Context and Patient Information, generate a prescription recommendation.
                
                **Output Format:**
                Provide your answer as a JSON object with:
                1. `recommended_medications`: List of objects with `drug`, `dosage`, and `instructions`
                2. `rationale`: String explaining the reasoning
                """
            )

            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            
            patient_info = f"""
            Diagnosis: {prescription_info['diagnosis']}
            Symptoms: {', '.join(prescription_info['symptoms'])}
            Current Medications: {', '.join(prescription_info.get('current_medications', []))}
            Historical Analysis: {historical_analysis}
            """
            
            response = await chain.arun(
                patient_info=patient_info,
                guidelines="\n\n".join(guidelines)
            )
            
            # Parse the response and return the recommendations
            # This is a simplified version - you'll need to add proper parsing
            return {
                "prescription_type": "NEW",
                "diagnosis": prescription_info["diagnosis"],
                "symptoms": prescription_info["symptoms"],
                "medicines": [
                    {
                        "medicine_name": "Sample Medicine",
                        "dosage": "10mg",
                        "frequency": "Once daily",
                        "instructions": "Take with food",
                        "duration": 30,
                        "chemical_composition": "Sample composition"
                    }
                ],
                "lab_tests": []
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return None 