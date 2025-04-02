import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk

# Download NLTK data (only the first time)
nltk.download("punkt")
nltk.download("punkt_tab")  # Added download for punkt_tab resource

from nltk.tokenize import sent_tokenize

def extract_text_from_pdf(pdf_path):
    text_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)
    return "\n".join(text_pages)

def split_into_sections(text, max_chunk_length=500):
    # Split text into chunks using sentence tokenization
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_embeddings(sections, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sections, convert_to_tensor=False)
    return np.array(embeddings, dtype=np.float32)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

if __name__ == "__main__":
    pdf_path = 'd:/arogo/auto_presc_RAG/data/Bookshelf_NBK209539.pdf'
    
    # 1. Extract text from the PDF file
    text = extract_text_from_pdf(pdf_path)
    
    # 2. Preprocess text: Segment text into manageable sections/chunks
    sections = split_into_sections(text, max_chunk_length=500)
    print("Number of sections extracted:", len(sections))
    
    # 3. Convert text sections to embeddings using a Sentence Transformer model
    embeddings = generate_embeddings(sections)
    
    # 4. Build a FAISS index for fast vector search
    index = build_faiss_index(embeddings)
    print("FAISS index built with", index.ntotal, "vectors.")

    # Optionally, save the FAISS index and sections mapping for retrieval
    faiss.write_index(index, 'd:/arogo/auto_presc_RAG/faiss_index.index')
    with open('d:/arogo/auto_presc_RAG/sections.txt', 'w', encoding='utf-8') as f:
        for section in sections:
            f.write(section + "\n\n")
    
    print("Index and sections saved; PDF text is now preprocessed and indexed for RAG integration.")