import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from llama_cpp import Llama

# Load environment variables from your local .env file
load_dotenv()

# --- CONFIGURATION (Environment-Driven for Portability) ---
DB_DIR = os.getenv("DB_DIR", "./chroma_db")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-base-en-v1.5")
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", -1))  # Default to -1 for Mac/GPU acceleration, or 0 for CPU
N_THREADS = int(os.getenv("N_THREADS", 4))

print("🔄 Initializing RAG Engine components...")

# 1. Load the Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 2. Load the Persistent Chroma Database
if not os.path.exists(DB_DIR):
    print(f"❌ Error: Database directory {DB_DIR} not found. Run ingest.py first!")
    exit(1)

print(f"📂 Loading persistent vector database from {DB_DIR}...")
vectordb = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embedding_model
)

# Create retriever (fetching top 5 chunks)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# 3. Load the Local Mistral LLM
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model file not found at {MODEL_PATH}. Please check your models folder.")
    exit(1)

print(f"🧠 Loading local Mistral model from {MODEL_PATH} (GPU Layers: {N_GPU_LAYERS})...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=N_THREADS,
    n_gpu_layers=N_GPU_LAYERS,
    verbose=False
)
print("✅ Local LLM loaded successfully!")

# --- STEP 43: Define Strict JSON Schema for Frontend/Database ---
class MedicalResponseSchema(BaseModel):
    conditions: list[str] = Field(description="List of potential medical conditions matching the symptoms.")
    medications: list[str] = Field(description="Recommended medications or essential drugs referencing standard guidelines.")
    treatments: list[str] = Field(description="Actionable treatment plans or lifestyle adjustments.")

def generate_response(prompt, max_tokens=400):
    """Sends the formatted prompt to the local Llama model."""
    model_output = llm(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.1,    # Low temperature keeps answers factual and grounded
        top_p=0.95,
        top_k=50,
        stop=["[/INST]", "</s>", "\n\n\n"]
    )
    return model_output['choices'][0]['text'].strip()

def build_prompt(query):
    """Retrieves relevant chunks and formats them into a strict JSON-enforced RAG prompt."""
    docs = retriever.invoke(query)
    
    # Extract context and track sources for citations
    context_blocks = []
    sources = []
    
    for i, doc in enumerate(docs):
        context_blocks.append(f"Source [{i+1}]: {doc.page_content}")
        src_name = doc.metadata.get("source", "Unknown PDF")
        page_num = doc.metadata.get("page", "Unknown")
        sources.append(f"- {src_name} (Page {page_num})")
        
    context = "\n\n".join(context_blocks)
    json_schema_example = json.dumps(MedicalResponseSchema.model_json_schema(), indent=2)
    
    prompt = f"""[INST] You are an expert clinical AI assistant. Your task is to analyze the patient symptom input using ONLY the provided reference text below. 

You must output your final response strictly as a valid JSON object matching this schema:
{json_schema_example}

Context:
{context}

Patient Symptom / Question: {query}
[/INST]
JSON Output:"""

    return prompt, list(set(sources))

def ask_medical_ai(query):
    print(f"\n🔍 Searching vector database for: '{query}'...")
    prompt, sources = build_prompt(query)
    
    print("🤖 Generating structured JSON response using local Mistral...")
    raw_answer = generate_response(prompt)
    
    # Parse and validate output JSON
    try:
        cleaned_output = raw_answer
        if "```json" in cleaned_output:
            cleaned_output = cleaned_output.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_output:
            cleaned_output = cleaned_output.split("```")[1].split("```")[0].strip()
            
        parsed_json = json.loads(cleaned_output)
        final_output = parsed_json
    except json.JSONDecodeError:
        final_output = {
            "error": "Failed to parse LLM output as strict JSON",
            "raw_output": raw_answer
        }
        
    print("\n" + "="*50)
    print("📋 STRUCTURED JSON ANSWER:")
    print("="*50)
    print(json.dumps(final_output, indent=2))
    print("\n" + "="*50)
    print("📚 SOURCES & CITATIONS:")
    print("="*50)
    for src in sources:
        print(src)
    print("="*50 + "\n")
    return final_output

if __name__ == "__main__":
    # Test query to verify your entire RAG pipeline!
    sample_query = "Patient experiencing severe headache, high fever, and body aches."
    ask_medical_ai(sample_query)