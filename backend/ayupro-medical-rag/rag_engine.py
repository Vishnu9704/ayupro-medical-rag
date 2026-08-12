import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from llama_cpp import Llama

# --- CONFIGURATION ---
DB_DIR = "./chroma_db"
MODEL_PATH = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"

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

print(f"🧠 Loading local Mistral model from {MODEL_PATH} (This may take a moment)...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,      # Increase thread count slightly for faster local generation
    verbose=False
)
print("✅ Local LLM loaded successfully!")

def generate_response(prompt, max_tokens=300):
    """Sends the formatted prompt to the local Llama model."""
    model_output = llm(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.1,    # Low temperature keeps answers factual and grounded
        top_p=0.95,
        top_k=50,
        stop=["Question:", "\n\n\n"]
    )
    return model_output['choices'][0]['text'].strip()

def build_prompt(query):
    """Retrieves relevant chunks and formats them into a strict RAG prompt."""
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
    
    prompt = f"""[INST] You are an expert medical AI assistant. Answer the question using ONLY the factual context provided below. If the answer cannot be found in the context, state clearly that you do not know. Do not hallucinate or make up medical information.

Context:
{context}

Question: {query}
[/INST]
Answer:"""

    return prompt, list(set(sources))

def ask_medical_ai(query):
    print(f"\n🔍 Searching vector database for: '{query}'...")
    prompt, sources = build_prompt(query)
    
    print("🤖 Generating medical response using local Mistral...")
    answer = generate_response(prompt)
    
    print("\n" + "="*50)
    print("📋 ANSWER:")
    print("="*50)
    print(answer)
    print("\n" + "="*50)
    print("📚 SOURCES & CITATIONS:")
    print("="*50)
    for src in sources:
        print(src)
    print("="*50 + "\n")

if __name__ == "__main__":
    # Test query to verify your entire RAG pipeline!
    sample_query = "What are the essential guidelines for medicine usage?"
    ask_medical_ai(sample_query)